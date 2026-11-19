use anyhow::{anyhow, Result};
use chrono::{DateTime, Utc};
use futures_util::StreamExt;
use reqwest::{
    header::{HeaderMap, RETRY_AFTER},
    RequestBuilder, Response, StatusCode,
};
use serde_json::Value;
use std::time::{Duration, Instant, SystemTime, UNIX_EPOCH};

const DEFAULT_MAX_ATTEMPTS: usize = 6;
const DEFAULT_MAX_DURATION_SECS: u64 = 300;
const MAX_CONFIGURED_ATTEMPTS: usize = 10;
const MAX_CONFIGURED_DURATION_SECS: u64 = 300;
const BASE_DELAY_MS: u64 = 250;
const MAX_BACKOFF_MS: u64 = 8_000;
const MAX_ERROR_RESPONSE_BYTES: usize = 64 * 1024;
const MAX_ERROR_DIAGNOSTIC_BYTES: usize = MAX_ERROR_RESPONSE_BYTES + 512;

#[derive(Clone, Copy)]
pub(super) struct RetryPolicy {
    max_attempts: usize,
    max_duration: Duration,
    base_delay: Duration,
    max_backoff: Duration,
}

impl RetryPolicy {
    pub(super) fn from_env() -> Self {
        let max_attempts = std::env::var("AICHAT_HTTP_RETRY_MAX_ATTEMPTS")
            .ok()
            .and_then(|value| value.parse::<usize>().ok())
            .unwrap_or(DEFAULT_MAX_ATTEMPTS)
            .clamp(1, MAX_CONFIGURED_ATTEMPTS);
        let max_duration = std::env::var("AICHAT_HTTP_RETRY_MAX_DURATION_SECS")
            .ok()
            .and_then(|value| value.parse::<u64>().ok())
            .unwrap_or(DEFAULT_MAX_DURATION_SECS)
            .min(MAX_CONFIGURED_DURATION_SECS);

        Self {
            max_attempts,
            max_duration: Duration::from_secs(max_duration),
            base_delay: Duration::from_millis(BASE_DELAY_MS),
            max_backoff: Duration::from_millis(MAX_BACKOFF_MS),
        }
    }
}

pub(super) struct ResponseBody {
    pub(super) bytes: Vec<u8>,
    pub(super) truncated: bool,
    read_error: Option<String>,
}

pub(super) async fn read_response_body(response: Response) -> ResponseBody {
    let status = response.status();
    let mut stream = response.bytes_stream();
    let mut bytes = Vec::new();
    let mut truncated = false;
    let mut read_error = None;

    while let Some(chunk) = stream.next().await {
        let chunk = match chunk {
            Ok(chunk) => chunk,
            Err(error) => {
                read_error = Some(format!(
                    "Failed to read provider error response (HTTP {}): {}",
                    status.as_u16(),
                    error.without_url()
                ));
                break;
            }
        };
        let remaining = MAX_ERROR_RESPONSE_BYTES - bytes.len();
        if chunk.len() > remaining {
            bytes.extend_from_slice(&chunk[..remaining]);
            truncated = true;
            break;
        }
        bytes.extend_from_slice(&chunk);
    }

    ResponseBody {
        bytes,
        truncated,
        read_error,
    }
}

fn sanitized_body(body: &ResponseBody) -> String {
    let raw = String::from_utf8_lossy(&body.bytes);
    let text = match serde_json::from_slice::<Value>(&body.bytes) {
        Ok(mut data) => {
            let original = data.clone();
            redact_json_secrets(&mut data);
            if data == original {
                raw.to_string()
            } else {
                serde_json::to_string(&data).unwrap_or_else(|_| redact_secret_text(&raw))
            }
        }
        Err(_) => redact_secret_text(&raw),
    };
    let mut text = text;
    if body.truncated {
        text.push_str(&format!(
            "\n[response body truncated after {MAX_ERROR_RESPONSE_BYTES} bytes]"
        ));
    }
    if let Some(error) = &body.read_error {
        text.push_str(&format!("\n[response body incomplete: {error}]"));
    }
    text
}

fn bound_diagnostic(mut text: String) -> String {
    if text.len() > MAX_ERROR_DIAGNOSTIC_BYTES {
        let mut end = MAX_ERROR_DIAGNOSTIC_BYTES - 32;
        while !text.is_char_boundary(end) {
            end -= 1;
        }
        text.truncate(end);
        text.push_str(" [diagnostic truncated]");
    }
    text
}

pub(super) fn response_body_text(body: &ResponseBody) -> String {
    bound_diagnostic(sanitized_body(body))
}

pub(super) fn format_stream_error(event: &str, payload: &str) -> String {
    let body = ResponseBody {
        bytes: payload.as_bytes().to_vec(),
        truncated: false,
        read_error: None,
    };
    let diagnostic = response_body_text(&body);
    if event.is_empty() {
        format!("Provider reported an in-stream error: {diagnostic}")
    } else {
        format!("Provider reported an in-stream error ({event}): {diagnostic}")
    }
}

pub(super) fn format_response_error(status: StatusCode, body: &ResponseBody) -> String {
    response_error_message(status, body)
}

fn response_error_message(status: StatusCode, body: &ResponseBody) -> String {
    let body = response_body_text(body);
    let body = if body.trim().is_empty() {
        "empty response body"
    } else {
        &body
    };
    bound_diagnostic(format!(
        "Provider returned HTTP {}: {body}",
        status.as_u16()
    ))
}

pub(super) struct RetryState {
    policy: RetryPolicy,
    started: Instant,
    attempt: usize,
    last_diagnostic: Option<String>,
    last_transport_diagnostic: Option<String>,
}

impl RetryState {
    pub(super) fn new(policy: RetryPolicy) -> Self {
        Self {
            policy,
            started: Instant::now(),
            attempt: 1,
            last_diagnostic: None,
            last_transport_diagnostic: None,
        }
    }

    pub(super) fn retry_response(
        &mut self,
        status: StatusCode,
        headers: &HeaderMap,
        body: &ResponseBody,
        allow_retry: bool,
    ) -> Result<Duration> {
        self.last_diagnostic = Some(response_error_message(status, body));
        self.last_transport_diagnostic = None;

        let required_delay = retry_delay(headers, &body.bytes)
            .map_err(|reason| self.with_last_diagnostic(anyhow!(reason)))?;

        if let Some(reason) = permanent_failure(status, &body.bytes) {
            return Err(self.with_last_diagnostic(anyhow!(reason)));
        }
        if !allow_retry || !is_retryable_status(status) {
            return Err(self.with_last_diagnostic(anyhow!(
                "HTTP {} response was not retried",
                status.as_u16()
            )));
        }
        if body.truncated || body.read_error.is_some() {
            return Err(self.with_last_diagnostic(anyhow!(
                "HTTP {} response body is incomplete; provider retry instructions cannot be determined, so retrying was refused",
                status.as_u16()
            )));
        }

        self.schedule(&format!("HTTP {}", status.as_u16()), required_delay)
            .map_err(|error| self.with_last_diagnostic(error))
    }

    pub(super) fn retry_connection(&mut self, error: reqwest::Error) -> Result<Duration> {
        self.last_transport_diagnostic = Some(format!(
            "Connection failure before a provider HTTP response was received: {}",
            error.without_url()
        ));
        self.schedule("connection failure before request submission", None)
            .map_err(|error| self.with_last_diagnostic(error))
    }

    pub(super) fn transport_error(
        &mut self,
        context: &str,
        error: reqwest::Error,
    ) -> anyhow::Error {
        let transport_diagnostic = format!("{context}: {}", error.without_url());
        self.last_transport_diagnostic = Some(transport_diagnostic.clone());
        self.with_last_diagnostic(anyhow!("{transport_diagnostic}"))
    }

    pub(super) fn with_last_diagnostic(&self, error: anyhow::Error) -> anyhow::Error {
        let message = error.to_string();
        let mut diagnostics = Vec::new();
        for diagnostic in [&self.last_diagnostic, &self.last_transport_diagnostic]
            .into_iter()
            .flatten()
        {
            if !message.contains(diagnostic) {
                diagnostics.push(diagnostic.as_str());
            }
        }
        if diagnostics.is_empty() {
            error
        } else {
            diagnostics.push(&message);
            anyhow!(diagnostics.join("\n"))
        }
    }

    fn schedule(&mut self, reason: &str, requested_delay: Option<Duration>) -> Result<Duration> {
        if self.attempt >= self.policy.max_attempts {
            anyhow::bail!(
                "HTTP recovery exhausted after {} attempts ({reason})",
                self.attempt
            );
        }

        let delay = self
            .backoff_delay()
            .max(requested_delay.unwrap_or_default());
        let remaining = self
            .policy
            .max_duration
            .saturating_sub(self.started.elapsed());
        if delay > remaining {
            anyhow::bail!(
                "HTTP recovery stopped after {} attempts because the next wait ({:.2}s) does not fit the remaining retry admission budget ({:.2}s); {reason}",
                self.attempt,
                delay.as_secs_f64(),
                remaining.as_secs_f64()
            );
        }

        self.attempt += 1;
        log::warn!(
            "Provider request failed ({reason}); retrying attempt {} of {} in {:.2}s",
            self.attempt,
            self.policy.max_attempts,
            delay.as_secs_f64()
        );
        Ok(delay)
    }

    fn backoff_delay(&self) -> Duration {
        let exponent = self.attempt.saturating_sub(1).min(16) as u32;
        let base_ms = self.policy.base_delay.as_millis() as u64;
        let upper_ms = base_ms
            .saturating_mul(1_u64 << exponent)
            .min(self.policy.max_backoff.as_millis() as u64);
        let lower_ms = upper_ms / 2;
        let jitter_range = upper_ms - lower_ms + 1;
        let jitter = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .unwrap_or_default()
            .subsec_nanos() as u64
            % jitter_range;

        Duration::from_millis(lower_ms + jitter)
    }
}

pub(crate) async fn send_with_retry(builder: RequestBuilder) -> Result<Response> {
    send_with_retry_policy(builder, RetryPolicy::from_env()).await
}

async fn send_with_retry_policy(builder: RequestBuilder, policy: RetryPolicy) -> Result<Response> {
    let allow_retry = builder.try_clone().is_some();
    let mut builder = Some(builder);
    let mut state = RetryState::new(policy);

    loop {
        let request = if allow_retry {
            builder
                .as_ref()
                .and_then(RequestBuilder::try_clone)
                .ok_or_else(|| anyhow!("Request could not be cloned for a retry"))?
        } else {
            builder
                .take()
                .ok_or_else(|| anyhow!("Non-cloneable request was already sent"))?
        };

        let response = match request.send().await {
            Ok(response) => response,
            Err(error) if allow_retry && error.is_connect() => {
                let delay = state.retry_connection(error)?;
                tokio::time::sleep(delay).await;
                continue;
            }
            Err(error) if error.is_timeout() => {
                return Err(state.transport_error(
                    "Provider request timed out; no HTTP response was received and the request was not replayed because the provider may have received it",
                    error,
                ));
            }
            Err(error) => {
                return Err(state.transport_error(
                    "Transport failure; no provider HTTP response was received",
                    error,
                ));
            }
        };

        if response.status().is_success() {
            return Ok(response);
        }

        let status = response.status();
        let headers = response.headers().clone();
        let body = read_response_body(response).await;
        let delay = state.retry_response(status, &headers, &body, allow_retry)?;
        tokio::time::sleep(delay).await;
    }
}

pub(super) fn is_retryable_status(status: StatusCode) -> bool {
    matches!(
        status.as_u16(),
        408 | 425 | 429 | 500 | 502 | 503 | 504 | 529
    )
}

fn parse_retry_after_header(headers: &HeaderMap) -> Option<Duration> {
    let value = headers.get(RETRY_AFTER)?.to_str().ok()?;
    if let Ok(seconds) = value.parse::<u64>() {
        return Some(Duration::from_secs(seconds));
    }

    let retry_at = DateTime::parse_from_rfc2822(value).ok()?.timestamp();
    Some(Duration::from_secs(
        retry_at.saturating_sub(Utc::now().timestamp()).max(0) as u64,
    ))
}

fn parse_google_duration(value: &str) -> Option<Duration> {
    let value = value.strip_suffix('s')?;
    let (seconds, fraction) = value.split_once('.').unwrap_or((value, ""));
    if seconds.is_empty()
        || !seconds.bytes().all(|byte| byte.is_ascii_digit())
        || !fraction.bytes().all(|byte| byte.is_ascii_digit())
        || (value.contains('.') && fraction.is_empty())
        || fraction.len() > 9
    {
        return None;
    }

    let seconds = seconds.parse::<u64>().ok()?;
    if seconds > 315_576_000_000 {
        return None;
    }
    let nanos = if fraction.is_empty() {
        0
    } else {
        format!("{fraction:0<9}").parse::<u32>().ok()?
    };
    if seconds == 315_576_000_000 && nanos > 0 {
        return None;
    }
    Duration::from_secs(seconds).checked_add(Duration::from_nanos(u64::from(nanos)))
}

fn google_error_details(data: &Value) -> Vec<&Value> {
    if let Some(details) = data
        .pointer("/error/details")
        .or_else(|| data.get("details"))
        .and_then(Value::as_array)
    {
        return details.iter().collect();
    }

    let Some(envelopes) = data.as_array() else {
        return Vec::new();
    };
    let wrapped_details = envelopes
        .iter()
        .filter_map(|envelope| envelope.pointer("/error/details").and_then(Value::as_array))
        .flatten()
        .collect::<Vec<_>>();

    if wrapped_details.is_empty() {
        envelopes.iter().collect()
    } else {
        wrapped_details
    }
}

fn google_retry_info(body: &[u8]) -> Result<Option<Duration>, &'static str> {
    let Ok(data) = serde_json::from_slice::<Value>(body) else {
        return Ok(None);
    };

    let mut delay = None;
    for detail in google_error_details(&data) {
        if detail.get("@type").and_then(Value::as_str)
            != Some("type.googleapis.com/google.rpc.RetryInfo")
        {
            continue;
        }

        let retry_delay = detail
            .get("retryDelay")
            .and_then(Value::as_str)
            .and_then(parse_google_duration)
            .ok_or("recognized google.rpc.RetryInfo has a missing or malformed retryDelay")?;
        delay = Some(delay.map_or(retry_delay, |current: Duration| current.max(retry_delay)));
    }

    Ok(delay)
}

fn retry_delay(headers: &HeaderMap, body: &[u8]) -> Result<Option<Duration>, &'static str> {
    let header_delay = parse_retry_after_header(headers);
    let google_delay = google_retry_info(body)?;
    Ok(header_delay.into_iter().chain(google_delay).max())
}

fn google_has_daily_quota(data: &Value) -> bool {
    const DAILY_QUOTA_IDS: &[&str] = &[
        "GenerateRequestsPerDayPerProjectPerModel-FreeTier",
        "GenerateContentInputTokensPerModelPerDay-FreeTier",
    ];
    // Gemini quota failures expose quota IDs such as GenerateRequestsPerDayPerProjectPerModel-FreeTier.
    google_error_details(data).into_iter().any(|detail| {
        if detail.get("@type").and_then(Value::as_str)
            != Some("type.googleapis.com/google.rpc.QuotaFailure")
        {
            return false;
        }
        detail
            .get("violations")
            .and_then(Value::as_array)
            .into_iter()
            .flatten()
            .filter_map(|violation| violation.get("quotaId").and_then(Value::as_str))
            .any(|quota_id| DAILY_QUOTA_IDS.contains(&quota_id))
    })
}

fn permanent_failure(status: StatusCode, body: &[u8]) -> Option<&'static str> {
    if status != StatusCode::TOO_MANY_REQUESTS {
        return None;
    }
    let data = serde_json::from_slice::<Value>(body).ok()?;
    const QUOTA_CODES: &[&str] = &[
        "insufficient_quota",
        "billing_hard_limit_reached",
        "credit_balance_exhausted",
        "organization_usage_limit_exceeded",
        "organization_spend_limit_exceeded",
        "project_spend_limit_exceeded",
    ];
    const AUTH_CODES: &[&str] = &["invalid_api_key", "authentication_error", "invalid_grant"];
    let code_paths = [
        "/error/code",
        "/error/type",
        "/error",
        "/code",
        "/type",
        "/Error/Code",
        "/__type",
        "/errors/0/code",
        "/errors/0/reason",
    ];

    let has_code = |codes: &[&str]| {
        code_paths.iter().any(|path| {
            data.pointer(path)
                .and_then(Value::as_str)
                .is_some_and(|actual| codes.iter().any(|code| actual.eq_ignore_ascii_case(code)))
        })
    };

    if has_code(QUOTA_CODES) {
        return Some("The provider reports a permanent billing or usage limit; check its billing and quota settings");
    }
    if has_code(AUTH_CODES) {
        return Some(
            "The provider rejected the request credentials; check the authentication configuration",
        );
    }
    if google_has_daily_quota(&data) {
        return Some("Google reports a documented daily quota violation; retrying within the configured window is unlikely to help");
    }
    None
}

fn redact_json_secrets(value: &mut Value) {
    match value {
        Value::Object(object) => {
            for (key, value) in object.iter_mut() {
                let key = key
                    .chars()
                    .filter(|value| value.is_ascii_alphanumeric())
                    .flat_map(char::to_lowercase)
                    .collect::<String>();
                let sensitive = matches!(
                    key.as_str(),
                    "apikey" | "authorization" | "token" | "credential"
                ) || key.contains("apikey")
                    || key.contains("accesstoken")
                    || key.contains("accesskey")
                    || key.contains("refreshtoken")
                    || key.contains("sessiontoken")
                    || key.contains("secret")
                    || key.contains("password")
                    || key.contains("credential")
                    || key.contains("privatekey")
                    || key.contains("signature");
                if sensitive {
                    *value = Value::String("[REDACTED]".to_string());
                } else {
                    redact_json_secrets(value);
                }
            }
        }
        Value::Array(values) => {
            for value in values {
                redact_json_secrets(value);
            }
        }
        Value::String(value) => *value = redact_secret_text(value),
        _ => {}
    }
}

fn redact_secret_text(value: &str) -> String {
    let mut value = redact_auth_scheme(value, "bearer");
    value = redact_auth_scheme(&value, "basic");
    value = redact_assignment_values(&value);
    for prefix in ["sk-", "AIza", "AKIA", "xoxb-", "xoxp-", "xoxa-"] {
        value = redact_prefixed_token(&value, prefix);
    }
    value
        .chars()
        .filter(|value| !value.is_control() || matches!(*value, '\n' | '\r' | '\t'))
        .collect()
}

fn redact_assignment_values(input: &str) -> String {
    const MARKERS: &[&str] = &[
        "authorization",
        "access_token",
        "access-token",
        "access token",
        "accesskey",
        "secret_access_key",
        "secret-access-key",
        "refresh_token",
        "refresh-token",
        "refresh token",
        "session_token",
        "session-token",
        "session token",
        "client_secret",
        "client-secret",
        "client secret",
        "private_key",
        "private-key",
        "private key",
        "secret_key",
        "secret-key",
        "secret key",
        "api_key",
        "api-key",
        "api key",
        "x-api-key",
        "x_api_key",
        "apikey",
        "password",
        "credential",
        "secret",
        "token",
    ];
    let mut value = input.to_string();
    let mut cursor = 0;

    loop {
        let lower = value.to_ascii_lowercase();
        let found = MARKERS
            .iter()
            .filter_map(|marker| {
                lower[cursor..]
                    .find(marker)
                    .map(|offset| (cursor + offset, *marker))
            })
            .filter(|(index, _)| {
                *index == 0 || !lower.as_bytes()[*index - 1].is_ascii_alphanumeric()
            })
            .min_by_key(|(index, _)| *index);
        let Some((start, marker)) = found else {
            break;
        };

        let mut position = start + marker.len();
        while position < value.len()
            && (value.as_bytes()[position].is_ascii_whitespace()
                || matches!(value.as_bytes()[position], b'"' | b'\''))
        {
            position += 1;
        }
        if position >= value.len() || !matches!(value.as_bytes()[position], b':' | b'=') {
            cursor = start + marker.len();
            continue;
        }
        position += 1;
        while position < value.len()
            && (value.as_bytes()[position].is_ascii_whitespace()
                || matches!(value.as_bytes()[position], b'"' | b'\''))
        {
            position += 1;
        }
        let value_start = position;
        let quote = value
            .as_bytes()
            .get(value_start.wrapping_sub(1))
            .copied()
            .filter(|quote| matches!(*quote, b'"' | b'\''));
        let value_end = if let Some(quote) = quote {
            value.as_bytes()[value_start..]
                .iter()
                .position(|byte| *byte == quote)
                .map(|offset| value_start + offset)
                .unwrap_or(value.len())
        } else {
            value.as_bytes()[value_start..]
                .iter()
                .position(|byte| {
                    byte.is_ascii_whitespace() || matches!(*byte, b',' | b';' | b'&' | b'}' | b']')
                })
                .map(|offset| value_start + offset)
                .unwrap_or(value.len())
        };

        if value_end > value_start {
            value.replace_range(value_start..value_end, "[REDACTED]");
            cursor = value_start + "[REDACTED]".len();
        } else {
            cursor = start + marker.len();
        }
    }

    value
}

fn redact_auth_scheme(input: &str, scheme: &str) -> String {
    let mut value = input.to_string();
    let mut cursor = 0;
    loop {
        let lower = value.to_ascii_lowercase();
        let Some(offset) = lower[cursor..].find(scheme) else {
            break;
        };
        let start = cursor + offset;
        let after_scheme = start + scheme.len();
        if start > 0 && value.as_bytes()[start - 1].is_ascii_alphanumeric() {
            cursor = after_scheme;
            continue;
        }
        let mut token_start = after_scheme;
        while token_start < value.len() && value.as_bytes()[token_start].is_ascii_whitespace() {
            token_start += 1;
        }
        let token_end = value.as_bytes()[token_start..]
            .iter()
            .position(|byte| {
                byte.is_ascii_whitespace()
                    || matches!(*byte, b'"' | b'\'' | b',' | b';' | b'}' | b']')
            })
            .map(|offset| token_start + offset)
            .unwrap_or(value.len());
        if token_end > token_start {
            value.replace_range(token_start..token_end, "[REDACTED]");
            cursor = token_start + "[REDACTED]".len();
        } else {
            cursor = after_scheme;
        }
    }
    value
}

fn redact_prefixed_token(input: &str, prefix: &str) -> String {
    let mut value = input.to_string();
    let mut cursor = 0;
    loop {
        let lower = value.to_ascii_lowercase();
        let Some(offset) = lower[cursor..].find(&prefix.to_ascii_lowercase()) else {
            break;
        };
        let start = cursor + offset;
        let token_end = value.as_bytes()[start..]
            .iter()
            .position(|byte| !(byte.is_ascii_alphanumeric() || matches!(*byte, b'-' | b'_' | b'.')))
            .map(|offset| start + offset)
            .unwrap_or(value.len());
        let boundary = start == 0 || !value.as_bytes()[start - 1].is_ascii_alphanumeric();
        if boundary && token_end.saturating_sub(start) >= 20 {
            value.replace_range(start..token_end, "[REDACTED]");
            cursor = start + "[REDACTED]".len();
        } else {
            cursor = start + prefix.len();
        }
    }
    value
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;
    use std::{
        net::SocketAddr,
        sync::{
            atomic::{AtomicUsize, Ordering},
            Arc,
        },
    };
    use tokio::{
        io::{AsyncReadExt, AsyncWriteExt},
        net::{TcpListener, TcpStream},
        task::JoinHandle,
    };

    struct MockResponse {
        status: u16,
        headers: Vec<(&'static str, String)>,
        content_type: &'static str,
        body: String,
        delay: Duration,
        content_length: Option<usize>,
    }

    impl MockResponse {
        fn json(status: u16, body: &str) -> Self {
            Self {
                status,
                headers: Vec::new(),
                content_type: "application/json",
                body: body.to_string(),
                delay: Duration::ZERO,
                content_length: None,
            }
        }

        fn sse(body: &str) -> Self {
            Self {
                status: 200,
                headers: Vec::new(),
                content_type: "text/event-stream",
                body: body.to_string(),
                delay: Duration::ZERO,
                content_length: None,
            }
        }

        fn header(mut self, name: &'static str, value: &str) -> Self {
            self.headers.push((name, value.to_string()));
            self
        }

        fn delay(mut self, delay: Duration) -> Self {
            self.delay = delay;
            self
        }

        fn disconnect_after_body(mut self) -> Self {
            self.content_length = Some(self.body.len() + 32);
            self
        }
    }

    fn quick_policy(max_attempts: usize) -> RetryPolicy {
        RetryPolicy {
            max_attempts,
            max_duration: Duration::from_secs(3),
            base_delay: Duration::from_millis(20),
            max_backoff: Duration::from_millis(40),
        }
    }

    async fn spawn_server(
        responses: Vec<MockResponse>,
    ) -> (String, Arc<AtomicUsize>, JoinHandle<()>) {
        let listener = TcpListener::bind("127.0.0.1:0").await.unwrap();
        spawn_server_on(listener, responses).await
    }

    async fn spawn_server_on(
        listener: TcpListener,
        responses: Vec<MockResponse>,
    ) -> (String, Arc<AtomicUsize>, JoinHandle<()>) {
        let address = listener.local_addr().unwrap();
        let requests = Arc::new(AtomicUsize::new(0));
        let requests_task = requests.clone();
        let task = tokio::spawn(async move {
            for response in responses {
                let (mut socket, _) = listener.accept().await.unwrap();
                requests_task.fetch_add(1, Ordering::SeqCst);
                read_request(&mut socket).await;
                if !response.delay.is_zero() {
                    tokio::time::sleep(response.delay).await;
                }

                let body = response.body.into_bytes();
                let content_length = response.content_length.unwrap_or(body.len());
                let reason = match response.status {
                    200 => "OK",
                    429 => "Too Many Requests",
                    500 => "Internal Server Error",
                    502 => "Bad Gateway",
                    503 => "Service Unavailable",
                    504 => "Gateway Timeout",
                    529 => "Overloaded",
                    _ => "Error",
                };
                let mut headers = format!(
                    "HTTP/1.1 {} {}\r\nContent-Length: {}\r\nContent-Type: {}\r\nConnection: close\r\n",
                    response.status,
                    reason,
                    content_length,
                    response.content_type
                );
                for (name, value) in response.headers {
                    headers.push_str(&format!("{name}: {value}\r\n"));
                }
                headers.push_str("\r\n");
                let _ = socket.write_all(headers.as_bytes()).await;
                let _ = socket.write_all(&body).await;
            }
        });

        (format!("http://{address}"), requests, task)
    }

    async fn read_request(socket: &mut TcpStream) {
        let mut bytes = Vec::new();
        let mut chunk = [0_u8; 2048];
        loop {
            let count = match socket.read(&mut chunk).await {
                Ok(count) if count > 0 => count,
                _ => return,
            };
            bytes.extend_from_slice(&chunk[..count]);

            if let Some(header_end) = bytes.windows(4).position(|value| value == b"\r\n\r\n") {
                let headers = String::from_utf8_lossy(&bytes[..header_end]);
                let content_length = headers
                    .lines()
                    .find_map(|line| {
                        let (name, value) = line.split_once(':')?;
                        name.eq_ignore_ascii_case("content-length")
                            .then(|| value.trim().parse::<usize>().ok())
                            .flatten()
                    })
                    .unwrap_or_default();
                let body = &bytes[header_end + 4..];
                let is_chunked = headers.lines().any(|line| {
                    line.split_once(':').is_some_and(|(name, value)| {
                        name.eq_ignore_ascii_case("transfer-encoding")
                            && value.to_ascii_lowercase().contains("chunked")
                    })
                });
                if (is_chunked && chunked_body_complete(body))
                    || (!is_chunked && bytes.len() >= header_end + 4 + content_length)
                {
                    return;
                }
            }
        }
    }

    fn post(client: &reqwest::Client, url: &str) -> RequestBuilder {
        client.post(url).json(&json!({"input": "test"}))
    }

    fn chunked_body_complete(mut body: &[u8]) -> bool {
        loop {
            let Some(line_end) = body.windows(2).position(|value| value == b"\r\n") else {
                return false;
            };
            let size_line = String::from_utf8_lossy(&body[..line_end]);
            let size = size_line
                .split(';')
                .next()
                .and_then(|value| usize::from_str_radix(value.trim(), 16).ok());
            let Some(size) = size else {
                return false;
            };
            body = &body[line_end + 2..];
            if size == 0 {
                return body.starts_with(b"\r\n")
                    || body.windows(4).any(|value| value == b"\r\n\r\n");
            }
            if body.len() < size + 2 || &body[size..size + 2] != b"\r\n" {
                return false;
            }
            body = &body[size + 2..];
        }
    }

    #[test]
    fn formats_provider_error_shapes_and_preserves_unknown_bodies() {
        let cases = [
            (
                br#"{"error":{"type":"invalid_request_error","code":"model_not_found","message":"Unknown model"}}"#.as_slice(),
                &["HTTP 429", "Unknown model", "invalid_request_error", "model_not_found"][..],
            ),
            (
                br#"{"type":"error","error":{"type":"overloaded_error","message":"Overloaded"}}"#.as_slice(),
                &["HTTP 429", "Overloaded", "overloaded_error"][..],
            ),
            (
                br#"{"message":"Rate limit reached","code":"rate_limit_exceeded"}"#.as_slice(),
                &["HTTP 429", "Rate limit reached", "rate_limit_exceeded"][..],
            ),
            (
                br#"{"message":"Request throttled","__type":"ThrottlingException"}"#.as_slice(),
                &["HTTP 429", "Request throttled", "ThrottlingException"][..],
            ),
            (
                br#"{"error":{"code":429,"message":"Quota exceeded","status":"RESOURCE_EXHAUSTED","details":[{"@type":"type.googleapis.com/google.rpc.RetryInfo","retryDelay":"2s"}]}}"#.as_slice(),
                &["HTTP 429", "Quota exceeded", "RESOURCE_EXHAUSTED", "RetryInfo", "retryDelay"][..],
            ),
        ];

        for (bytes, expected) in cases {
            let body = ResponseBody {
                bytes: bytes.to_vec(),
                truncated: false,
                read_error: None,
            };
            let diagnostic = format_response_error(StatusCode::TOO_MANY_REQUESTS, &body);
            for value in expected {
                assert!(diagnostic.contains(value), "{diagnostic} missing {value}");
            }
        }

        let raw = b"upstream proxy rejected request: gateway said unavailable";
        let body = ResponseBody {
            bytes: raw.to_vec(),
            truncated: false,
            read_error: None,
        };
        let diagnostic = format_response_error(StatusCode::BAD_GATEWAY, &body);
        assert!(diagnostic.contains("HTTP 502"));
        assert!(diagnostic.contains("gateway said unavailable"));

        let raw_json = b"{\n  \"error\": {\n    \"message\": \"provider detail\",\n    \"code\": \"rate_limit_exceeded\"\n  }\n}";
        let body = ResponseBody {
            bytes: raw_json.to_vec(),
            truncated: false,
            read_error: None,
        };
        let diagnostic = format_response_error(StatusCode::TOO_MANY_REQUESTS, &body);
        assert!(diagnostic.contains(std::str::from_utf8(raw_json).unwrap()));
    }

    #[test]
    fn preserves_google_oauth_error_description_uri_and_other_fields() {
        let body = ResponseBody {
            bytes: br#"{"error":"invalid_grant","error_description":"The refresh token has expired or been revoked. Re-authorize the account.","error_uri":"https://developers.google.com/identity/protocols/oauth2","error_hint":"Reconnect the Google account","request_id":"oauth-request-123","provider_detail":"Contact support with this request ID","client_secret":"oauth-test-secret"}"#.to_vec(),
            truncated: false,
            read_error: None,
        };

        let diagnostic = format_response_error(StatusCode::BAD_REQUEST, &body);
        for value in [
            "HTTP 400",
            "invalid_grant",
            "The refresh token has expired or been revoked. Re-authorize the account.",
            "error_uri",
            "https://developers.google.com/identity/protocols/oauth2",
            "Reconnect the Google account",
            "oauth-request-123",
            "Contact support with this request ID",
        ] {
            assert!(diagnostic.contains(value), "{diagnostic} missing {value}");
        }
        assert!(!diagnostic.contains("oauth-test-secret"), "{diagnostic}");
        assert!(diagnostic.contains("[REDACTED]"), "{diagnostic}");
    }

    #[test]
    fn redacts_credentials_from_json_and_unparsed_error_bodies() {
        let json_body = ResponseBody {
            bytes: br#"{"service":{"api_key":"json-secret-key","refresh_token":"refresh-secret","note":"Authorization: Bearer header-secret","key":"provider-diagnostic-key","request_id":"request-123"}}"#.to_vec(),
            truncated: false,
            read_error: None,
        };
        let diagnostic = format_response_error(StatusCode::BAD_REQUEST, &json_body);
        for secret in ["json-secret-key", "refresh-secret", "header-secret"] {
            assert!(!diagnostic.contains(secret), "{diagnostic}");
        }
        assert!(diagnostic.contains("[REDACTED]"));
        assert!(diagnostic.contains("provider-diagnostic-key"));
        assert!(diagnostic.contains("request-123"));

        let raw_body = ResponseBody {
            bytes: b"upstream echoed Authorization: Bearer plain-secret-token, api_key=other-secret, API key: freeform-secret"
                .to_vec(),
            truncated: false,
            read_error: None,
        };
        let diagnostic = format_response_error(StatusCode::BAD_GATEWAY, &raw_body);
        for secret in ["plain-secret-token", "other-secret", "freeform-secret"] {
            assert!(!diagnostic.contains(secret), "{diagnostic}");
        }

        let truncated = ResponseBody {
            bytes: vec![b'x'; MAX_ERROR_RESPONSE_BYTES],
            truncated: true,
            read_error: None,
        };
        let diagnostic = format_response_error(StatusCode::BAD_GATEWAY, &truncated);
        assert!(diagnostic.contains("response body truncated after 65536 bytes"));
    }

    #[tokio::test]
    async fn quota_mentions_without_structured_permanent_evidence_remain_retryable() {
        for body in [
            r#"{"error":{"message":"You have exceeded your daily quota. Try again tomorrow."}}"#,
            r#"{"error":{"message":"Daily quota has been exceeded. Retry in 24 hours."}}"#,
            r#"{"error":{"message":"Monthly limit exceeded. Please retry after renewal."}}"#,
        ] {
            let (url, requests, server) = spawn_server(vec![
                MockResponse::json(429, body),
                MockResponse::json(200, r#"{"ok":true}"#),
            ])
            .await;
            let response =
                send_with_retry_policy(post(&reqwest::Client::new(), &url), quick_policy(3))
                    .await
                    .unwrap();

            assert!(response.status().is_success(), "{body}");
            assert_eq!(requests.load(Ordering::SeqCst), 2, "{body}");
            server.await.unwrap();
        }
    }

    #[tokio::test]
    async fn retries_gemini_quota_error_with_nested_google_retry_info() {
        let body = r#"{"error":{"code":429,"message":"You exceeded your current quota for this request","status":"RESOURCE_EXHAUSTED","details":[{"@type":"type.googleapis.com/google.rpc.QuotaFailure","violations":[{"quotaId":"GenerateRequestsPerMinutePerProjectPerModel-FreeTier"}]},{"@type":"type.googleapis.com/google.rpc.RetryInfo","retryDelay":"0.08s"}]}}"#;
        let (url, requests, server) = spawn_server(vec![
            MockResponse::json(429, body),
            MockResponse::json(200, r#"{"ok":true}"#),
        ])
        .await;
        let started = Instant::now();
        let response = send_with_retry_policy(post(&reqwest::Client::new(), &url), quick_policy(3))
            .await
            .unwrap();

        assert!(response.status().is_success());
        assert!(started.elapsed() >= Duration::from_millis(70));
        assert_eq!(requests.load(Ordering::SeqCst), 2);
        server.await.unwrap();
    }

    #[tokio::test]
    async fn preserves_provider_diagnostic_when_retry_attempts_are_exhausted() {
        let body = r#"{"error":{"type":"overloaded_error","code":"UPSTREAM_BUSY","message":"Provider cluster overloaded; capacity unavailable"}}"#;
        let (url, requests, server) = spawn_server(vec![
            MockResponse::json(503, body),
            MockResponse::json(503, body),
        ])
        .await;
        let error = send_with_retry_policy(post(&reqwest::Client::new(), &url), quick_policy(2))
            .await
            .unwrap_err()
            .to_string();

        assert!(error.contains("HTTP 503"), "{error}");
        assert!(error.contains("Provider cluster overloaded"), "{error}");
        assert!(error.contains("overloaded_error"), "{error}");
        assert!(error.contains("UPSTREAM_BUSY"), "{error}");
        assert!(error.contains("2 attempts"), "{error}");
        assert!(!error.contains("provider request failed"), "{error}");
        assert_eq!(requests.load(Ordering::SeqCst), 2);
        server.await.unwrap();
    }

    #[tokio::test]
    async fn bounds_unrecognized_response_body_size() {
        let body = format!(
            "proxy response: {}",
            "x".repeat(MAX_ERROR_RESPONSE_BYTES + 1_024)
        );
        let (url, requests, server) = spawn_server(vec![MockResponse::json(400, &body)]).await;
        let error = send_with_retry_policy(post(&reqwest::Client::new(), &url), quick_policy(3))
            .await
            .unwrap_err()
            .to_string();

        assert!(error.contains("HTTP 400"), "{error}");
        assert!(error.contains("response body truncated"), "{error}");
        assert!(
            error.len() <= MAX_ERROR_RESPONSE_BYTES + 256,
            "{}",
            error.len()
        );
        assert_eq!(requests.load(Ordering::SeqCst), 1);
        server.await.unwrap();
    }

    #[tokio::test]
    async fn refuses_retry_when_google_retry_info_is_beyond_diagnostic_limit() {
        let body = format!(
            r#"{{"error":{{"message":"{}","details":[{{"@type":"type.googleapis.com/google.rpc.RetryInfo","retryDelay":"120s"}}]}}}}"#,
            "x".repeat(MAX_ERROR_RESPONSE_BYTES)
        );
        assert!(
            body.find("\"retryDelay\"").unwrap() > MAX_ERROR_RESPONSE_BYTES,
            "RetryInfo must be beyond the retained response-body prefix"
        );

        let (url, requests, server) = spawn_server(vec![MockResponse::json(429, &body)]).await;
        let error = send_with_retry_policy(post(&reqwest::Client::new(), &url), quick_policy(3))
            .await
            .unwrap_err()
            .to_string();

        assert!(error.contains("HTTP 429"), "{error}");
        assert!(
            error.contains("response body truncated after 65536 bytes"),
            "{error}"
        );
        assert!(
            error.contains("provider retry instructions cannot be determined"),
            "{error}"
        );
        assert_eq!(requests.load(Ordering::SeqCst), 1);
        server.await.unwrap();
    }

    #[tokio::test]
    async fn preserves_partial_provider_body_when_response_read_fails() {
        let body =
            r#"{"error":{"message":"partial provider diagnostic","api_key":"partial-secret""#;
        let (url, requests, server) =
            spawn_server(vec![MockResponse::json(429, body).disconnect_after_body()]).await;
        let error = send_with_retry_policy(post(&reqwest::Client::new(), &url), quick_policy(3))
            .await
            .unwrap_err()
            .to_string();

        for expected in [
            "HTTP 429",
            "partial provider diagnostic",
            "response body incomplete",
            "Failed to read provider error response (HTTP 429)",
        ] {
            assert!(error.contains(expected), "{error} missing {expected}");
        }
        assert!(!error.contains("partial-secret"), "{error}");
        assert_eq!(requests.load(Ordering::SeqCst), 1);
        server.await.unwrap();
    }

    #[tokio::test]
    async fn retries_429_and_respects_retry_after() {
        let (url, requests, server) = spawn_server(vec![
            MockResponse::json(
                429,
                r#"{"error":{"type":"rate_limit_error","message":"Rate limit reached"}}"#,
            )
            .header("Retry-After", "1"),
            MockResponse::json(200, r#"{"ok":true}"#),
        ])
        .await;
        let started = Instant::now();
        let response = send_with_retry_policy(post(&reqwest::Client::new(), &url), quick_policy(3))
            .await
            .unwrap();

        assert!(response.status().is_success());
        assert!(started.elapsed() >= Duration::from_millis(950));
        assert_eq!(requests.load(Ordering::SeqCst), 2);
        server.await.unwrap();
    }

    #[tokio::test]
    async fn retries_429_without_retry_after() {
        let (url, requests, server) = spawn_server(vec![
            MockResponse::json(429, r#"{"error":{"message":"Too many requests"}}"#),
            MockResponse::json(200, r#"{"ok":true}"#),
        ])
        .await;
        let response = send_with_retry_policy(post(&reqwest::Client::new(), &url), quick_policy(3))
            .await
            .unwrap();

        assert!(response.status().is_success());
        assert_eq!(requests.load(Ordering::SeqCst), 2);
        server.await.unwrap();
    }

    #[tokio::test]
    async fn does_not_retry_permanent_quota_failures() {
        let (url, requests, server) = spawn_server(vec![MockResponse::json(
            429,
            r#"{"error":{"type":"insufficient_quota","message":"You exceeded your current quota. Check billing."}}"#,
        )])
        .await;
        let error = send_with_retry_policy(post(&reqwest::Client::new(), &url), quick_policy(4))
            .await
            .unwrap_err();

        assert!(error.to_string().contains("billing"));
        assert_eq!(requests.load(Ordering::SeqCst), 1);
        server.await.unwrap();
    }

    #[tokio::test]
    async fn retries_server_errors_and_anthropic_overload() {
        for status in [408, 425, 500, 502, 503, 504, 529] {
            let (url, requests, server) = spawn_server(vec![
                MockResponse::json(
                    status,
                    r#"{"error":{"type":"overloaded_error","message":"Overloaded"}}"#,
                ),
                MockResponse::json(200, r#"{"ok":true}"#),
            ])
            .await;
            let response =
                send_with_retry_policy(post(&reqwest::Client::new(), &url), quick_policy(3))
                    .await
                    .unwrap();

            assert!(response.status().is_success());
            assert_eq!(requests.load(Ordering::SeqCst), 2);
            server.await.unwrap();
        }
    }

    #[tokio::test]
    async fn stops_after_the_configured_attempt_limit() {
        let (url, requests, server) = spawn_server(vec![
            MockResponse::json(503, r#"{"message":"Temporarily unavailable"}"#),
            MockResponse::json(503, r#"{"message":"Temporarily unavailable"}"#),
            MockResponse::json(503, r#"{"message":"Temporarily unavailable"}"#),
        ])
        .await;
        let error = send_with_retry_policy(post(&reqwest::Client::new(), &url), quick_policy(3))
            .await
            .unwrap_err();

        assert!(error.to_string().contains("3 attempts"));
        assert_eq!(requests.load(Ordering::SeqCst), 3);
        server.await.unwrap();
    }

    #[tokio::test]
    async fn retries_connection_establishment_failures() {
        let reserved = TcpListener::bind("127.0.0.1:0").await.unwrap();
        let address: SocketAddr = reserved.local_addr().unwrap();
        drop(reserved);

        let server_task = tokio::spawn(async move {
            tokio::time::sleep(Duration::from_millis(30)).await;
            let listener = TcpListener::bind(address).await.unwrap();
            let (_, _, server) =
                spawn_server_on(listener, vec![MockResponse::json(200, r#"{"ok":true}"#)]).await;
            server.await.unwrap();
        });

        let response = send_with_retry_policy(
            post(&reqwest::Client::new(), &format!("http://{address}")),
            quick_policy(8),
        )
        .await
        .unwrap();
        assert!(response.status().is_success());
        server_task.await.unwrap();
    }

    #[tokio::test]
    async fn does_not_replay_ambiguous_request_timeouts() {
        let (url, requests, server) = spawn_server(vec![
            MockResponse::json(200, r#"{"ok":true}"#).delay(Duration::from_millis(100))
        ])
        .await;
        let client = reqwest::Client::builder()
            .timeout(Duration::from_millis(20))
            .build()
            .unwrap();
        let result = send_with_retry_policy(post(&client, &url), quick_policy(4)).await;

        assert!(result.is_err());
        assert_eq!(requests.load(Ordering::SeqCst), 1);
        server.await.unwrap();
    }

    #[tokio::test]
    async fn cancellation_interrupts_retry_backoff() {
        let (url, requests, server) = spawn_server(vec![MockResponse::json(
            429,
            r#"{"error":{"message":"Rate limit reached"}}"#,
        )
        .header("Retry-After", "1")])
        .await;
        let request = tokio::spawn(send_with_retry_policy(
            post(&reqwest::Client::new(), &url),
            quick_policy(4),
        ));

        tokio::time::sleep(Duration::from_millis(30)).await;
        request.abort();
        assert!(request.await.unwrap_err().is_cancelled());
        assert_eq!(requests.load(Ordering::SeqCst), 1);
        server.await.unwrap();
    }

    const CAPTURED_GEMINI_RATE_LIMIT: &str = r#"[{"error":{"code":429,"message":"You exceeded your current quota, please check your plan and billing details. For more information on this error, head to: https://aistudio.google.com/docs/rate-limits. To monitor your current usage, head to: https://ai.dev/rate-limit. * Quota exceeded for metric: generativelanguage.googleapis.com/generate_content_free_tier_requests, limit: 15, model: gemini-3.5-flash-lite, description: Generate content requests per minute limit exceeded on Free Tier. See https://aistudio.google.com/docs/rate-limits.Please retry in 59.599883052s.","status":"RESOURCE_EXHAUSTED","details":[{"@type":"type.googleapis.com/google.rpc.Help","links":[{"description":"Learn more about Gemini API quotas","url":"https://aistudio.google.com/docs/rate-limits"}]},{"@type":"type.googleapis.com/google.rpc.QuotaFailure","violations":[{"description":"Generate content requests per minute limit exceeded on Free Tier. See https://aistudio.google.com/docs/rate-limits.","quotaMetric":"generativelanguage.googleapis.com/generate_content_free_tier_requests","quotaId":"GenerateRequestsPerMinutePerProjectPerModel-FreeTier","quotaDimensions":{"model":"gemini-3.5-flash-lite","location":"global"},"quotaValue":"15"}]},{"@type":"type.googleapis.com/google.rpc.RetryInfo","retryDelay":"59s"}]}}]"#;

    fn production_default_policy() -> RetryPolicy {
        RetryPolicy {
            max_attempts: DEFAULT_MAX_ATTEMPTS,
            max_duration: Duration::from_secs(DEFAULT_MAX_DURATION_SECS),
            base_delay: Duration::from_millis(BASE_DELAY_MS),
            max_backoff: Duration::from_millis(MAX_BACKOFF_MS),
        }
    }

    #[test]
    fn captured_gemini_array_envelope_retries_minute_quota_after_provider_delay() {
        let body = ResponseBody {
            bytes: CAPTURED_GEMINI_RATE_LIMIT.as_bytes().to_vec(),
            truncated: false,
            read_error: None,
        };
        let data: Value = serde_json::from_slice(&body.bytes).unwrap();

        assert!(data.is_array());
        assert!(!google_has_daily_quota(&data));
        assert!(permanent_failure(StatusCode::TOO_MANY_REQUESTS, &body.bytes).is_none());

        let mut state = RetryState::new(production_default_policy());
        let delay = state
            .retry_response(
                StatusCode::TOO_MANY_REQUESTS,
                &HeaderMap::new(),
                &body,
                true,
            )
            .unwrap();

        assert!(delay >= Duration::from_secs(59), "{delay:?}");
        assert_eq!(state.attempt, 2);
    }

    #[test]
    fn array_wrapped_gemini_daily_quota_remains_permanent() {
        let body = br#"[{"error":{"code":429,"message":"Daily requests quota exhausted","status":"RESOURCE_EXHAUSTED","details":[{"@type":"type.googleapis.com/google.rpc.QuotaFailure","violations":[{"quotaId":"GenerateRequestsPerDayPerProjectPerModel-FreeTier","description":"daily request cap reached"}]},{"@type":"type.googleapis.com/google.rpc.RetryInfo","retryDelay":"59s"}]}}]"#;
        let data: Value = serde_json::from_slice(body).unwrap();

        assert!(google_has_daily_quota(&data));
        assert!(permanent_failure(StatusCode::TOO_MANY_REQUESTS, body).is_some());

        let response_body = ResponseBody {
            bytes: body.to_vec(),
            truncated: false,
            read_error: None,
        };
        let mut state = RetryState::new(production_default_policy());
        let error = state
            .retry_response(
                StatusCode::TOO_MANY_REQUESTS,
                &HeaderMap::new(),
                &response_body,
                true,
            )
            .unwrap_err();

        let diagnostic = error.to_string();
        assert!(diagnostic.contains("daily quota"));
        assert!(diagnostic.contains("Daily requests quota exhausted"));
        assert!(diagnostic.contains("GenerateRequestsPerDayPerProjectPerModel-FreeTier"));
        assert!(diagnostic.contains("daily request cap reached"));
        assert_eq!(state.attempt, 1);
    }

    #[test]
    fn parses_supported_retry_instructions_and_classifies_verified_failures() {
        let nested = br#"{"error":{"details":[{"@type":"type.googleapis.com/google.rpc.RetryInfo","retryDelay":"59s"}]}}"#;
        assert_eq!(
            google_retry_info(nested).unwrap(),
            Some(Duration::from_secs(59))
        );

        let top_level =
            br#"[{"@type":"type.googleapis.com/google.rpc.RetryInfo","retryDelay":"1.5s"}]"#;
        assert_eq!(
            google_retry_info(top_level).unwrap(),
            Some(Duration::from_millis(1500))
        );

        let unrelated =
            br#"{"metadata":{"retryDelay":"1s"},"error":{"message":"retry later","retryAfter":2}}"#;
        assert_eq!(google_retry_info(unrelated).unwrap(), None);

        let malformed = br#"{"error":{"details":[{"@type":"type.googleapis.com/google.rpc.RetryInfo","retryDelay":"later"}]}}"#;
        assert!(google_retry_info(malformed).is_err());

        let mut headers = HeaderMap::new();
        headers.insert(RETRY_AFTER, "2".parse().unwrap());
        assert_eq!(
            parse_retry_after_header(&headers),
            Some(Duration::from_secs(2))
        );
        let longer_google = br#"{"error":{"details":[{"@type":"type.googleapis.com/google.rpc.RetryInfo","retryDelay":"3.5s"}]}}"#;
        assert_eq!(
            retry_delay(&headers, longer_google).unwrap(),
            Some(Duration::from_millis(3500))
        );
        let longer_header = br#"{"error":{"details":[{"@type":"type.googleapis.com/google.rpc.RetryInfo","retryDelay":"1s"}]}}"#;
        assert_eq!(
            retry_delay(&headers, longer_header).unwrap(),
            Some(Duration::from_secs(2))
        );

        let date = (Utc::now() + chrono::Duration::seconds(2))
            .format("%a, %d %b %Y %H:%M:%S GMT")
            .to_string();
        headers.insert(RETRY_AFTER, date.parse().unwrap());
        assert!(parse_retry_after_header(&headers).unwrap() >= Duration::from_secs(1));

        assert!(permanent_failure(
            StatusCode::TOO_MANY_REQUESTS,
            br#"{"error":{"type":"insufficient_quota"}}"#
        )
        .is_some());
        assert!(permanent_failure(
            StatusCode::TOO_MANY_REQUESTS,
            br#"{"error":{"code":"project_spend_limit_exceeded"}}"#
        )
        .is_some());
        assert!(permanent_failure(
            StatusCode::TOO_MANY_REQUESTS,
            br#"{"error":{"code":"invalid_api_key"}}"#
        )
        .is_some());
        assert!(permanent_failure(
            StatusCode::TOO_MANY_REQUESTS,
            br#"{"error":{"code":"ServiceQuotaExceededException"}}"#
        )
        .is_none());
        assert!(permanent_failure(
            StatusCode::TOO_MANY_REQUESTS,
            br#"{"error":{"message":"Daily quota exceeded; please retry tomorrow"}}"#
        )
        .is_none());
        assert!(permanent_failure(
            StatusCode::BAD_GATEWAY,
            br#"{"error":{"type":"insufficient_quota"}}"#
        )
        .is_none());

        let google_daily = br#"{"error":{"details":[{"@type":"type.googleapis.com/google.rpc.QuotaFailure","violations":[{"quotaId":"GenerateRequestsPerDayPerProjectPerModel-FreeTier"}]},{"@type":"type.googleapis.com/google.rpc.RetryInfo","retryDelay":"59s"}]}}"#;
        assert!(permanent_failure(StatusCode::TOO_MANY_REQUESTS, google_daily).is_some());
        let google_minute = br#"{"error":{"details":[{"@type":"type.googleapis.com/google.rpc.QuotaFailure","violations":[{"quotaId":"GenerateRequestsPerMinutePerProjectPerModel-FreeTier"}]}]}}"#;
        assert!(permanent_failure(StatusCode::TOO_MANY_REQUESTS, google_minute).is_none());

        assert!(is_retryable_status(StatusCode::REQUEST_TIMEOUT));
        assert!(is_retryable_status(StatusCode::TOO_EARLY));
        assert!(is_retryable_status(StatusCode::TOO_MANY_REQUESTS));
    }

    #[test]
    fn scheduler_accepts_google_delay_and_counts_total_attempts() {
        assert_eq!(DEFAULT_MAX_ATTEMPTS, 6);
        assert_eq!(DEFAULT_MAX_DURATION_SECS, 300);
        assert_eq!(MAX_CONFIGURED_ATTEMPTS, 10);
        assert_eq!(MAX_CONFIGURED_DURATION_SECS, 300);
        let policy = RetryPolicy {
            max_attempts: DEFAULT_MAX_ATTEMPTS,
            max_duration: Duration::from_secs(DEFAULT_MAX_DURATION_SECS),
            base_delay: Duration::from_millis(BASE_DELAY_MS),
            max_backoff: Duration::from_millis(MAX_BACKOFF_MS),
        };

        let mut three_second_delay = RetryState::new(policy);
        assert_eq!(
            three_second_delay
                .schedule("HTTP 429", Some(Duration::from_secs(3)))
                .unwrap(),
            Duration::from_secs(3)
        );

        let mut long_delay = RetryState::new(policy);
        assert_eq!(
            long_delay
                .schedule("HTTP 429", Some(Duration::from_secs(59)))
                .unwrap(),
            Duration::from_secs(59)
        );

        let mut attempts = RetryState::new(policy);
        for _ in 0..4 {
            attempts.schedule("HTTP 429", None).unwrap();
        }
        assert_eq!(attempts.attempt, 5);
        attempts.schedule("HTTP 429", None).unwrap();
        assert_eq!(attempts.attempt, 6);
        assert!(attempts.schedule("HTTP 429", None).is_err());
    }

    #[tokio::test]
    async fn retries_repeated_429_responses_within_the_attempt_budget() {
        let responses = vec![
            MockResponse::json(429, r#"{"error":{"message":"Rate limited"}}"#),
            MockResponse::json(429, r#"{"error":{"message":"Rate limited"}}"#),
            MockResponse::json(429, r#"{"error":{"message":"Rate limited"}}"#),
            MockResponse::json(429, r#"{"error":{"message":"Rate limited"}}"#),
            MockResponse::json(200, r#"{"ok":true}"#),
        ];
        let (url, requests, server) = spawn_server(responses).await;
        let response = send_with_retry_policy(post(&reqwest::Client::new(), &url), quick_policy(6))
            .await
            .unwrap();

        assert!(response.status().is_success());
        assert_eq!(requests.load(Ordering::SeqCst), 5);
        server.await.unwrap();
    }

    #[tokio::test]
    async fn malformed_google_retry_info_never_falls_back_to_a_shorter_delay() {
        let body = r#"{"error":{"code":"rate_limit_exceeded","message":"Rate limited","request_id":"malformed-retry-123","details":[{"@type":"type.googleapis.com/google.rpc.RetryInfo","retryDelay":"later"},{"@type":"type.googleapis.com/google.rpc.QuotaFailure","violations":[{"quotaId":"GenerateRequestsPerMinutePerProjectPerModel-FreeTier"}]}]}}"#;
        let (url, requests, server) = spawn_server(vec![
            MockResponse::json(429, body).header("Retry-After", "0"),
            MockResponse::json(200, r#"{"ok":true}"#),
        ])
        .await;
        let error = send_with_retry_policy(post(&reqwest::Client::new(), &url), quick_policy(3))
            .await
            .unwrap_err()
            .to_string();

        assert!(error.contains("HTTP 429"), "{error}");
        assert!(error.contains("malformed retryDelay"), "{error}");
        assert!(error.contains(r#""retryDelay":"later""#), "{error}");
        assert!(error.contains("rate_limit_exceeded"), "{error}");
        assert!(error.contains("malformed-retry-123"), "{error}");
        assert!(
            error.contains("GenerateRequestsPerMinutePerProjectPerModel-FreeTier"),
            "{error}"
        );
        assert_eq!(requests.load(Ordering::SeqCst), 1);
        server.abort();
    }

    #[tokio::test]
    async fn stops_when_retry_after_exceeds_the_total_retry_budget() {
        let (url, requests, server) = spawn_server(vec![MockResponse::json(
            429,
            r#"{"error":{"message":"Rate limit reached"}}"#,
        )
        .header("Retry-After", "1")])
        .await;
        let mut policy = quick_policy(4);
        policy.max_duration = Duration::from_millis(100);
        let error = send_with_retry_policy(post(&reqwest::Client::new(), &url), policy)
            .await
            .unwrap_err();

        assert!(error.to_string().contains("retry admission budget"));
        assert!(error.to_string().contains("next wait (1.00s)"));
        assert!(error.to_string().contains("Rate limit reached"));
        assert_eq!(requests.load(Ordering::SeqCst), 1);
        server.await.unwrap();
    }

    #[tokio::test]
    async fn retries_sse_handshake_without_duplicating_tool_calls() {
        let (url, requests, server) = spawn_server(vec![
            MockResponse::json(503, r#"{"message":"Temporarily unavailable"}"#),
            MockResponse::sse("data: tool-call\n\n"),
        ])
        .await;
        let (tx, _rx) = tokio::sync::mpsc::unbounded_channel();
        let mut handler = crate::client::SseHandler::new(tx, crate::utils::AbortSignalInner::new());
        let mut handled = 0;

        super::super::stream::sse_stream_with_policy(
            post(&reqwest::Client::new(), &url),
            |message| {
                if message.data == "tool-call" {
                    handled += 1;
                    handler.tool_call(crate::client::ToolCall::new(
                        "probe".to_string(),
                        json!({}),
                        Some("call-1".to_string()),
                    ))?;
                }
                Ok(false)
            },
            quick_policy(3),
        )
        .await
        .unwrap();

        let (_, calls) = handler.take();
        assert_eq!(handled, 1);
        assert_eq!(calls.len(), 1);
        assert_eq!(requests.load(Ordering::SeqCst), 2);
        server.await.unwrap();
    }

    #[tokio::test]
    async fn retries_ambiguous_quota_exceptions_when_server_signals_retry() {
        for code in ["ServiceQuotaExceededException", "QuotaExceededException"] {
            let body = format!(r#"{{"error":{{"code":"{code}","message":"Quota exceeded"}}}}"#);
            let (url, requests, server) = spawn_server(vec![
                MockResponse::json(429, &body).header("Retry-After", "0"),
                MockResponse::json(200, r#"{"ok":true}"#),
            ])
            .await;
            let response =
                send_with_retry_policy(post(&reqwest::Client::new(), &url), quick_policy(3))
                    .await
                    .unwrap();

            assert!(response.status().is_success(), "{code}");
            assert_eq!(requests.load(Ordering::SeqCst), 2, "{code}");
            server.await.unwrap();
        }
    }

    #[tokio::test]
    async fn retries_rate_limits_when_daily_or_monthly_quota_is_only_mentioned() {
        let cases = [
            r#"{"error":{"message":"Daily quota available: 20 requests; request rate limit exceeded per minute."}}"#,
            r#"{"error":{"message":"Monthly limit allows 1,000 requests; per-second rate limit exceeded."}}"#,
        ];
        for body in cases {
            let (url, requests, server) = spawn_server(vec![
                MockResponse::json(429, body),
                MockResponse::json(200, r#"{"ok":true}"#),
            ])
            .await;
            let response =
                send_with_retry_policy(post(&reqwest::Client::new(), &url), quick_policy(3))
                    .await
                    .unwrap();

            assert!(response.status().is_success(), "{body}");
            assert_eq!(requests.load(Ordering::SeqCst), 2, "{body}");
            server.await.unwrap();
        }
    }

    #[tokio::test]
    async fn does_not_retry_verified_billing_auth_or_long_window_quota_failures() {
        let cases = [
            r#"{"error":{"type":"insufficient_quota","message":"You exceeded your current quota"}}"#,
            r#"{"error":{"code":"billing_hard_limit_reached","message":"Billing hard limit reached"}}"#,
            r#"{"error":{"code":"credit_balance_exhausted","message":"Prepaid credits exhausted"}}"#,
            r#"{"error":{"code":"organization_spend_limit_exceeded","message":"Organization spend limit reached"}}"#,
            r#"{"error":{"code":"invalid_api_key","message":"Invalid API key"}}"#,
            r#"{"error":{"details":[{"@type":"type.googleapis.com/google.rpc.QuotaFailure","violations":[{"quotaId":"GenerateRequestsPerDayPerProjectPerModel-FreeTier"}]},{"@type":"type.googleapis.com/google.rpc.RetryInfo","retryDelay":"1s"}]}}"#,
        ];
        for body in cases {
            let (url, requests, server) = spawn_server(vec![
                MockResponse::json(429, body).header("Retry-After", "0")
            ])
            .await;
            let error =
                send_with_retry_policy(post(&reqwest::Client::new(), &url), quick_policy(4))
                    .await
                    .unwrap_err()
                    .to_string();

            assert!(error.contains("HTTP 429"), "{error}");
            assert_eq!(requests.load(Ordering::SeqCst), 1, "{body}");
            server.await.unwrap();
        }
    }

    #[tokio::test]
    async fn does_not_replay_non_cloneable_http_requests() {
        let (url, requests, server) = spawn_server(vec![MockResponse::json(
            503,
            r#"{"message":"Temporarily unavailable"}"#,
        )])
        .await;
        let body =
            reqwest::Body::wrap_stream(futures_util::stream::iter(vec![Ok::<_, std::io::Error>(
                bytes::Bytes::from_static(b"one-shot body"),
            )]));
        let request = reqwest::Client::new()
            .post(&url)
            .header("content-type", "application/octet-stream")
            .body(body);
        assert!(request.try_clone().is_none());

        let error = send_with_retry_policy(request, quick_policy(4))
            .await
            .unwrap_err()
            .to_string();
        assert!(error.contains("HTTP 503"), "{error}");
        assert_eq!(requests.load(Ordering::SeqCst), 1);
        server.await.unwrap();
    }

    #[tokio::test]
    async fn does_not_retry_non_cloneable_sse_request() {
        let (url, requests, server) =
            spawn_server(vec![MockResponse::sse("data: single-response\n\n")]).await;
        let body =
            reqwest::Body::wrap_stream(futures_util::stream::iter(vec![Ok::<_, std::io::Error>(
                bytes::Bytes::from_static(b"non-cloneable request body"),
            )]));
        let request = reqwest::Client::new()
            .post(&url)
            .header("content-type", "application/octet-stream")
            .body(body);
        assert!(request.try_clone().is_none());

        let mut delivered = Vec::new();
        super::super::stream::sse_stream_with_policy(
            request,
            |message| {
                delivered.push(message.data);
                Ok(false)
            },
            quick_policy(4),
        )
        .await
        .unwrap();

        assert_eq!(delivered, vec!["single-response"]);
        assert_eq!(requests.load(Ordering::SeqCst), 1);
        server.await.unwrap();
    }

    #[tokio::test]
    async fn does_not_replay_stream_after_partial_content() {
        let (url, requests, server) =
            spawn_server(vec![MockResponse::sse("data: partial-output\n\n")]).await;
        let mut delivered = Vec::new();

        super::super::stream::sse_stream_with_policy(
            post(&reqwest::Client::new(), &url),
            |message| {
                delivered.push(message.data);
                Ok(false)
            },
            quick_policy(4),
        )
        .await
        .unwrap();

        assert_eq!(delivered, vec!["partial-output"]);
        assert_eq!(requests.load(Ordering::SeqCst), 1);
        server.await.unwrap();
    }
    #[test]
    fn retry_admission_budget_preserves_a_provider_requested_three_second_wait() {
        let body = ResponseBody {
            bytes: br#"{"error":{"code":"rate_limit_exceeded","message":"provider rate limit","details":[{"request_id":"budget-error-123"}]}}"#.to_vec(),
            truncated: false,
            read_error: None,
        };
        let mut headers = HeaderMap::new();
        headers.insert(RETRY_AFTER, "3".parse().unwrap());
        let mut policy = quick_policy(4);
        policy.max_duration = Duration::from_millis(100);
        let mut state = RetryState::new(policy);
        let error = state
            .retry_response(StatusCode::TOO_MANY_REQUESTS, &headers, &body, true)
            .unwrap_err()
            .to_string();

        assert!(error.contains("next wait (3.00s)"), "{error}");
        assert!(
            error.contains("remaining retry admission budget"),
            "{error}"
        );
        assert!(error.contains("provider rate limit"), "{error}");
        assert!(error.contains("rate_limit_exceeded"), "{error}");
        assert!(error.contains("budget-error-123"), "{error}");
        assert_eq!(state.attempt, 1);
    }

    #[tokio::test]
    async fn preserves_latest_provider_response_when_attempts_are_exhausted() {
        let first = r#"{"error":{"message":"first provider response"}}"#;
        let latest = r#"{"error":{"code":"rate_limit_exceeded","message":"latest provider response","details":[{"request_id":"latest-request-456","quota":"per-minute"}]}}"#;
        let (url, requests, server) = spawn_server(vec![
            MockResponse::json(429, first).header("Retry-After", "0"),
            MockResponse::json(429, latest).header("Retry-After", "0"),
        ])
        .await;
        let error = send_with_retry_policy(post(&reqwest::Client::new(), &url), quick_policy(2))
            .await
            .unwrap_err()
            .to_string();

        assert!(error.contains("HTTP 429"), "{error}");
        assert!(error.contains("latest provider response"), "{error}");
        assert!(error.contains("rate_limit_exceeded"), "{error}");
        assert!(error.contains("latest-request-456"), "{error}");
        assert!(error.contains("per-minute"), "{error}");
        assert!(
            error.contains("HTTP recovery exhausted after 2 attempts"),
            "{error}"
        );
        assert!(!error.contains("first provider response"), "{error}");
        assert_eq!(requests.load(Ordering::SeqCst), 2);
        server.await.unwrap();
    }

    #[tokio::test]
    async fn preserves_provider_response_when_a_retry_fails_at_transport_layer() {
        let body = r#"{"error":{"code":"rate_limit_exceeded","message":"provider asked us to retry","details":[{"request_id":"transport-followup-789","quota":"minute-limit"}]}}"#;
        let (url, requests, server) = spawn_server(vec![
            MockResponse::json(429, body).header("Retry-After", "0")
        ])
        .await;
        let error = send_with_retry_policy(post(&reqwest::Client::new(), &url), quick_policy(2))
            .await
            .unwrap_err()
            .to_string();

        assert!(error.contains("HTTP 429"), "{error}");
        assert!(error.contains("provider asked us to retry"), "{error}");
        assert!(error.contains("rate_limit_exceeded"), "{error}");
        assert!(error.contains("transport-followup-789"), "{error}");
        assert!(error.contains("minute-limit"), "{error}");
        assert!(
            error.contains("Connection failure before a provider HTTP response was received"),
            "{error}"
        );
        assert!(
            error.contains("HTTP recovery exhausted after 2 attempts"),
            "{error}"
        );
        assert_eq!(requests.load(Ordering::SeqCst), 1);
        server.await.unwrap();
    }

    #[tokio::test]
    async fn sse_handshake_failure_preserves_provider_response_and_transport_error() {
        let body = r#"{"error":{"code":"overloaded_error","message":"stream handshake rejected","details":[{"request_id":"sse-handshake-321","reason":"upstream busy"}]}}"#;
        let (url, requests, server) = spawn_server(vec![
            MockResponse::json(503, body).header("Retry-After", "0")
        ])
        .await;
        let error = super::super::stream::sse_stream_with_policy(
            post(&reqwest::Client::new(), &url),
            |_| Ok(false),
            quick_policy(2),
        )
        .await
        .unwrap_err()
        .to_string();

        assert!(error.contains("HTTP 503"), "{error}");
        assert!(error.contains("stream handshake rejected"), "{error}");
        assert!(error.contains("overloaded_error"), "{error}");
        assert!(error.contains("sse-handshake-321"), "{error}");
        assert!(error.contains("upstream busy"), "{error}");
        assert!(
            error.contains("Connection failure before a provider HTTP response was received"),
            "{error}"
        );
        assert_eq!(requests.load(Ordering::SeqCst), 1);
        server.await.unwrap();
    }

    fn test_stream_handler() -> (
        crate::client::SseHandler,
        tokio::sync::mpsc::UnboundedReceiver<crate::client::SseEvent>,
    ) {
        let (tx, rx) = tokio::sync::mpsc::unbounded_channel();
        (
            crate::client::SseHandler::new(tx, crate::utils::AbortSignalInner::new()),
            rx,
        )
    }

    #[tokio::test]
    async fn anthropic_stream_error_preserves_details_redacts_secrets_and_never_replays() {
        let body = concat!(
            "event: content_block_delta\n",
            "data: {\"type\":\"content_block_delta\",\"delta\":{\"text\":\"partial output\"}}\n\n",
            "event: error\n",
            "data: {\"type\":\"error\",\"error\":{\"type\":\"overloaded_error\",\"message\":\"Claude is overloaded\",\"request_id\":\"req-claude-123\",\"provider_detail\":\"upstream capacity exhausted\",\"api_key\":\"anthropic-test-secret\"}}\n\n"
        );
        let (url, requests, server) = spawn_server(vec![MockResponse::sse(body)]).await;
        let (mut handler, _rx) = test_stream_handler();
        let error = super::super::claude::claude_chat_completions_streaming(
            post(&reqwest::Client::new(), &url),
            &mut handler,
            &crate::client::Model::new("claude", "claude-sonnet"),
        )
        .await
        .unwrap_err()
        .to_string();

        for expected in [
            "in-stream error (error)",
            "overloaded_error",
            "Claude is overloaded",
            "req-claude-123",
            "upstream capacity exhausted",
        ] {
            assert!(error.contains(expected), "{error} missing {expected}");
        }
        assert!(!error.contains("anthropic-test-secret"), "{error}");
        assert!(error.contains("[REDACTED]"), "{error}");
        assert_eq!(requests.load(Ordering::SeqCst), 1);
        server.await.unwrap();
    }

    #[tokio::test]
    async fn anthropic_normal_content_reasoning_and_tool_events_are_preserved() {
        let body = concat!(
            "event: content_block_delta\n",
            "data: {\"type\":\"content_block_delta\",\"delta\":{\"thinking\":\"checking the result\"}}\n\n",
            "event: content_block_stop\n",
            "data: {\"type\":\"content_block_stop\"}\n\n",
            "event: content_block_delta\n",
            "data: {\"type\":\"content_block_delta\",\"delta\":{\"text\":\"answer\"}}\n\n",
            "event: content_block_start\n",
            "data: {\"type\":\"content_block_start\",\"content_block\":{\"type\":\"tool_use\",\"name\":\"lookup\",\"id\":\"tool-1\"}}\n\n",
            "event: content_block_delta\n",
            "data: {\"type\":\"content_block_delta\",\"delta\":{\"partial_json\":\"{\\\"query\\\":\\\"rust\\\"}\"}}\n\n",
            "event: content_block_stop\n",
            "data: {\"type\":\"content_block_stop\"}\n\n"
        );
        let (url, requests, server) = spawn_server(vec![MockResponse::sse(body)]).await;
        let (mut handler, _rx) = test_stream_handler();
        super::super::claude::claude_chat_completions_streaming(
            post(&reqwest::Client::new(), &url),
            &mut handler,
            &crate::client::Model::new("claude", "claude-sonnet"),
        )
        .await
        .unwrap();

        let (text, calls) = handler.take();
        assert!(
            text.contains("<think>\nchecking the result\n</think>"),
            "{text}"
        );
        assert!(text.contains("answer"), "{text}");
        assert_eq!(calls.len(), 1);
        assert_eq!(calls[0].name, "lookup");
        assert_eq!(calls[0].arguments, json!({"query": "rust"}));
        assert_eq!(requests.load(Ordering::SeqCst), 1);
        server.await.unwrap();
    }

    #[tokio::test]
    async fn openai_stream_error_envelope_preserves_details_redacts_secrets_and_never_replays() {
        let body = "data: {\"error\":{\"type\":\"rate_limit_error\",\"code\":\"rate_limit_exceeded\",\"message\":\"OpenAI stream rate limited\",\"request_id\":\"req-openai-456\",\"api_key\":\"openai-test-secret\"}}\n\n";
        let (url, requests, server) = spawn_server(vec![MockResponse::sse(body)]).await;
        let (mut handler, _rx) = test_stream_handler();
        let error = super::super::openai::openai_chat_completions_streaming(
            post(&reqwest::Client::new(), &url),
            &mut handler,
            &crate::client::Model::new("openai", "gpt-test"),
        )
        .await
        .unwrap_err()
        .to_string();

        for expected in [
            "rate_limit_error",
            "rate_limit_exceeded",
            "OpenAI stream rate limited",
            "req-openai-456",
        ] {
            assert!(error.contains(expected), "{error} missing {expected}");
        }
        assert!(!error.contains("openai-test-secret"), "{error}");
        assert!(error.contains("[REDACTED]"), "{error}");
        assert_eq!(requests.load(Ordering::SeqCst), 1);
        server.await.unwrap();
    }

    #[tokio::test]
    async fn openai_normal_content_reasoning_and_tool_events_are_preserved() {
        let body = concat!(
            "data: {\"choices\":[{\"delta\":{\"reasoning_content\":\"planning\"}}]}\n\n",
            "data: {\"choices\":[{\"delta\":{\"content\":\"answer\"}}]}\n\n",
            "data: {\"choices\":[{\"delta\":{\"tool_calls\":[{\"index\":0,\"id\":\"call-1\",\"type\":\"function\",\"function\":{\"name\":\"lookup\",\"arguments\":\"{\\\"query\\\":\\\"rust\\\"}\"}}]}}]}\n\n",
            "data: [DONE]\n\n"
        );
        let (url, requests, server) = spawn_server(vec![MockResponse::sse(body)]).await;
        let (mut handler, _rx) = test_stream_handler();
        super::super::openai::openai_chat_completions_streaming(
            post(&reqwest::Client::new(), &url),
            &mut handler,
            &crate::client::Model::new("openai", "gpt-test"),
        )
        .await
        .unwrap();

        let (text, calls) = handler.take();
        assert!(text.contains("<think>\nplanning\n</think>"), "{text}");
        assert!(text.contains("answer"), "{text}");
        assert_eq!(calls.len(), 1);
        assert_eq!(calls[0].name, "lookup");
        assert_eq!(calls[0].arguments, json!({"query": "rust"}));
        assert_eq!(requests.load(Ordering::SeqCst), 1);
        server.await.unwrap();
    }

    #[tokio::test]
    async fn gemini_stream_error_envelope_preserves_details_redacts_secrets_and_never_replays() {
        let body = r#"[{"error":{"code":429,"message":"Gemini quota exhausted","status":"RESOURCE_EXHAUSTED","details":[{"reason":"per-minute quota"}],"api_key":"gemini-test-secret"}}]"#;
        let (url, requests, server) = spawn_server(vec![MockResponse::json(200, body)]).await;
        let (mut handler, _rx) = test_stream_handler();
        let error = super::super::vertexai::gemini_chat_completions_streaming(
            post(&reqwest::Client::new(), &url),
            &mut handler,
            &crate::client::Model::new("gemini", "gemini-test"),
        )
        .await
        .unwrap_err()
        .to_string();

        for expected in [
            "in-stream error (error)",
            "Gemini quota exhausted",
            "RESOURCE_EXHAUSTED",
            "per-minute quota",
        ] {
            assert!(error.contains(expected), "{error} missing {expected}");
        }
        assert!(!error.contains("gemini-test-secret"), "{error}");
        assert!(error.contains("[REDACTED]"), "{error}");
        assert_eq!(requests.load(Ordering::SeqCst), 1);
        server.await.unwrap();
    }

    #[test]
    fn bedrock_stream_exception_diagnostics_are_redacted_without_losing_provider_details() {
        let error = format_stream_error(
            "modelStreamErrorException",
            r#"{"originalStatusCode":429,"originalMessage":"Model overloaded; retry later","requestId":"bedrock-789","secret":"bedrock-test-secret"}"#,
        );

        for expected in [
            "modelStreamErrorException",
            "429",
            "Model overloaded; retry later",
            "bedrock-789",
        ] {
            assert!(error.contains(expected), "{error} missing {expected}");
        }
        assert!(!error.contains("bedrock-test-secret"), "{error}");
        assert!(error.contains("[REDACTED]"), "{error}");
    }
}
