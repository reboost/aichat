use anyhow::{anyhow, Result};
use chrono::{DateTime, Utc};
use futures_util::StreamExt;
use reqwest::{
    header::{HeaderMap, RETRY_AFTER},
    RequestBuilder, Response, StatusCode,
};
use serde_json::Value;
use std::time::{Duration, Instant, SystemTime, UNIX_EPOCH};

const DEFAULT_MAX_ATTEMPTS: usize = 4;
const DEFAULT_MAX_DURATION_SECS: u64 = 30;
const MAX_CONFIGURED_ATTEMPTS: usize = 10;
const MAX_CONFIGURED_DURATION_SECS: u64 = 300;
const BASE_DELAY_MS: u64 = 250;
const MAX_BACKOFF_MS: u64 = 8_000;
const MAX_ERROR_RESPONSE_BYTES: usize = 64 * 1024;

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
            .and_then(|v| v.parse::<usize>().ok())
            .unwrap_or(DEFAULT_MAX_ATTEMPTS)
            .clamp(1, MAX_CONFIGURED_ATTEMPTS);
        let max_duration_secs = std::env::var("AICHAT_HTTP_RETRY_MAX_DURATION_SECS")
            .ok()
            .and_then(|v| v.parse::<u64>().ok())
            .unwrap_or(DEFAULT_MAX_DURATION_SECS)
            .min(MAX_CONFIGURED_DURATION_SECS);

        Self {
            max_attempts,
            max_duration: Duration::from_secs(max_duration_secs),
            base_delay: Duration::from_millis(BASE_DELAY_MS),
            max_backoff: Duration::from_millis(MAX_BACKOFF_MS),
        }
    }
}

pub(super) struct ResponseBody {
    pub(super) bytes: Vec<u8>,
    pub(super) truncated: bool,
}

pub(super) async fn read_response_body(response: Response) -> Result<ResponseBody> {
    let status = response.status();
    let mut stream = response.bytes_stream();
    let mut bytes = Vec::new();
    let mut truncated = false;

    while let Some(chunk) = stream.next().await {
        let chunk = chunk.map_err(|err| {
            anyhow!(
                "Failed to read provider error response (HTTP {}): {}",
                status.as_u16(),
                err.without_url()
            )
        })?;
        let remaining = MAX_ERROR_RESPONSE_BYTES.saturating_sub(bytes.len());
        if chunk.len() > remaining {
            bytes.extend_from_slice(&chunk[..remaining]);
            truncated = true;
            break;
        }
        bytes.extend_from_slice(&chunk);
    }

    Ok(ResponseBody { bytes, truncated })
}

pub(super) fn response_body_text(body: &ResponseBody) -> String {
    let text = redact_secret_text(&String::from_utf8_lossy(&body.bytes));
    if body.truncated {
        format!("{text}\n[response body truncated after {MAX_ERROR_RESPONSE_BYTES} bytes]")
    } else {
        text
    }
}

pub(super) fn format_response_error(status: StatusCode, body: &ResponseBody) -> String {
    response_error_message(status, body)
}

pub(super) struct RetryState {
    policy: RetryPolicy,
    started: Instant,
    attempt: usize,
    last_diagnostic: Option<String>,
}

impl RetryState {
    pub(super) fn new(policy: RetryPolicy) -> Self {
        Self {
            policy,
            started: Instant::now(),
            attempt: 1,
            last_diagnostic: None,
        }
    }

    pub(super) fn retry_response(
        &mut self,
        status: StatusCode,
        headers: &HeaderMap,
        body: &ResponseBody,
        allow_retry: bool,
    ) -> Result<Duration> {
        let diagnostic = response_error_message(status, body);
        self.last_diagnostic = Some(diagnostic.clone());

        let structured_hint = structured_retry_hint(&body.bytes);
        let header_retry_after = parse_retry_after_header(headers);
        let retry_after = header_retry_after.or(structured_hint.delay);
        let has_retry_signal = header_retry_after.is_some()
            || structured_hint.explicit
            || contains_text_retry_hint(&body.bytes);

        if let Some(reason) = permanent_failure(&body.bytes, has_retry_signal) {
            anyhow::bail!("{diagnostic}. {reason}");
        }
        if !allow_retry || !is_retryable_status(status) {
            anyhow::bail!("{diagnostic}");
        }

        match self.schedule(
            &format!("HTTP {}", status.as_u16()),
            retry_after,
            Some(&diagnostic),
        ) {
            Ok(delay) => Ok(delay),
            Err(err) => Err(self.with_last_diagnostic(err)),
        }
    }

    pub(super) fn retry_connection(&mut self) -> Result<Duration> {
        let diagnostic = self.last_diagnostic.clone();
        match self.schedule(
            "connection failure before request submission",
            None,
            diagnostic.as_deref(),
        ) {
            Ok(delay) => Ok(delay),
            Err(err) => Err(self.with_last_diagnostic(err)),
        }
    }

    pub(super) fn with_last_diagnostic(&self, err: anyhow::Error) -> anyhow::Error {
        match &self.last_diagnostic {
            Some(diagnostic) if !err.to_string().contains(diagnostic) => {
                anyhow!("{diagnostic}; {err}")
            }
            _ => err,
        }
    }

    fn schedule(
        &mut self,
        reason: &str,
        retry_after: Option<Duration>,
        diagnostic: Option<&str>,
    ) -> Result<Duration> {
        if self.attempt >= self.policy.max_attempts {
            anyhow::bail!(
                "HTTP recovery exhausted after {} attempts ({reason})",
                self.attempt
            );
        }

        let delay = self.backoff_delay().max(retry_after.unwrap_or_default());
        if self.started.elapsed().saturating_add(delay) > self.policy.max_duration {
            anyhow::bail!(
                "HTTP recovery stopped after {} attempts because the next delay ({:.2}s) exceeds the maximum retry duration ({:.2}s); {reason}",
                self.attempt,
                delay.as_secs_f64(),
                self.policy.max_duration.as_secs_f64()
            );
        }

        self.attempt += 1;
        let attempt = self.attempt;
        let max_attempts = self.policy.max_attempts;
        let delay_secs = delay.as_secs_f64();
        if let Some(diagnostic) = diagnostic {
            let diagnostic = diagnostic.chars().take(1_000).collect::<String>();
            log::warn!(
                "Provider request failed ({reason}): {diagnostic}; retrying attempt {attempt} of {max_attempts} in {delay_secs:.2}s"
            );
        } else {
            log::warn!(
                "Provider request failed ({reason}); retrying attempt {attempt} of {max_attempts} in {delay_secs:.2}s"
            );
        }
        Ok(delay)
    }

    fn backoff_delay(&self) -> Duration {
        let exponent = self.attempt.saturating_sub(1).min(16) as u32;
        let base_ms = self.policy.base_delay.as_millis() as u64;
        let upper_ms = base_ms
            .saturating_mul(1_u64 << exponent)
            .min(self.policy.max_backoff.as_millis() as u64);
        let lower_ms = upper_ms / 2;
        let jitter_range = upper_ms.saturating_sub(lower_ms).saturating_add(1);
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
            Err(err) if allow_retry && err.is_connect() => {
                let delay = state.retry_connection()?;
                tokio::time::sleep(delay).await;
                continue;
            }
            Err(err) if err.is_timeout() => {
                let error = anyhow!(
                    "Provider request timed out and was not replayed because the provider may have received it: {}",
                    err.without_url()
                );
                return Err(state.with_last_diagnostic(error));
            }
            Err(err) => {
                return Err(state.with_last_diagnostic(err.without_url().into()));
            }
        };

        if response.status().is_success() {
            return Ok(response);
        }

        let status = response.status();
        let headers = response.headers().clone();
        let body = read_response_body(response)
            .await
            .map_err(|err| state.with_last_diagnostic(err))?;
        let delay = state.retry_response(status, &headers, &body, allow_retry)?;
        tokio::time::sleep(delay).await;
    }
}

pub(super) fn is_retryable_status(status: StatusCode) -> bool {
    matches!(status.as_u16(), 408 | 425 | 429 | 500 | 502 | 503 | 504 | 529)
}

fn parse_retry_after_header(headers: &HeaderMap) -> Option<Duration> {
    let value = headers.get(RETRY_AFTER)?.to_str().ok()?;
    if let Ok(seconds) = value.parse::<u64>() {
        return Some(Duration::from_secs(seconds));
    }

    let retry_at = DateTime::parse_from_rfc2822(value).ok()?.timestamp();
    let seconds = retry_at.saturating_sub(Utc::now().timestamp());
    Some(Duration::from_secs(seconds.max(0) as u64))
}

#[derive(Default)]
struct StructuredRetryHint {
    explicit: bool,
    delay: Option<Duration>,
}

fn structured_retry_hint(body: &[u8]) -> StructuredRetryHint {
    fn visit(value: &Value, retry_info: bool, hint: &mut StructuredRetryHint) {
        match value {
            Value::Object(object) => {
                let is_retry_info = retry_info
                    || object
                        .get("@type")
                        .and_then(Value::as_str)
                        .is_some_and(|value| value.ends_with("google.rpc.RetryInfo"));
                if is_retry_info {
                    hint.explicit = true;
                }

                for (key, value) in object {
                    let key = key.to_ascii_lowercase().replace('_', "").replace('-', "");
                    if key == "retryinfo" {
                        hint.explicit = true;
                    } else if matches!(key.as_str(), "retrydelay" | "retryafter") {
                        hint.explicit = true;
                        if hint.delay.is_none() {
                            hint.delay = parse_retry_delay(value);
                        }
                    }
                    visit(value, is_retry_info, hint);
                }
            }
            Value::Array(values) => {
                for value in values {
                    visit(value, retry_info, hint);
                }
            }
            _ => {}
        }
    }

    let mut hint = StructuredRetryHint::default();
    if let Ok(value) = serde_json::from_slice::<Value>(body) {
        visit(&value, false, &mut hint);
    }
    hint
}

fn parse_retry_delay(value: &Value) -> Option<Duration> {
    let seconds = match value {
        Value::Number(number) => number.as_f64()?,
        Value::String(value) => {
            let value = value.trim();
            if let Some(value) = value.strip_suffix("ms") {
                value.parse::<f64>().ok()? / 1_000.0
            } else if let Some(value) = value.strip_suffix('s') {
                value.parse::<f64>().ok()?
            } else {
                value.parse::<f64>().ok()?
            }
        }
        _ => return None,
    };
    if !seconds.is_finite() || seconds < 0.0 {
        return None;
    }
    Some(Duration::from_secs_f64(
        seconds.min(MAX_CONFIGURED_DURATION_SECS as f64 + 1.0),
    ))
}

fn contains_text_retry_hint(body: &[u8]) -> bool {
    let text = String::from_utf8_lossy(body).to_ascii_lowercase();
    [
        "retryinfo",
        "retry_after",
        "retry-after",
        "retry after",
        "retry in",
        "try again",
        "rate_limit",
        "rate limit",
        "perminute",
        "per_minute",
        "per minute",
        "per second",
        "per-second",
        "requests per minute",
    ]
    .iter()
    .any(|value| text.contains(value))
}

fn permanent_failure(body: &[u8], has_retry_hint: bool) -> Option<&'static str> {
    let text = String::from_utf8_lossy(body).to_ascii_lowercase();

    let permanent = [
        "insufficient_quota",
        "billing_hard_limit_reached",
        "billing_not_active",
        "billing disabled",
        "billing error",
        "billing_error",
        "billing_limit_exceeded",
        "insufficient_credits",
        "credit_limit_exceeded",
        "payment_required",
        "payment required",
        "credit balance is too low",
        "account_deactivated",
        "account is deactivated",
        "invalid_api_key",
        "authentication_error",
        "invalid authentication",
        "invalid_request_error",
    ];
    if permanent.iter().any(|value| text.contains(value)) {
        return Some(
            "Check the provider's billing status, quota, credentials, and request configuration",
        );
    }

    let hard_exhausted_quota = [
        "permanent quota",
        "exceeded your current quota",
        "quota is exhausted",
        "quota exhausted",
        "no remaining quota",
        "out of quota",
        "daily quota exhausted",
        "monthly quota exhausted",
        "daily quota limit exceeded",
        "monthly quota limit exceeded",
        "daily limit reached",
        "monthly limit reached",
        "per-day quota exhausted",
        "per-month quota exhausted",
    ]
    .iter()
    .any(|value| text.contains(value));
    if hard_exhausted_quota {
        return Some(
            "The provider reports an exhausted quota; check its quota limits or billing settings",
        );
    }

    let ambiguous_quota = [
        "servicequotaexceeded",
        "quotaexceededexception",
        "quota_exceeded",
        "quota exceeded",
        "quota has been exceeded",
        "daily quota",
        "monthly quota",
        "daily limit",
        "monthly limit",
    ]
    .iter()
    .any(|value| text.contains(value));

    if ambiguous_quota && !has_retry_hint {
        Some("The provider reports an exhausted quota; check its quota limits or billing settings")
    } else {
        None
    }
}

fn response_error_message(status: StatusCode, body: &ResponseBody) -> String {
    let body_text = redact_secret_text(&String::from_utf8_lossy(&body.bytes));
    let mut diagnostic = match serde_json::from_slice::<Value>(&body.bytes) {
        Ok(mut data) => {
            redact_json_secrets(&mut data);
            let message = first_string(
                &data,
                &[
                    "/error/message",
                    "/message",
                    "/detail",
                    "/error",
                    "/errors/0/message",
                    "/errors/0/title",
                    "/Message",
                    "/Error/Message",
                    "/title",
                ],
            );
            let error_type = first_string(&data, &["/error/type", "/__type", "/type"]);
            let code = first_string(
                &data,
                &[
                    "/error/code",
                    "/code",
                    "/errors/0/code",
                    "/errors/0/reason",
                ],
            );
            let provider_status = first_string(&data, &["/error/status", "/status"]);
            let details = first_value(
                &data,
                &[
                    "/error/details",
                    "/details",
                    "/error/innererror",
                    "/error/inner_error",
                    "/errors",
                ],
            );

            let mut parts = Vec::new();
            if let Some(message) = message {
                parts.push(message);
            }
            if let Some(error_type) = error_type {
                parts.push(format!("type: {error_type}"));
            }
            if let Some(code) = code {
                parts.push(format!("code: {code}"));
            }
            if let Some(provider_status) = provider_status {
                parts.push(format!("provider status: {provider_status}"));
            }
            if let Some(details) = details {
                if let Ok(details) = serde_json::to_string(details) {
                    parts.push(format!("details: {details}"));
                }
            }
            if parts.is_empty() {
                serde_json::to_string(&data).unwrap_or(body_text)
            } else {
                parts.join("; ")
            }
        }
        Err(_) if body_text.trim().is_empty() => "empty response body".to_string(),
        Err(_) => body_text,
    };

    if body.truncated {
        diagnostic.push_str(&format!(
            " [response body truncated after {MAX_ERROR_RESPONSE_BYTES} bytes]"
        ));
    }
    format!("Provider returned HTTP {}: {diagnostic}", status.as_u16())
}

fn first_string(value: &Value, paths: &[&str]) -> Option<String> {
    paths.iter().find_map(|path| {
        let value = value.pointer(path)?;
        match value {
            Value::String(value) if !value.trim().is_empty() => Some(value.clone()),
            Value::Number(value) => Some(value.to_string()),
            _ => None,
        }
    })
}

fn first_value<'a>(value: &'a Value, paths: &[&str]) -> Option<&'a Value> {
    paths.iter().find_map(|path| value.pointer(path))
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
                    "key" | "apikey" | "authorization" | "token" | "credential"
                ) || key.contains("apikey")
                    || key.contains("accesstoken")
                    || key.contains("refreshtoken")
                    || key.contains("secret")
                    || key.contains("password")
                    || key.contains("credential")
                    || key.contains("privatekey");
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
        .filter(|value| !value.is_control() || matches!(*value, '\n' | '\t'))
        .collect()
}

fn redact_assignment_values(input: &str) -> String {
    const MARKERS: &[&str] = &[
        "authorization",
        "access_token",
        "access-token",
        "accesskey",
        "secret_access_key",
        "secret-access-key",
        "refresh_token",
        "session_token",
        "session-token",
        "client_secret",
        "private_key",
        "private-key",
        "secret_key",
        "secret-key",
        "api_key",
        "api-key",
        "x-api-key",
        "x_api_key",
        "apikey",
        "password",
        "credential",
        "secret",
        "token",
        "key",
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
            value[value_start..]
                .as_bytes()
                .iter()
                .position(|byte| *byte == quote)
                .map(|offset| value_start + offset)
                .unwrap_or(value.len())
        } else {
            value[value_start..]
                .as_bytes()
                .iter()
                .position(|byte| {
                    byte.is_ascii_whitespace()
                        || matches!(*byte, b',' | b';' | b'&' | b'}' | b']')
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
        let token_end = value[token_start..]
            .as_bytes()
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
        let token_end = value[start..]
            .as_bytes()
            .iter()
            .position(|byte| {
                !(byte.is_ascii_alphanumeric() || matches!(*byte, b'-' | b'_' | b'.'))
            })
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
    }

    impl MockResponse {
        fn json(status: u16, body: &str) -> Self {
            Self {
                status,
                headers: Vec::new(),
                content_type: "application/json",
                body: body.to_string(),
                delay: Duration::ZERO,
            }
        }

        fn sse(body: &str) -> Self {
            Self {
                status: 200,
                headers: Vec::new(),
                content_type: "text/event-stream",
                body: body.to_string(),
                delay: Duration::ZERO,
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
                    body.len(),
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
        };
        let diagnostic = format_response_error(StatusCode::BAD_GATEWAY, &body);
        assert!(diagnostic.contains("HTTP 502"));
        assert!(diagnostic.contains("gateway said unavailable"));
    }

    #[test]
    fn redacts_credentials_from_json_and_unparsed_error_bodies() {
        let json_body = ResponseBody {
            bytes: br#"{"service":{"api_key":"json-secret-key","refresh_token":"refresh-secret","note":"Authorization: Bearer header-secret"}}"#.to_vec(),
            truncated: false,
        };
        let diagnostic = format_response_error(StatusCode::BAD_REQUEST, &json_body);
        for secret in ["json-secret-key", "refresh-secret", "header-secret"] {
            assert!(!diagnostic.contains(secret), "{diagnostic}");
        }
        assert!(diagnostic.contains("[REDACTED]"));

        let raw_body = ResponseBody {
            bytes: b"upstream echoed Authorization: Bearer plain-secret-token and api_key=other-secret"
                .to_vec(),
            truncated: false,
        };
        let diagnostic = format_response_error(StatusCode::BAD_GATEWAY, &raw_body);
        for secret in ["plain-secret-token", "other-secret"] {
            assert!(!diagnostic.contains(secret), "{diagnostic}");
        }
    }

    #[tokio::test]
    async fn retries_gemini_quota_error_with_nested_google_retry_info() {
        let body = r#"{"error":{"code":429,"message":"Quota exceeded for this request","status":"RESOURCE_EXHAUSTED","details":[{"@type":"type.googleapis.com/google.rpc.RetryInfo","retryDelay":"0.08s"}]}}"#;
        let (url, requests, server) = spawn_server(vec![
            MockResponse::json(429, body),
            MockResponse::json(200, r#"{"ok":true}"#),
        ])
        .await;
        let started = Instant::now();
        let response = send_with_retry_policy(
            post(&reqwest::Client::new(), &url),
            quick_policy(3),
        )
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
        let error = send_with_retry_policy(
            post(&reqwest::Client::new(), &url),
            quick_policy(2),
        )
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
        let body = format!("proxy response: {}", "x".repeat(MAX_ERROR_RESPONSE_BYTES + 1_024));
        let (url, requests, server) =
            spawn_server(vec![MockResponse::json(400, &body)]).await;
        let error = send_with_retry_policy(
            post(&reqwest::Client::new(), &url),
            quick_policy(3),
        )
        .await
        .unwrap_err()
        .to_string();

        assert!(error.contains("HTTP 400"), "{error}");
        assert!(error.contains("response body truncated"), "{error}");
        assert!(error.len() <= MAX_ERROR_RESPONSE_BYTES + 256, "{}", error.len());
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

    #[test]
    fn identifies_permanent_quota_errors_and_retry_after_values() {
        assert!(permanent_failure(br#"{"error":{"code":"insufficient_quota"}}"#, false).is_some());
        assert!(permanent_failure(br#"{"error":{"type":"billing_error"}}"#, false).is_some());
        assert!(permanent_failure(br#"{"error":{"reason":"QUOTA_EXCEEDED"}}"#, false).is_some());
        assert!(permanent_failure(
            br#"{"error":{"reason":"RATE_LIMIT_EXCEEDED","message":"Quota exceeded per minute"}}"#,
            false
        )
        .is_none());
        assert!(permanent_failure(br#"{"error":{"message":"Quota exceeded"}}"#, true).is_none());
        assert!(permanent_failure(
            br#"{"error":{"code":"ServiceQuotaExceededException"}}"#,
            false
        )
        .is_some());
        assert!(
            permanent_failure(br#"{"error":{"code":"QuotaExceededException"}}"#, false).is_some()
        );

        let mut headers = HeaderMap::new();
        headers.insert(RETRY_AFTER, "2".parse().unwrap());
        assert_eq!(parse_retry_after_header(&headers), Some(Duration::from_secs(2)));
        let date = (Utc::now() + chrono::Duration::seconds(2))
            .format("%a, %d %b %Y %H:%M:%S GMT")
            .to_string();
        headers.insert(RETRY_AFTER, date.parse().unwrap());
        assert!(parse_retry_after_header(&headers).unwrap() >= Duration::from_secs(1));

        let hint = structured_retry_hint(
            br#"{"error":{"details":[{"@type":"type.googleapis.com/google.rpc.RetryInfo","retryDelay":"1.5s"}]}}"#,
        );
        assert!(hint.explicit);
        assert_eq!(hint.delay, Some(Duration::from_millis(1500)));
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

        assert!(error.to_string().contains("maximum retry duration"));
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
    async fn does_not_retry_hard_quota_or_billing_failures_with_retry_after() {
        let cases = [
            r#"{"error":{"code":"ServiceQuotaExceededException","message":"Daily quota exhausted","details":[{"@type":"type.googleapis.com/google.rpc.RetryInfo","retryDelay":"1s"}]}}"#,
            r#"{"error":{"type":"insufficient_quota","message":"You exceeded your current quota","details":[{"@type":"type.googleapis.com/google.rpc.RetryInfo","retryDelay":"1s"}]}}"#,
            r#"{"error":{"type":"billing_hard_limit_reached","message":"Billing hard limit reached","details":[{"@type":"type.googleapis.com/google.rpc.RetryInfo","retryDelay":"1s"}]}}"#,
            r#"{"error":{"message":"Monthly quota exhausted","details":[{"@type":"type.googleapis.com/google.rpc.RetryInfo","retryDelay":"1s"}]}}"#,
        ];
        for body in cases {
            let (url, requests, server) = spawn_server(vec![
                MockResponse::json(429, body).header("Retry-After", "0")
            ])
            .await;
            let error =
                send_with_retry_policy(post(&reqwest::Client::new(), &url), quick_policy(4))
                    .await
                    .unwrap_err();

            assert!(error.to_string().contains("HTTP 429"));
            assert_eq!(requests.load(Ordering::SeqCst), 1, "{body}");
            server.await.unwrap();
        }
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
}
