use anyhow::{anyhow, Context, Result};
use chrono::{DateTime, Utc};
use reqwest::{
    header::{HeaderMap, RETRY_AFTER},
    RequestBuilder, Response, StatusCode,
};
use std::time::{Duration, Instant, SystemTime, UNIX_EPOCH};

const DEFAULT_MAX_ATTEMPTS: usize = 4;
const DEFAULT_MAX_DURATION_SECS: u64 = 30;
const MAX_CONFIGURED_ATTEMPTS: usize = 10;
const MAX_CONFIGURED_DURATION_SECS: u64 = 300;
const BASE_DELAY_MS: u64 = 250;
const MAX_BACKOFF_MS: u64 = 8_000;

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

pub(super) struct RetryState {
    policy: RetryPolicy,
    started: Instant,
    attempt: usize,
}

impl RetryState {
    pub(super) fn new(policy: RetryPolicy) -> Self {
        Self {
            policy,
            started: Instant::now(),
            attempt: 1,
        }
    }

    pub(super) fn retry_response(
        &mut self,
        status: StatusCode,
        headers: &HeaderMap,
        body: &[u8],
    ) -> Result<Duration> {
        let message = response_error_message(status, body);
        let retry_after = parse_retry_after(headers);
        if let Some(reason) = permanent_failure(body, retry_after.is_some()) {
            anyhow::bail!("{message}. {reason}");
        }
        if !is_retryable_status(status) {
            anyhow::bail!("{message}");
        }

        self.schedule(&format!("HTTP {}", status.as_u16()), retry_after)
            .map_err(|err| anyhow!("{message}; {err}"))
    }

    pub(super) fn retry_connection(&mut self) -> Result<Duration> {
        self.schedule("connection failure before request submission", None)
    }

    fn schedule(&mut self, reason: &str, retry_after: Option<Duration>) -> Result<Duration> {
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
    let mut state = RetryState::new(policy);

    loop {
        let request = match builder.try_clone() {
            Some(request) => request,
            None => return builder.send().await.map_err(Into::into),
        };

        match request.send().await {
            Ok(response) => {
                let status = response.status();
                if !is_retryable_status(status) {
                    return Ok(response);
                }

                let headers = response.headers().clone();
                let body = response.bytes().await.with_context(|| {
                    format!(
                        "Failed to read provider error response (HTTP {})",
                        status.as_u16()
                    )
                })?;
                let delay = state.retry_response(status, &headers, &body)?;
                tokio::time::sleep(delay).await;
            }
            Err(err) if err.is_connect() => {
                let delay = state.retry_connection()?;
                tokio::time::sleep(delay).await;
            }
            Err(err) if err.is_timeout() => {
                anyhow::bail!(
                    "Provider request timed out and was not replayed because the provider may have received it"
                );
            }
            Err(err) => return Err(err.without_url().into()),
        }
    }
}

pub(super) fn is_retryable_status(status: StatusCode) -> bool {
    matches!(status.as_u16(), 429 | 500 | 502 | 503 | 504 | 529)
}

fn parse_retry_after(headers: &HeaderMap) -> Option<Duration> {
    let value = headers.get(RETRY_AFTER)?.to_str().ok()?;
    if let Ok(seconds) = value.parse::<u64>() {
        return Some(Duration::from_secs(seconds));
    }

    let retry_at = DateTime::parse_from_rfc2822(value).ok()?.timestamp();
    let seconds = retry_at.saturating_sub(Utc::now().timestamp());
    Some(Duration::from_secs(seconds.max(0) as u64))
}

fn permanent_failure(body: &[u8], has_retry_after: bool) -> Option<&'static str> {
    let text = String::from_utf8_lossy(body).to_ascii_lowercase();
    let retry_hint = has_retry_after
        || [
            "retryinfo",
            "retry_after",
            "retry-after",
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
        .any(|value| text.contains(value));

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
        "daily quota",
        "monthly quota",
        "per-day quota",
        "per-month quota",
    ]
    .iter()
    .any(|value| text.contains(value));
    if hard_exhausted_quota {
        return Some(
            "The provider reports an exhausted quota; check its quota limits or billing settings",
        );
    }

    // These provider error codes are ambiguous on their own: rate/quota
    // windows can recover, while exhausted account quotas cannot.
    let ambiguous_quota = [
        "servicequotaexceeded",
        "quotaexceededexception",
        "quota_exceeded",
        "quota exceeded",
        "quota has been exceeded",
    ]
    .iter()
    .any(|value| text.contains(value));

    if ambiguous_quota && !retry_hint {
        Some("The provider reports an exhausted quota; check its quota limits or billing settings")
    } else {
        None
    }
}

fn response_error_message(status: StatusCode, body: &[u8]) -> String {
    let data: serde_json::Value = match serde_json::from_slice(body) {
        Ok(data) => data,
        Err(_) => return format!("Provider returned HTTP {}", status.as_u16()),
    };

    let message = data
        .pointer("/error/message")
        .and_then(serde_json::Value::as_str)
        .or_else(|| data.pointer("/message").and_then(serde_json::Value::as_str))
        .or_else(|| data.pointer("/detail").and_then(serde_json::Value::as_str))
        .or_else(|| data.pointer("/error").and_then(serde_json::Value::as_str))
        .unwrap_or("provider request failed");
    let code = data
        .pointer("/error/type")
        .and_then(serde_json::Value::as_str)
        .or_else(|| {
            data.pointer("/error/code")
                .and_then(serde_json::Value::as_str)
        })
        .or_else(|| data.pointer("/code").and_then(serde_json::Value::as_str));
    let message: String = message
        .chars()
        .filter(|value| !value.is_control())
        .take(300)
        .collect();

    match code {
        Some(code) => format!(
            "Provider returned HTTP {}: {} (code: {})",
            status.as_u16(),
            message,
            code
        ),
        None => format!("Provider returned HTTP {}: {}", status.as_u16(), message),
    }
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
        for status in [500, 502, 503, 504, 529] {
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
        assert_eq!(parse_retry_after(&headers), Some(Duration::from_secs(2)));
        let date = (Utc::now() + chrono::Duration::seconds(2))
            .format("%a, %d %b %Y %H:%M:%S GMT")
            .to_string();
        headers.insert(RETRY_AFTER, date.parse().unwrap());
        assert!(parse_retry_after(&headers).unwrap() >= Duration::from_secs(1));
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
            r#"{"error":{"code":"ServiceQuotaExceededException","message":"Daily quota exhausted"}}"#,
            r#"{"error":{"type":"insufficient_quota","message":"You exceeded your current quota"}}"#,
            r#"{"error":{"type":"billing_hard_limit_reached","message":"Billing hard limit reached"}}"#,
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
