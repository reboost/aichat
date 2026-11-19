use super::retry::{
    format_response_error, read_response_body, response_body_text, RetryPolicy, RetryState,
};
use super::ToolCall;
use crate::utils::AbortSignal;

use anyhow::{anyhow, bail, Context, Result};
use eventsource_stream::Eventsource;
use futures_util::{Stream, StreamExt};
use reqwest::RequestBuilder;
use reqwest_eventsource::{retry::Never, Error as EventSourceError, Event, RequestBuilderExt};
use tokio::sync::mpsc::UnboundedSender;

pub struct SseHandler {
    sender: UnboundedSender<SseEvent>,
    abort_signal: AbortSignal,
    buffer: String,
    tool_calls: Vec<ToolCall>,
}

impl SseHandler {
    pub fn new(sender: UnboundedSender<SseEvent>, abort_signal: AbortSignal) -> Self {
        Self {
            sender,
            abort_signal,
            buffer: String::new(),
            tool_calls: Vec::new(),
        }
    }

    pub fn text(&mut self, text: &str) -> Result<()> {
        // debug!("HandleText: {}", text);
        if text.is_empty() {
            return Ok(());
        }
        self.buffer.push_str(text);
        let ret = self
            .sender
            .send(SseEvent::Text(text.to_string()))
            .with_context(|| "Failed to send SseEvent:Text");
        if let Err(err) = ret {
            if self.abort_signal.aborted() {
                return Ok(());
            }
            return Err(err);
        }
        Ok(())
    }

    pub fn done(&mut self) {
        // debug!("HandleDone");
        let ret = self.sender.send(SseEvent::Done);
        if ret.is_err() {
            if self.abort_signal.aborted() {
                return;
            }
            warn!("Failed to send SseEvent:Done");
        }
    }

    pub fn tool_call(&mut self, call: ToolCall) -> Result<()> {
        // debug!("HandleCall: {:?}", call);
        self.tool_calls.push(call);
        Ok(())
    }

    pub fn abort(&self) -> AbortSignal {
        self.abort_signal.clone()
    }

    pub fn tool_calls(&self) -> &[ToolCall] {
        &self.tool_calls
    }

    pub fn take(self) -> (String, Vec<ToolCall>) {
        let Self {
            buffer, tool_calls, ..
        } = self;
        (buffer, tool_calls)
    }
}

#[derive(Debug)]
pub enum SseEvent {
    Text(String),
    Done,
}

#[derive(Debug)]
pub struct SseMmessage {
    #[allow(unused)]
    pub event: String,
    pub data: String,
}

pub async fn sse_stream<F>(builder: RequestBuilder, handle: F) -> Result<()>
where
    F: FnMut(SseMmessage) -> Result<bool>,
{
    sse_stream_with_policy(builder, handle, RetryPolicy::from_env()).await
}

pub(super) async fn sse_stream_with_policy<F>(
    builder: RequestBuilder,
    mut handle: F,
    policy: RetryPolicy,
) -> Result<()>
where
    F: FnMut(SseMmessage) -> Result<bool>,
{
    if builder.try_clone().is_none() {
        return sse_stream_once(builder, handle).await;
    }

    let mut retry_state = RetryState::new(policy);

    loop {
        let attempt_builder = builder
            .try_clone()
            .ok_or_else(|| anyhow!("Streaming request cannot be safely cloned for retries"))?;
        let mut es = attempt_builder
            .eventsource()
            .map_err(|error| {
                retry_state.with_last_diagnostic(anyhow!(
                    "Streaming request could not be prepared for sending; no provider HTTP response was received: {error}"
                ))
            })?;
        // Disable reqwest-eventsource's automatic reconnects; all retries must pass
        // through the shared policy, and no request may be replayed after Open.
        es.set_retry_policy(Box::new(Never));
        let mut opened = false;
        let mut retry_delay = None;

        while let Some(event) = es.next().await {
            match event {
                Ok(Event::Open) => opened = true,
                Ok(Event::Message(message)) => {
                    let message = SseMmessage {
                        event: message.event,
                        data: message.data,
                    };
                    if handle(message)? {
                        es.close();
                        return Ok(());
                    }
                }
                Err(EventSourceError::StreamEnded) => {
                    es.close();
                    break;
                }
                Err(EventSourceError::InvalidStatusCode(status, response)) => {
                    let headers = response.headers().clone();
                    let body = read_response_body(response)
                        .await
                        .map_err(|error| retry_state.with_last_diagnostic(error))?;
                    retry_delay = Some(retry_state.retry_response(status, &headers, &body, true)?);
                    es.close();
                    break;
                }
                Err(EventSourceError::Transport(err)) if !opened && err.is_connect() => {
                    retry_delay = Some(retry_state.retry_connection(err)?);
                    es.close();
                    break;
                }
                Err(EventSourceError::InvalidContentType(header_value, response)) => {
                    let status = response.status();
                    let body = read_response_body(response).await?;
                    bail!(
                        "Invalid response event-stream. HTTP {} content-type: {}, data: {}",
                        status.as_u16(),
                        header_value.to_str().unwrap_or_default(),
                        response_body_text(&body)
                    );
                }
                Err(EventSourceError::Transport(err)) => {
                    let context = if opened {
                        "Streaming transport failed after the provider HTTP response opened; the stream was not replayed"
                    } else if err.is_timeout() {
                        "Streaming request timed out before a provider HTTP response was received; it was not replayed because the provider may have received it"
                    } else {
                        "Streaming transport failed before a provider HTTP response was received; the request was not replayed"
                    };
                    return Err(retry_state.transport_error(context, err));
                }
                Err(err) => return Err(retry_state.with_last_diagnostic(err.into())),
            }
        }

        match retry_delay {
            Some(delay) => tokio::time::sleep(delay).await,
            None => return Ok(()),
        }
    }
}

async fn sse_stream_once<F>(builder: RequestBuilder, mut handle: F) -> Result<()>
where
    F: FnMut(SseMmessage) -> Result<bool>,
{
    let response = builder
        .header(reqwest::header::ACCEPT, "text/event-stream")
        .send()
        .await
        .map_err(|err| {
            anyhow!(
                "Streaming transport failure before a provider HTTP response was received: {}",
                err.without_url()
            )
        })?;
    let status = response.status();
    if status != reqwest::StatusCode::OK {
        let body = read_response_body(response).await?;
        bail!("{}", format_response_error(status, &body));
    }

    let content_type = response
        .headers()
        .get(reqwest::header::CONTENT_TYPE)
        .and_then(|value| value.to_str().ok())
        .unwrap_or_default()
        .to_string();
    if !content_type
        .split(';')
        .next()
        .is_some_and(|value| value.trim().eq_ignore_ascii_case("text/event-stream"))
    {
        let status = response.status();
        let body = read_response_body(response).await?;
        bail!(
            "Invalid response event-stream. HTTP {} content-type: {content_type}, data: {}",
            status.as_u16(),
            response_body_text(&body)
        );
    }

    let mut events = response.bytes_stream().eventsource();
    while let Some(event) = events.next().await {
        let event = match event {
            Ok(event) => event,
            Err(eventsource_stream::EventStreamError::Transport(err)) if err.is_timeout() => {
                bail!(
                    "Provider stream timed out after the HTTP response opened; the stream was not replayed: {}",
                    err.without_url()
                );
            }
            Err(eventsource_stream::EventStreamError::Transport(err)) => {
                bail!(
                    "Provider stream transport failed after the HTTP response opened; the stream was not replayed: {}",
                    err.without_url()
                );
            }
            Err(err) => bail!("{err}"),
        };
        if handle(SseMmessage {
            event: event.event,
            data: event.data,
        })? {
            return Ok(());
        }
    }

    Ok(())
}

pub async fn json_stream<S, F, E>(mut stream: S, mut handle: F) -> Result<()>
where
    S: Stream<Item = Result<bytes::Bytes, E>> + Unpin,
    F: FnMut(&str) -> Result<()>,
    E: std::error::Error,
{
    let mut parser = JsonStreamParser::default();
    let mut unparsed_bytes = vec![];
    while let Some(chunk_bytes) = stream.next().await {
        let chunk_bytes =
            chunk_bytes.map_err(|err| anyhow!("Failed to read json stream, {err}"))?;
        unparsed_bytes.extend(chunk_bytes);
        match std::str::from_utf8(&unparsed_bytes) {
            Ok(text) => {
                parser.process(text, &mut handle)?;
                unparsed_bytes.clear();
            }
            Err(_) => {
                continue;
            }
        }
    }
    if !unparsed_bytes.is_empty() {
        let text = std::str::from_utf8(&unparsed_bytes)?;
        parser.process(text, &mut handle)?;
    }

    Ok(())
}

#[derive(Debug, Default)]
struct JsonStreamParser {
    buffer: Vec<char>,
    cursor: usize,
    start: Option<usize>,
    balances: Vec<char>,
    quoting: bool,
    escape: bool,
}

impl JsonStreamParser {
    fn process<F>(&mut self, text: &str, handle: &mut F) -> Result<()>
    where
        F: FnMut(&str) -> Result<()>,
    {
        self.buffer.extend(text.chars());

        for i in self.cursor..self.buffer.len() {
            let ch = self.buffer[i];
            if self.quoting {
                if ch == '\\' {
                    self.escape = !self.escape;
                } else {
                    if !self.escape && ch == '"' {
                        self.quoting = false;
                    }
                    self.escape = false;
                }
                continue;
            }
            match ch {
                '"' => {
                    self.quoting = true;
                    self.escape = false;
                }
                '{' => {
                    if self.balances.is_empty() {
                        self.start = Some(i);
                    }
                    self.balances.push(ch);
                }
                '[' => {
                    if self.start.is_some() {
                        self.balances.push(ch);
                    }
                }
                '}' => {
                    self.balances.pop();
                    if self.balances.is_empty() {
                        if let Some(start) = self.start.take() {
                            let value: String = self.buffer[start..=i].iter().collect();
                            handle(&value)?;
                        }
                    }
                }
                ']' => {
                    self.balances.pop();
                }
                _ => {}
            }
        }
        self.cursor = self.buffer.len();
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    use bytes::Bytes;
    use futures_util::stream;
    use rand::Rng;

    fn split_chunks(text: &str) -> Vec<Vec<u8>> {
        let mut rng = rand::rng();
        let len = text.len();
        let cut1 = rng.random_range(1..len - 1);
        let cut2 = rng.random_range(cut1 + 1..len);
        let chunk1 = text.as_bytes()[..cut1].to_vec();
        let chunk2 = text.as_bytes()[cut1..cut2].to_vec();
        let chunk3 = text.as_bytes()[cut2..].to_vec();
        vec![chunk1, chunk2, chunk3]
    }

    macro_rules! assert_json_stream {
        ($input:expr, $output:expr) => {
            let chunks: Vec<_> = split_chunks($input)
                .into_iter()
                .map(|chunk| Ok::<_, std::convert::Infallible>(Bytes::from(chunk)))
                .collect();
            let stream = stream::iter(chunks);
            let mut output = vec![];
            let ret = json_stream(stream, |data| {
                output.push(data.to_string());
                Ok(())
            })
            .await;
            assert!(ret.is_ok());
            assert_eq!($output.replace("\r\n", "\n"), output.join("\n"))
        };
    }

    #[tokio::test]
    async fn test_json_stream_ndjson() {
        let data = r#"{"key": "value"}
{"key": "value2"}
{"key": "value3"}"#;
        assert_json_stream!(data, data);
    }

    #[tokio::test]
    async fn test_json_stream_array() {
        let input = r#"[
{"key": "value"},
{"key": "value2"},
{"key": "value3"},"#;
        let output = r#"{"key": "value"}
{"key": "value2"}
{"key": "value3"}"#;
        assert_json_stream!(input, output);
    }
}
