//! Minimal hand-rolled client for the DeepSeek Responses API.
//!
//! The API is stateless: every request must resend the full conversation as
//! `input` items. Items returned by the model should be echoed back verbatim,
//! which is why [`ResponseItem`] keeps the raw JSON object around.

use crate::provider_config::{Config, ConfigError, ReasoningEffort, WireApi};
use eventsource_stream::Eventsource;
use futures_util::{Stream, StreamExt};
use serde::{Deserialize, Serialize};
use serde_json::{Map, Value, json};
use std::pin::Pin;
use std::time::Duration;

const DEFAULT_TIMEOUT: Duration = Duration::from_secs(300);

/// Errors surfaced by the client.
#[derive(Debug, thiserror::Error)]
pub enum LlmError {
    #[error(transparent)]
    Config(#[from] ConfigError),
    #[error("provider `{0}` does not define `env_key`")]
    MissingEnvKey(String),
    #[error("environment variable `{0}` is not set")]
    MissingApiKey(String),
    #[error("wire_api {0:?} is not supported yet; conway currently speaks the Responses API")]
    UnsupportedWireApi(WireApi),
    #[error("http request failed: {0}")]
    Http(#[from] reqwest::Error),
    #[error("api returned {status}: {body}")]
    Api { status: u16, body: String },
    #[error("failed to decode response body: {0}")]
    Decode(#[from] serde_json::Error),
    #[error("event stream failed: {0}")]
    Stream(String),
}

/// A boxed SSE stream of [`StreamEvent`]s.
pub type EventStream = Pin<Box<dyn Stream<Item = Result<StreamEvent, LlmError>> + Send>>;

pub struct LlmClient {
    http: reqwest::Client,
    endpoint: String,
    api_key: String,
    model: String,
    reasoning: Option<ReasoningEffort>,
}

impl std::fmt::Debug for LlmClient {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("LlmClient")
            .field("endpoint", &self.endpoint)
            .field("model", &self.model)
            .field("reasoning", &self.reasoning)
            .field("api_key", &"<redacted>")
            .finish_non_exhaustive()
    }
}

impl LlmClient {
    /// Builds a client from explicit values. `base_url` is the provider's API
    /// root (for example `https://api.deepseek.com/v1`); `/responses` is
    /// appended if it is not already there.
    pub fn new(
        base_url: &str,
        api_key: impl Into<String>,
        model: impl Into<String>,
        reasoning: Option<ReasoningEffort>,
    ) -> Result<Self, LlmError> {
        let http = reqwest::Client::builder().timeout(DEFAULT_TIMEOUT).build()?;
        Ok(Self {
            http,
            endpoint: responses_endpoint(base_url),
            api_key: api_key.into(),
            model: model.into(),
            reasoning,
        })
    }

    /// Builds a client from the parsed `config.toml`, reading the API key from
    /// the provider's `env_key`.
    pub fn from_config(config: &Config) -> Result<Self, LlmError> {
        let provider = config.provider()?;
        if provider.wire_api != WireApi::Responses {
            return Err(LlmError::UnsupportedWireApi(provider.wire_api));
        }

        let env_key = provider
            .env_key
            .as_deref()
            .ok_or_else(|| LlmError::MissingEnvKey(provider.name.clone()))?;
        let api_key =
            std::env::var(env_key).map_err(|_| LlmError::MissingApiKey(env_key.to_string()))?;
        if api_key.trim().is_empty() {
            return Err(LlmError::MissingApiKey(env_key.to_string()));
        }

        Self::new(&provider.base_url, api_key, config.model.clone(), config.model_reasoning_effort)
    }

    pub fn model(&self) -> String {
        self.model.clone()
    }

    pub fn endpoint(&self) -> String {
        self.endpoint.clone()
    }

    pub fn reasoning_effort(&self) -> String {
        match &self.reasoning {
            Some(effort) => effort.to_string(),
            None => "none".to_string(),
        }
    }

    /// Sends a request and waits for the complete response.
    pub async fn create(&self, request: ResponseRequest) -> Result<Response, LlmError> {
        let request = self.prepare(request, false);
        let http = self.send(&request).await?;
        let body = http.text().await?;
        Ok(serde_json::from_str(&body)?)
    }

    /// Sends a request with `stream = true` and returns the SSE event stream.
    pub async fn create_stream(&self, request: ResponseRequest) -> Result<EventStream, LlmError> {
        let request = self.prepare(request, true);
        let http = self.send(&request).await?;
        let events = http.bytes_stream().eventsource().map(|event| {
            let event = event.map_err(|err| LlmError::Stream(err.to_string()))?;
            serde_json::from_str::<StreamEvent>(&event.data).map_err(LlmError::from)
        });
        Ok(Box::pin(events))
    }

    /// Convenience wrapper for a single string prompt.
    pub async fn ask(&self, prompt: impl Into<String>) -> Result<Response, LlmError> {
        self.create(ResponseRequest::new(self.model.clone()).prompt(prompt))
            .await
    }

    /// Convenience wrapper for a single string prompt with streaming enabled.
    pub async fn ask_stream(&self, prompt: impl Into<String>) -> Result<EventStream, LlmError> {
        self.create_stream(ResponseRequest::new(self.model.clone()).prompt(prompt))
            .await
    }

    fn prepare(&self, mut request: ResponseRequest, stream: bool) -> ResponseRequest {
        if request.model.is_empty() {
            request.model = self.model.clone();
        }
        if request.reasoning.is_none() {
            request.reasoning = self.reasoning.map(|effort| Reasoning { effort });
        }
        request.stream = stream.then_some(true);
        request
    }

    async fn send(&self, request: &ResponseRequest) -> Result<reqwest::Response, LlmError> {
        let response = self
            .http
            .post(&self.endpoint)
            .bearer_auth(&self.api_key)
            .json(request)
            .send()
            .await?;

        let status = response.status();
        if !status.is_success() {
            let body = response.text().await.unwrap_or_default();
            return Err(LlmError::Api { status: status.as_u16(), body });
        }
        Ok(response)
    }
}

fn responses_endpoint(base_url: &str) -> String {
    let base = base_url.trim_end_matches('/');
    if base.ends_with("/responses") { base.to_string() } else { format!("{base}/responses") }
}

/// Request body accepted by `POST {base_url}/responses`.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct ResponseRequest {
    pub model: String,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub instructions: Option<String>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub input: Option<Input>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub reasoning: Option<Reasoning>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub max_output_tokens: Option<u32>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub temperature: Option<f32>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub stream: Option<bool>,
}

impl ResponseRequest {
    pub fn new(model: impl Into<String>) -> Self {
        Self {
            model: model.into(),
            instructions: None,
            input: None,
            reasoning: None,
            max_output_tokens: None,
            temperature: None,
            stream: None,
        }
    }

    pub fn instructions(mut self, instructions: impl Into<String>) -> Self {
        self.instructions = Some(instructions.into());
        self
    }

    pub fn input(mut self, input: Input) -> Self {
        self.input = Some(input);
        self
    }

    pub fn input_items(mut self, items: impl IntoIterator<Item = InputItem>) -> Self {
        self.input = Some(Input::Items(items.into_iter().collect()));
        self
    }

    /// Sets `input` to a bare string, which the API treats as one user message.
    pub fn prompt(mut self, prompt: impl Into<String>) -> Self {
        self.input = Some(Input::Text(prompt.into()));
        self
    }

    pub fn reasoning(mut self, effort: ReasoningEffort) -> Self {
        self.reasoning = Some(Reasoning { effort });
        self
    }

    pub fn max_output_tokens(mut self, tokens: u32) -> Self {
        self.max_output_tokens = Some(tokens);
        self
    }

    pub fn temperature(mut self, temperature: f32) -> Self {
        self.temperature = Some(temperature);
        self
    }
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct Reasoning {
    pub effort: ReasoningEffort,
}

/// `input` accepts either a plain string or a list of input items.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(untagged)]
pub enum Input {
    Text(String),
    Items(Vec<InputItem>),
}

/// One entry of the conversation history, kept as raw JSON so unknown item
/// types (`reasoning`, `custom_tool_call`, ...) survive a round trip.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(transparent)]
pub struct InputItem(pub Value);

impl InputItem {
    pub fn message(role: &str, text: impl Into<String>) -> Self {
        Self(json!({
            "type": "message",
            "role": role,
            "content": [{"type": "input_text", "text": text.into()}],
        }))
    }

    pub fn user(text: impl Into<String>) -> Self {
        Self::message("user", text)
    }

    pub fn system(text: impl Into<String>) -> Self {
        Self::message("system", text)
    }

    pub fn developer(text: impl Into<String>) -> Self {
        Self::message("developer", text)
    }

    pub fn assistant(text: impl Into<String>) -> Self {
        Self::message("assistant", text)
    }

    pub fn function_call_output(call_id: impl Into<String>, output: impl Into<String>) -> Self {
        Self(json!({
            "type": "function_call_output",
            "call_id": call_id.into(),
            "output": output.into(),
        }))
    }

    pub fn raw(value: Value) -> Self {
        Self(value)
    }
}

impl From<&ResponseItem> for InputItem {
    fn from(item: &ResponseItem) -> Self {
        Self(Value::Object(item.0.clone()))
    }
}

/// Response object returned by `POST /responses`.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct Response {
    #[serde(default)]
    pub id: String,
    #[serde(default)]
    pub model: String,
    #[serde(default)]
    pub status: ResponseStatus,
    #[serde(default)]
    pub output: Vec<ResponseItem>,
    #[serde(default)]
    pub usage: Option<Usage>,
    #[serde(default)]
    pub error: Option<ApiErrorBody>,
    #[serde(default)]
    pub incomplete_details: Option<IncompleteDetails>,
    #[serde(flatten)]
    pub extra: Map<String, Value>,
}

impl Response {
    /// Concatenated `output_text` blocks of all `message` items.
    pub fn output_text(&self) -> Option<String> {
        let text: String = self.output.iter().filter_map(ResponseItem::output_text).collect();
        (!text.is_empty()).then_some(text)
    }

    /// Concatenated `reasoning_text` blocks of all `reasoning` items.
    pub fn reasoning_text(&self) -> Option<String> {
        let text: String = self.output.iter().filter_map(ResponseItem::reasoning_text).collect();
        (!text.is_empty()).then_some(text)
    }

    pub fn function_calls(&self) -> impl Iterator<Item = &ResponseItem> {
        self.output.iter().filter(|item| item.kind() == Some("function_call"))
    }

    pub fn is_failed(&self) -> bool {
        self.status == ResponseStatus::Failed
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Default, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum ResponseStatus {
    #[default]
    InProgress,
    Completed,
    Incomplete,
    Failed,
    #[serde(other)]
    Unknown,
}

/// One output item. Stored as raw JSON; use the accessors for the common shapes.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(transparent)]
pub struct ResponseItem(pub Map<String, Value>);

impl ResponseItem {
    pub fn kind(&self) -> Option<&str> {
        self.0.get("type").and_then(Value::as_str)
    }

    pub fn output_text(&self) -> Option<String> {
        self.joined_content("output_text")
    }

    pub fn reasoning_text(&self) -> Option<String> {
        self.joined_content("reasoning_text")
    }

    pub fn call_id(&self) -> Option<&str> {
        self.0.get("call_id").and_then(Value::as_str)
    }

    pub fn name(&self) -> Option<&str> {
        self.0.get("name").and_then(Value::as_str)
    }

    pub fn arguments(&self) -> Option<&str> {
        self.0.get("arguments").and_then(Value::as_str)
    }

    pub fn as_value(&self) -> Value {
        Value::Object(self.0.clone())
    }

    fn joined_content(&self, block_kind: &str) -> Option<String> {
        let parts = self.0.get("content")?.as_array()?;
        let text: String = parts
            .iter()
            .filter(|part| part.get("type").and_then(Value::as_str) == Some(block_kind))
            .filter_map(|part| part.get("text").and_then(Value::as_str))
            .collect();
        (!text.is_empty()).then_some(text)
    }
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct Usage {
    #[serde(default)]
    pub input_tokens: u64,
    #[serde(default)]
    pub output_tokens: u64,
    #[serde(default)]
    pub total_tokens: u64,
    #[serde(default)]
    pub input_tokens_details: Option<TokenDetails>,
    #[serde(default)]
    pub output_tokens_details: Option<TokenDetails>,
}

#[derive(Debug, Clone, PartialEq, Default, Serialize, Deserialize)]
pub struct TokenDetails {
    #[serde(default)]
    pub cached_tokens: u64,
    #[serde(default)]
    pub reasoning_tokens: u64,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct ApiErrorBody {
    #[serde(default)]
    pub code: Option<String>,
    #[serde(default)]
    pub message: Option<String>,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct IncompleteDetails {
    #[serde(default)]
    pub reason: Option<String>,
}

/// One server-sent event of a streaming response.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct StreamEvent {
    #[serde(rename = "type")]
    pub kind: String,
    #[serde(default)]
    pub delta: Option<String>,
    #[serde(default)]
    pub item_id: Option<String>,
    #[serde(default)]
    pub output_index: Option<u32>,
    #[serde(default)]
    pub item: Option<ResponseItem>,
    #[serde(default)]
    pub response: Option<Response>,
    #[serde(flatten)]
    pub extra: Map<String, Value>,
}

impl StreamEvent {
    /// `response.completed` / `response.incomplete` / `response.failed`.
    pub fn is_terminal(&self) -> bool {
        matches!(
            self.kind.as_str(),
            "response.completed" | "response.incomplete" | "response.failed"
        )
    }

    /// A chunk of visible assistant text (`response.output_text.delta`).
    pub fn is_output_text_delta(&self) -> bool {
        self.kind == "response.output_text.delta"
    }

    /// A chunk of chain-of-thought text (`response.reasoning_text.delta`).
    pub fn is_reasoning_delta(&self) -> bool {
        self.kind == "response.reasoning_text.delta"
    }

    pub fn delta_text(&self) -> Option<&str> {
        self.delta.as_deref()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::provider_config::ModelProvider;
    use std::collections::BTreeMap;

    fn provider(wire_api: WireApi, env_key: Option<&str>) -> ModelProvider {
        ModelProvider {
            name: "DeepSeek".to_string(),
            base_url: "https://api.deepseek.com/v1".to_string(),
            env_key: env_key.map(str::to_string),
            wire_api,
        }
    }

    fn config(wire_api: WireApi, env_key: Option<&str>) -> Config {
        Config {
            model: "deepseek-flash".to_string(),
            model_provider: "deepseek".to_string(),
            model_reasoning_effort: Some(ReasoningEffort::Medium),
            model_providers: BTreeMap::from([(
                "deepseek".to_string(),
                provider(wire_api, env_key),
            )]),
        }
    }

    #[test]
    fn builds_the_responses_endpoint() {
        assert_eq!(
            responses_endpoint("https://api.deepseek.com/v1"),
            "https://api.deepseek.com/v1/responses"
        );
        assert_eq!(
            responses_endpoint("https://api.deepseek.com/v1/"),
            "https://api.deepseek.com/v1/responses"
        );
        assert_eq!(
            responses_endpoint("https://api.deepseek.com/v1/responses"),
            "https://api.deepseek.com/v1/responses"
        );

        let client =
            LlmClient::new("https://api.deepseek.com/v1/", "sk-test", "deepseek-flash", None)
                .expect("client should build");
        assert_eq!(client.endpoint(), "https://api.deepseek.com/v1/responses");
        assert_eq!(client.model(), "deepseek-flash");
    }

    #[test]
    fn serializes_a_string_prompt_as_input() {
        let request = ResponseRequest::new("deepseek-flash").prompt("hi");
        let value = serde_json::to_value(request).expect("request serializes");

        assert_eq!(value["model"], "deepseek-flash");
        assert_eq!(value["input"], "hi");
    }

    #[test]
    fn serializes_items_reasoning_and_options() {
        let request = ResponseRequest::new("deepseek-flash")
            .instructions("You are helpful.")
            .reasoning(ReasoningEffort::Max)
            .max_output_tokens(1024)
            .temperature(0.2)
            .input_items([InputItem::user("hi")]);

        let value = serde_json::to_value(request).expect("request serializes");

        assert_eq!(value["instructions"], "You are helpful.");
        assert_eq!(value["reasoning"]["effort"], "max");
        assert_eq!(value["max_output_tokens"], 1024);
        assert_eq!(value["input"][0]["type"], "message");
        assert_eq!(value["input"][0]["role"], "user");
        assert_eq!(value["input"][0]["content"][0]["type"], "input_text");
        assert_eq!(value["input"][0]["content"][0]["text"], "hi");
        assert!(value.get("stream").is_none(), "stream is omitted until requested");
    }

    #[test]
    fn serializes_function_call_output_items() {
        let item = InputItem::function_call_output("call_1", "42");
        let value = serde_json::to_value(item).expect("item serializes");

        assert_eq!(value["type"], "function_call_output");
        assert_eq!(value["call_id"], "call_1");
        assert_eq!(value["output"], "42");
    }

    #[test]
    fn from_config_rejects_non_responses_wire_api() {
        let error = LlmClient::from_config(&config(WireApi::Chat, Some("DEEPSEEK_API_KEY")))
            .expect_err("chat wire api is not implemented");

        assert!(matches!(error, LlmError::UnsupportedWireApi(WireApi::Chat)));
    }

    #[test]
    fn from_config_requires_an_env_key() {
        let error = LlmClient::from_config(&config(WireApi::Responses, None))
            .expect_err("env_key is required");

        assert!(matches!(error, LlmError::MissingEnvKey(name) if name == "DeepSeek"));
    }

    #[test]
    fn from_config_reports_a_missing_environment_variable() {
        let error = LlmClient::from_config(&config(
            WireApi::Responses,
            Some("CONWAY_TEST_KEY_THAT_DOES_NOT_EXIST"),
        ))
        .expect_err("the variable is not set");

        assert!(
            matches!(error, LlmError::MissingApiKey(key) if key == "CONWAY_TEST_KEY_THAT_DOES_NOT_EXIST")
        );
    }

    #[test]
    fn deserializes_a_non_streaming_response() {
        let body = r#"
        {
            "id": "resp_1",
            "object": "response",
            "created_at": 1700000000,
            "status": "completed",
            "model": "deepseek-flash",
            "output": [
                {
                    "type": "reasoning",
                    "id": "rs_1",
                    "status": "completed",
                    "content": [{"type": "reasoning_text", "text": "thinking..."}]
                },
                {
                    "type": "message",
                    "id": "msg_1",
                    "status": "completed",
                    "role": "assistant",
                    "content": [{"type": "output_text", "text": "Hello!"}]
                }
            ],
            "usage": {
                "input_tokens": 10,
                "input_tokens_details": {"cached_tokens": 2},
                "output_tokens": 5,
                "output_tokens_details": {"reasoning_tokens": 3},
                "total_tokens": 15
            }
        }
        "#;

        let response: Response = serde_json::from_str(body).expect("response parses");

        assert_eq!(response.status, ResponseStatus::Completed);
        assert_eq!(response.output_text().as_deref(), Some("Hello!"));
        assert_eq!(response.reasoning_text().as_deref(), Some("thinking..."));
        assert_eq!(response.extra["object"], "response");
        let usage = response.usage.expect("usage is present");
        assert_eq!(usage.total_tokens, 15);
        assert_eq!(usage.input_tokens_details.expect("details").cached_tokens, 2);
        assert_eq!(usage.output_tokens_details.expect("details").reasoning_tokens, 3);
    }

    #[test]
    fn deserializes_function_calls() {
        let body = r#"
        {
            "status": "completed",
            "output": [{
                "type": "function_call",
                "call_id": "call_1",
                "name": "read_file",
                "arguments": "{\"path\":\"a.txt\"}"
            }]
        }
        "#;

        let response: Response = serde_json::from_str(body).expect("response parses");
        let call = response.function_calls().next().expect("one call");

        assert_eq!(call.call_id(), Some("call_1"));
        assert_eq!(call.name(), Some("read_file"));
        assert_eq!(call.arguments(), Some("{\"path\":\"a.txt\"}"));
        assert_eq!(response.output_text(), None);
    }

    #[test]
    fn output_items_can_be_fed_back_as_input() {
        let body =
            r#"{"type":"reasoning","id":"rs_1","content":[{"type":"reasoning_text","text":"hm"}]}"#;
        let item: ResponseItem = serde_json::from_str(body).expect("item parses");

        let echoed = InputItem::from(&item);

        assert_eq!(
            serde_json::to_value(&echoed).expect("serializes"),
            serde_json::from_str::<Value>(body).expect("json")
        );
    }

    #[test]
    fn parses_delta_and_terminal_stream_events() {
        let delta: StreamEvent = serde_json::from_str(
            r#"{"type":"response.output_text.delta","item_id":"msg_1","output_index":0,"delta":"Hel"}"#,
        )
        .expect("delta parses");
        assert_eq!(delta.kind, "response.output_text.delta");
        assert_eq!(delta.delta_text(), Some("Hel"));
        assert!(delta.is_output_text_delta());
        assert!(!delta.is_reasoning_delta());
        assert!(!delta.is_terminal());

        let done: StreamEvent = serde_json::from_str(
            r#"{"type":"response.completed","response":{"status":"completed","usage":{"total_tokens":3}}}"#,
        )
        .expect("terminal event parses");
        assert!(done.is_terminal());
        assert_eq!(done.response.expect("response").usage.expect("usage").total_tokens, 3);

        let reasoning: StreamEvent =
            serde_json::from_str(r#"{"type":"response.reasoning_text.delta","delta":"hmm"}"#)
                .expect("reasoning delta parses");
        assert_eq!(reasoning.delta_text(), Some("hmm"));
        assert!(reasoning.is_reasoning_delta());
        assert!(!reasoning.is_output_text_delta());
    }

    /// Hits the real API using `~/.conway/config.toml` and the env var it names.
    /// Run with: `cargo test live_ask -- --ignored --nocapture`
    #[tokio::test]
    #[ignore = "requires network and the API key from ~/.conway/config.toml"]
    async fn live_ask() {
        let config = crate::provider_config::read_config().expect("load real config");
        let client = LlmClient::from_config(&config).expect("build client");

        let response = client
            .ask("Reply with exactly one word: pong")
            .await
            .expect("request succeeds");
        let text = response.output_text().unwrap_or_default();

        eprintln!("live response: {text:?}");
        assert!(!text.is_empty(), "model returned no output text");
    }

    /// Run with: `cargo test live_stream -- --ignored --nocapture`
    #[tokio::test]
    #[ignore = "requires network and the API key from ~/.conway/config.toml"]
    async fn live_stream() {
        let config = crate::provider_config::read_config().expect("load real config");
        let client = LlmClient::from_config(&config).expect("build client");

        let mut stream = client
            .ask_stream("Count from 1 to 3, separated by spaces.")
            .await
            .expect("stream starts");

        let mut text = String::new();
        let mut reasoning = String::new();
        let mut terminal = false;
        while let Some(event) = stream.next().await {
            let event = event.expect("event parses");
            if event.is_output_text_delta()
                && let Some(delta) = event.delta_text()
            {
                text.push_str(delta);
            } else if event.is_reasoning_delta()
                && let Some(delta) = event.delta_text()
            {
                reasoning.push_str(delta);
            }
            if event.is_terminal() {
                terminal = true;
                break;
            }
        }

        eprintln!("live stream text: {text:?}");
        eprintln!("live stream reasoning: {reasoning:?}");
        assert!(terminal, "stream ended without a terminal event");
        assert!(!text.is_empty(), "stream produced no visible text deltas");
    }
}
