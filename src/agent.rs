//! The agent loop: model request -> tool calls -> results -> repeat.
//!
//! A "turn" is the whole run started by one user message. Within a turn the
//! loop may issue several model requests: each assistant response that carries
//! tool calls is executed and the results fed back, until the model replies
//! with no tool calls.

use std::path::Path;

use futures_util::StreamExt;
use serde_json::Value;

use crate::llm::{ChatClient, Input, InputItem, Response, ResponseRequest, ResponseStatus};
use crate::tools::{ToolContext, ToolOutput, ToolRegistry};

/// Hard cap on model requests per turn, to break runaway tool loops.
pub const MAX_TOOL_ROUNDS: usize = 20;

#[derive(Debug, thiserror::Error)]
pub enum AgentError {
    #[error(transparent)]
    Llm(#[from] crate::llm::LlmError),
    #[error("stopped after {0} tool rounds; the model kept issuing tool calls")]
    TooManyToolRounds(usize),
}

/// UI-facing events emitted while a turn runs.
pub enum AgentEvent {
    ReasoningDelta(String),
    TextDelta(String),
    ToolStart { call_id: String, name: String, arguments: String },
    ToolEnd { call_id: String, name: String, output: String, is_error: bool },
    TurnEnd { response: Response },
}

struct FunctionCall {
    call_id: String,
    name: String,
    arguments: String,
}

/// Run one turn, appending every message (including tool calls and results) to
/// `history` and streaming UI events to `on_event`.
pub async fn run_turn<C: ChatClient>(
    client: &C,
    registry: &ToolRegistry,
    system_prompt: &str,
    cwd: &Path,
    history: &mut Vec<InputItem>,
    mut on_event: impl FnMut(AgentEvent),
) -> Result<(), AgentError> {
    let tools = registry.definitions();
    let ctx = ToolContext::new(cwd);

    for _round in 0..MAX_TOOL_ROUNDS {
        let request = ResponseRequest::new(client.model_name())
            .instructions(system_prompt)
            .input(Input::Items(history.clone()))
            .tools(tools.clone());

        let mut stream = client.stream(request).await?;

        let mut stream_items: Vec<InputItem> = Vec::new();
        let mut response: Option<Response> = None;

        while let Some(event) = stream.next().await {
            let event = event?;
            if event.is_reasoning_delta()
                && let Some(delta) = event.delta_text()
            {
                on_event(AgentEvent::ReasoningDelta(delta.to_string()));
            } else if event.is_output_text_delta()
                && let Some(delta) = event.delta_text()
            {
                on_event(AgentEvent::TextDelta(delta.to_string()));
            }

            if event.kind == "response.output_item.done"
                && let Some(item) = &event.item
            {
                stream_items.push(InputItem::from(item));
            }

            if event.is_terminal() {
                response = event.response;
                break;
            }
        }

        // The terminal event's `response.output` is authoritative; fall back to
        // the per-item events when the stream closed without a full response.
        let items: Vec<InputItem> = match &response {
            Some(response) if !response.output.is_empty() => {
                response.output.iter().map(InputItem::from).collect()
            },
            _ => stream_items,
        };

        let status = response
            .as_ref()
            .map(|response| response.status)
            .unwrap_or(ResponseStatus::Completed);

        // Record the assistant turn before its tool results so the transcript
        // keeps protocol order (function_call items precede their outputs).
        history.extend(items.iter().cloned());

        let calls: Vec<FunctionCall> = items
            .iter()
            .filter(|item| item.kind() == Some("function_call"))
            .map(|item| FunctionCall {
                call_id: item.call_id().unwrap_or_default().to_string(),
                name: item.name().unwrap_or_default().to_string(),
                arguments: item.arguments().unwrap_or("{}").to_string(),
            })
            .collect();

        if let Some(response) = &response {
            on_event(AgentEvent::TurnEnd { response: response.clone() });
        }

        if calls.is_empty() {
            return Ok(());
        }

        if status != ResponseStatus::Completed {
            // Truncated or failed output: arguments may be incomplete, so never
            // execute them. Report each as an error the model can re-issue.
            for call in &calls {
                let output = ToolOutput::err(format!(
                    "Tool call `{}` was not executed: the response ended with status {:?}, so its arguments may be incomplete. Re-issue the call.",
                    call.name, status
                ));
                on_event(AgentEvent::ToolEnd {
                    call_id: call.call_id.clone(),
                    name: call.name.clone(),
                    output: output.text.clone(),
                    is_error: true,
                });
                history.push(InputItem::function_call_output(&call.call_id, &output.text));
            }
            continue;
        }

        for call in &calls {
            on_event(AgentEvent::ToolStart {
                call_id: call.call_id.clone(),
                name: call.name.clone(),
                arguments: call.arguments.clone(),
            });
            let output = execute_call(registry, &ctx, &call.name, &call.arguments).await;
            on_event(AgentEvent::ToolEnd {
                call_id: call.call_id.clone(),
                name: call.name.clone(),
                output: output.text.clone(),
                is_error: output.is_error,
            });
            history.push(InputItem::function_call_output(&call.call_id, &output.text));
        }
    }

    Err(AgentError::TooManyToolRounds(MAX_TOOL_ROUNDS))
}

async fn execute_call(
    registry: &ToolRegistry,
    ctx: &ToolContext,
    name: &str,
    arguments: &str,
) -> ToolOutput {
    let Some(tool) = registry.get(name) else {
        return ToolOutput::err(format!("Tool `{name}` not found"));
    };

    let args: Value = match serde_json::from_str(arguments) {
        Ok(args) => args,
        Err(error) => return ToolOutput::err(format!("Invalid arguments for `{name}`: {error}")),
    };

    tool.execute(ctx, args).await
}

#[cfg(test)]
mod tests {
    use std::collections::VecDeque;
    use std::sync::Mutex;

    use super::*;
    use crate::llm::{EventStream, LlmError, StreamEvent};

    /// A client that pops canned event sequences, one per model request.
    struct FakeClient {
        responses: Mutex<VecDeque<Vec<Result<StreamEvent, LlmError>>>>,
    }

    impl ChatClient for FakeClient {
        fn model_name(&self) -> &str {
            "fake"
        }

        async fn stream(&self, _request: ResponseRequest) -> Result<EventStream, LlmError> {
            let events = self.responses.lock().unwrap().pop_front().unwrap_or_default();
            Ok(Box::pin(futures_util::stream::iter(events)))
        }
    }

    fn sse(json: &str) -> Result<StreamEvent, LlmError> {
        Ok(serde_json::from_str(json).expect("test event parses"))
    }

    #[tokio::test]
    async fn executes_tool_calls_then_returns_final_text() {
        let dir = tempfile::tempdir().expect("tempdir");
        std::fs::write(dir.path().join("hello.txt"), "hello from conway\n").expect("write fixture");

        let registry = ToolRegistry::coding();

        let fake = FakeClient {
            responses: Mutex::new(VecDeque::from([
                vec![sse(
                    r#"{"type":"response.completed","response":{"status":"completed","output":[{"type":"function_call","call_id":"call_1","name":"read","arguments":"{\"path\":\"hello.txt\"}"}]}}"#,
                )],
                vec![sse(
                    r#"{"type":"response.completed","response":{"status":"completed","output":[{"type":"message","role":"assistant","content":[{"type":"output_text","text":"done"}]}]}}"#,
                )],
            ])),
        };

        let mut history = vec![InputItem::user("read hello.txt")];
        let mut events: Vec<AgentEvent> = Vec::new();

        run_turn(&fake, &registry, "test prompt", dir.path(), &mut history, |event| {
            events.push(event)
        })
        .await
        .expect("turn succeeds");

        assert_eq!(history.len(), 4, "user, function_call, function_call_output, message");
        assert_eq!(history[1].kind(), Some("function_call"));
        assert_eq!(history[2].kind(), Some("function_call_output"));
        assert_eq!(history[2].call_id(), Some("call_1"));
        assert!(
            history[2].0["output"]
                .as_str()
                .expect("output is text")
                .contains("hello from conway"),
            "tool output should carry the file contents"
        );

        assert!(
            events
                .iter()
                .any(|event| matches!(event, AgentEvent::ToolStart { name, .. } if name == "read"))
        );
        assert!(
            events
                .iter()
                .any(|event| matches!(event, AgentEvent::ToolEnd { is_error: false, .. }))
        );
    }

    #[tokio::test]
    async fn reports_unknown_tools_as_errors_without_stopping() {
        let dir = tempfile::tempdir().expect("tempdir");
        let registry = ToolRegistry::coding();

        let fake = FakeClient {
            responses: Mutex::new(VecDeque::from([
                vec![sse(
                    r#"{"type":"response.completed","response":{"status":"completed","output":[{"type":"function_call","call_id":"call_1","name":"nope","arguments":"{}"}]}}"#,
                )],
                vec![sse(
                    r#"{"type":"response.completed","response":{"status":"completed","output":[{"type":"message","role":"assistant","content":[{"type":"output_text","text":"recovered"}]}]}}"#,
                )],
            ])),
        };

        let mut history = vec![InputItem::user("do the thing")];
        let mut events: Vec<AgentEvent> = Vec::new();

        run_turn(&fake, &registry, "test prompt", dir.path(), &mut history, |event| {
            events.push(event)
        })
        .await
        .expect("turn succeeds");

        assert_eq!(history[2].kind(), Some("function_call_output"));
        assert!(
            history[2].0["output"]
                .as_str()
                .expect("output is text")
                .contains("not found")
        );
        assert!(
            events
                .iter()
                .any(|event| matches!(event, AgentEvent::ToolEnd { is_error: true, .. }))
        );
    }
}
