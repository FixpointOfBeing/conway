//! Line-oriented REPL: reads a prompt from stdin and streams the model's reply.

use futures_util::StreamExt;
use std::io::{self, IsTerminal, Write};
use tokio::io::AsyncBufReadExt;

use crate::llm::{Input, InputItem, LlmClient, Response, ResponseRequest, ResponseStatus};
use crate::prompt;
use crate::provider_config::read_config;

const BANNER: &str = r#"
  ___ ___  _ ____      ____ _ _   _
 / __/ _ \| '_ \ \ /\ / / _` | | | |
| (_| (_) | | | \ V  V / (_| | |_| |
 \___\___/|_| |_|\_/\_/ \__,_|\__, |
                              |___/
"#;

/// Loads `~/.conway/config.toml` and starts the REPL.
pub async fn run() -> Result<(), Box<dyn std::error::Error>> {
    let config = read_config()?;
    let client = LlmClient::from_config(&config)?;
    run_with_client(client).await
}

/// Starts the REPL with an already-constructed client.
pub async fn run_with_client(client: LlmClient) -> Result<(), Box<dyn std::error::Error>> {
    // ANSI styling only when attached to a terminal, so piped output stays clean.
    let color = io::stdout().is_terminal();

    let cwd = std::env::current_dir()?;
    let system_prompt = prompt::build_system_prompt(&cwd);

    let (accent, reset) = if color { ("\x1b[36m", "\x1b[0m") } else { ("", "") };
    println!("{accent}{BANNER}{reset}");
    println!(
        "  conway v{} · {} @ {}\n",
        env!("CARGO_PKG_VERSION"),
        client.model(),
        client.reasoning_effort()
    );
    println!("输入消息开始对话。/clear 清空上下文，/exit 退出。\n");

    let mut history: Vec<InputItem> = Vec::new();
    let mut lines = tokio::io::BufReader::new(tokio::io::stdin()).lines();

    loop {
        print!("› ");
        io::stdout().flush()?;

        let Some(line) = lines.next_line().await? else {
            println!();
            break;
        };
        let input = line.trim();

        match input.trim() {
            "" => continue,
            "/exit" | "/quit" => break,
            "/clear" => {
                history.clear();
                println!("上下文已清空。\n");
                continue;
            },
            _ => {},
        }

        // The Responses API is stateless: send the whole conversation each turn.
        history.push(InputItem::user(input));
        let request = ResponseRequest::new(client.model())
            .instructions(system_prompt.clone())
            .input(Input::Items(history.clone()));

        let mut stream = match client.create_stream(request).await {
            Ok(stream) => stream,
            Err(err) => {
                eprintln!("请求失败: {err}\n");
                continue;
            },
        };

        let mut reasoning_started = false;
        let mut text_started = false;
        // Fallback in case the terminal event does not carry the full response.
        let mut pending_items: Vec<InputItem> = Vec::new();

        while let Some(event) = stream.next().await {
            let event = match event {
                Ok(event) => event,
                Err(err) => {
                    eprintln!("\n流中断: {err}");
                    break;
                },
            };

            if event.is_reasoning_delta()
                && let Some(delta) = event.delta_text()
            {
                if !reasoning_started {
                    if color {
                        print!("\x1b[2m");
                    } else {
                        print!("[reasoning] ");
                    }
                }
                reasoning_started = true;
                print!("{delta}");
                io::stdout().flush()?;
            } else if event.is_output_text_delta()
                && let Some(delta) = event.delta_text()
            {
                if reasoning_started && !text_started {
                    if color {
                        println!("\x1b[0m");
                    } else {
                        println!();
                    }
                }
                text_started = true;
                print!("{delta}");
                io::stdout().flush()?;
            }

            if event.kind == "response.output_item.done"
                && let Some(item) = &event.item
            {
                pending_items.push(InputItem::from(item));
            }

            if event.is_terminal() {
                if (reasoning_started || text_started) && color {
                    print!("\x1b[0m");
                }
                println!();

                record_turn(event.response.as_ref(), &mut pending_items, &mut history);
                print_usage(event.response.as_ref());
                break;
            }
        }

        println!();
    }

    Ok(())
}

/// Appends the assistant turn to the history so the next request has context.
fn record_turn(
    response: Option<&Response>,
    pending_items: &mut Vec<InputItem>,
    history: &mut Vec<InputItem>,
) {
    if let Some(response) = response {
        if response.status != ResponseStatus::Completed {
            eprintln!("响应未正常完成: {:?}", response.status);
            if let Some(error) = &response.error {
                eprintln!("错误: {:?} {:?}", error.code, error.message);
            }
        }

        if !response.output.is_empty() {
            history.extend(response.output.iter().map(InputItem::from));
            return;
        }
    }

    history.append(pending_items);
}

fn print_usage(response: Option<&Response>) {
    if let Some(usage) = response.and_then(|response| response.usage.as_ref()) {
        println!(
            "[tokens: in {} · out {} · total {}]",
            usage.input_tokens, usage.output_tokens, usage.total_tokens
        );
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn response(json: &str) -> Response {
        serde_json::from_str(json).expect("test response parses")
    }

    #[test]
    fn record_turn_appends_the_response_output() {
        let response = response(
            r#"{
                "status": "completed",
                "output": [{
                    "type": "message",
                    "role": "assistant",
                    "content": [{"type": "output_text", "text": "hello"}]
                }]
            }"#,
        );
        let mut pending = vec![InputItem::user("fallback")];
        let mut history = vec![InputItem::user("hi")];

        record_turn(Some(&response), &mut pending, &mut history);

        assert_eq!(history.len(), 2);
        assert_eq!(history[1].0["type"], "message");
        assert_eq!(history[1].0["content"][0]["text"], "hello");
        assert_eq!(pending.len(), 1, "pending items are unused when output exists");
    }

    #[test]
    fn record_turn_falls_back_to_pending_items() {
        let mut pending = vec![InputItem::user("from events")];
        let mut history = vec![InputItem::user("hi")];

        let empty = response(r#"{"status":"completed","output":[]}"#);
        record_turn(Some(&empty), &mut pending, &mut history);

        assert_eq!(history.len(), 2);
        assert_eq!(history[1].0["content"][0]["text"], "from events");
    }
}
