//! Line-oriented REPL: reads a prompt from stdin and runs the agent loop,
//! streaming the model's reply and any tool calls.

use std::io::{self, IsTerminal, Write};
use tokio::io::AsyncBufReadExt;

use crate::agent::{self, AgentEvent};
use crate::llm::{InputItem, LlmClient, ResponseStatus};
use crate::prompt;
use crate::provider_config::read_config;
use crate::tools::ToolRegistry;

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
    let registry = ToolRegistry::coding();
    let system_prompt = prompt::build_system_prompt(&cwd, &registry.specs());

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

        // The Responses API is stateless: the agent loop resends the whole
        // conversation each model request.
        history.push(InputItem::user(input));

        let mut renderer = Renderer::new(color);
        let result =
            agent::run_turn(&client, &registry, &system_prompt, &cwd, &mut history, |event| {
                renderer.event(event)
            })
            .await;
        renderer.finish();

        if let Err(error) = result {
            eprintln!("请求失败: {error}\n");
        }

        println!();
    }

    Ok(())
}

/// Renders agent events to the terminal, keeping the reasoning/text stream
/// transitions clean around tool calls.
struct Renderer {
    color: bool,
    reasoning: bool,
    text: bool,
}

impl Renderer {
    fn new(color: bool) -> Self {
        Self { color, reasoning: false, text: false }
    }

    fn event(&mut self, event: AgentEvent) {
        match event {
            AgentEvent::ReasoningDelta(delta) => {
                if !self.reasoning {
                    self.reasoning = true;
                    if self.color {
                        print!("\x1b[2m");
                    } else {
                        print!("[reasoning] ");
                    }
                }
                print!("{delta}");
            },
            AgentEvent::TextDelta(delta) => {
                if self.reasoning && !self.text {
                    self.end_stream();
                }
                self.text = true;
                print!("{delta}");
            },
            AgentEvent::ToolStart { name, arguments, .. } => {
                self.end_stream();
                let args = truncate_chars(&arguments, 160);
                if self.color {
                    println!("\x1b[36m● {name} {args}\x1b[0m");
                } else {
                    println!("● {name} {args}");
                }
            },
            AgentEvent::ToolEnd { name: _, output, is_error, .. } => {
                let marker = if is_error { "✗" } else { "⎿" };
                let preview = preview_output(&output);
                if is_error && self.color {
                    println!("{marker} \x1b[31m{preview}\x1b[0m");
                } else {
                    println!("{marker} {preview}");
                }
            },
            AgentEvent::TurnEnd { response } => {
                self.end_stream();
                if response.status != ResponseStatus::Completed {
                    match &response.error {
                        Some(error) => eprintln!(
                            "响应未正常完成: {:?} ({:?})",
                            response.status,
                            error
                                .message
                                .as_deref()
                                .unwrap_or(error.code.as_deref().unwrap_or("unknown"))
                        ),
                        None => eprintln!("响应未正常完成: {:?}", response.status),
                    }
                }
                if let Some(usage) = &response.usage {
                    println!(
                        "[tokens: in {} · out {} · total {}]",
                        usage.input_tokens, usage.output_tokens, usage.total_tokens
                    );
                }
            },
        }
    }

    fn end_stream(&mut self) {
        if self.reasoning || self.text {
            if self.color {
                print!("\x1b[0m");
            }
            println!();
        }
        self.reasoning = false;
        self.text = false;
    }

    fn finish(&mut self) {
        self.end_stream();
    }
}

/// Truncate a string to `max` chars, appending an ellipsis when cut.
fn truncate_chars(text: &str, max: usize) -> String {
    let chars: Vec<char> = text.chars().collect();
    if chars.len() <= max {
        text.to_string()
    } else {
        format!("{}…", chars[..max].iter().collect::<String>())
    }
}

/// Show at most `max_lines` lines of a tool result, each capped to `max_chars`.
fn preview_output(text: &str) -> String {
    let max_lines = 20;
    let max_chars = 120;

    let mut out = String::new();
    let mut lines = text.lines();
    for line in lines.by_ref().take(max_lines) {
        if !out.is_empty() {
            out.push('\n');
        }
        out.push_str(&truncate_chars(line, max_chars));
    }
    if lines.next().is_some() {
        out.push_str("\n…");
    }
    out
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn truncate_chars_leaves_short_strings_alone() {
        assert_eq!(truncate_chars("abc", 10), "abc");
    }

    #[test]
    fn truncate_chars_adds_an_ellipsis() {
        assert_eq!(truncate_chars("abcdef", 3), "abc…");
    }

    #[test]
    fn preview_output_caps_lines_and_width() {
        let text = "a".repeat(200) + "\n" + &"b".repeat(5) + "\n" + &"c".repeat(5);
        let preview = preview_output(&text);

        assert!(preview.contains('…'));
        assert!(preview.lines().count() <= 20);
    }
}
