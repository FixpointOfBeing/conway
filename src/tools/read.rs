//! `read` tool: returns the contents of a text file, truncated.

use std::path::{Path, PathBuf};

use serde_json::{Value, json};

use crate::tools::{ToolContext, ToolOutput, ToolSpec};

const DEFAULT_MAX_LINES: usize = 2000;
const DEFAULT_MAX_BYTES: usize = 50 * 1024;

pub struct ReadTool;

impl ReadTool {
    pub fn spec(&self) -> ToolSpec {
        ToolSpec {
            name: "read",
            snippet: "Read file contents",
            description: "Read the contents of a text file. Output is truncated to 2000 lines or 50KB (whichever is hit first). Use offset/limit for large files. When you need the full file, continue with offset until complete.",
            parameters: json!({
                "type": "object",
                "properties": {
                    "path": {
                        "type": "string",
                        "description": "Path to the file to read (relative or absolute)"
                    },
                    "offset": {
                        "type": "integer",
                        "description": "Line number to start reading from (1-indexed)"
                    },
                    "limit": {
                        "type": "integer",
                        "description": "Maximum number of lines to read"
                    }
                },
                "required": ["path"],
                "additionalProperties": false
            }),
            guidelines: &["Use read to examine files instead of cat or sed"],
        }
    }

    pub async fn execute(&self, ctx: &ToolContext, args: Value) -> ToolOutput {
        let Some(path) = args.get("path").and_then(Value::as_str) else {
            return ToolOutput::err("read requires a string `path` argument");
        };
        if path.is_empty() {
            return ToolOutput::err("read: `path` must not be empty");
        }

        let offset = args
            .get("offset")
            .and_then(Value::as_u64)
            .map(|v| v.max(1) as usize)
            .unwrap_or(1);
        let limit = args.get("limit").and_then(Value::as_u64).map(|v| v as usize);
        if limit == Some(0) {
            return ToolOutput::err("read: `limit` must be a positive integer");
        }

        let absolute = resolve_path(&ctx.cwd, path);

        let bytes = match tokio::fs::read(&absolute).await {
            Ok(bytes) => bytes,
            Err(error) => {
                return ToolOutput::err(format!(
                    "read: cannot read {}: {error}",
                    absolute.display()
                ));
            },
        };

        match String::from_utf8(bytes) {
            Ok(text) => render(&text, offset, limit),
            Err(_) => ToolOutput::err(format!(
                "read: {} is not valid UTF-8 text (binary file?)",
                absolute.display()
            )),
        }
    }
}

fn resolve_path(cwd: &Path, raw: &str) -> PathBuf {
    let path = Path::new(raw);
    if path.is_absolute() { path.to_path_buf() } else { cwd.join(path) }
}

fn render(text: &str, offset: usize, limit: Option<usize>) -> ToolOutput {
    // Split mirrors the provider convention: a trailing newline yields one
    // final empty line, which is included in the line count.
    let all_lines: Vec<&str> = text.split('\n').collect();
    let total_lines = all_lines.len();

    let start = offset.saturating_sub(1);
    if start >= total_lines {
        return ToolOutput::err(format!(
            "read: offset {offset} is beyond end of file ({total_lines} lines total)"
        ));
    }

    let selected: Vec<&str> = match limit {
        Some(limit) => all_lines[start..all_lines.len().min(start + limit)].to_vec(),
        None => all_lines[start..].to_vec(),
    };

    let (kept, truncated, by) = truncate_head(&selected);

    let mut output = kept.join("\n");

    let start_display = start + 1;
    let shown = kept.len();
    let end_display = start_display + shown.saturating_sub(1);
    let next_offset = end_display + 1;

    if truncated {
        let reason = match by {
            "lines" => format!("{DEFAULT_MAX_LINES} line limit"),
            _ => format!("{}KB limit", DEFAULT_MAX_BYTES / 1024),
        };
        output.push_str(&format!(
            "\n\n[Showing lines {start_display}-{end_display} of {total_lines} ({reason}). Use offset={next_offset} to continue.]"
        ));
    } else if let Some(limit) = limit
        && shown == limit
        && start + shown < total_lines
    {
        let remaining = total_lines - (start + shown);
        output.push_str(&format!(
            "\n\n[{remaining} more line{} in file. Use offset={next_offset} to continue.]",
            if remaining == 1 { "" } else { "s" }
        ));
    }

    ToolOutput::ok(output)
}

/// Keep the first `max_lines` / `max_bytes`, whichever is hit first. Never
/// splits a line; if the very first line alone exceeds the byte limit it is
/// still kept (partially) so the result is never empty.
fn truncate_head<'a>(lines: &'a [&'a str]) -> (Vec<&'a str>, bool, &'static str) {
    let mut kept: Vec<&str> = Vec::new();
    let mut bytes = 0usize;
    let mut truncated = false;
    let mut by = "lines";

    for line in lines {
        if kept.len() >= DEFAULT_MAX_LINES {
            truncated = true;
            by = "lines";
            break;
        }
        let line_bytes = line.len() + 1; // +1 for the newline separator
        if bytes + line_bytes > DEFAULT_MAX_BYTES {
            if kept.is_empty() {
                kept.push(line);
            }
            truncated = true;
            by = "bytes";
            break;
        }
        kept.push(line);
        bytes += line_bytes;
    }

    (kept, truncated, by)
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::fs;

    #[test]
    fn render_applies_offset_and_limit() {
        let output = render("a\nb\nc\nd", 2, Some(2));

        assert!(!output.is_error);
        assert!(output.text.starts_with("b\nc"));
        assert!(output.text.contains("more line"), "expected a continuation footer");
        assert!(output.text.contains("offset=4"));
    }

    #[test]
    fn render_reports_offset_beyond_end() {
        let output = render("a\nb\n", 5, None);

        assert!(output.is_error);
        assert!(output.text.contains("beyond end"));
    }

    #[tokio::test]
    async fn execute_reads_a_relative_path() {
        let dir = tempfile::tempdir().expect("tempdir");
        fs::write(dir.path().join("hello.txt"), "hello\nworld\n").expect("write fixture");

        let ctx = ToolContext::new(dir.path());
        let output = ReadTool.execute(&ctx, json!({ "path": "hello.txt" })).await;

        assert!(!output.is_error);
        assert_eq!(output.text, "hello\nworld\n");
    }

    #[tokio::test]
    async fn execute_reports_a_missing_file() {
        let dir = tempfile::tempdir().expect("tempdir");
        let ctx = ToolContext::new(dir.path());

        let output = ReadTool.execute(&ctx, json!({ "path": "nope.txt" })).await;

        assert!(output.is_error);
        assert!(output.text.contains("nope.txt"));
    }

    #[tokio::test]
    async fn execute_rejects_non_utf8_bytes() {
        let dir = tempfile::tempdir().expect("tempdir");
        fs::write(dir.path().join("bin.dat"), [0xFFu8, 0xFE, 0x00]).expect("write fixture");

        let ctx = ToolContext::new(dir.path());
        let output = ReadTool.execute(&ctx, json!({ "path": "bin.dat" })).await;

        assert!(output.is_error);
        assert!(output.text.contains("not valid UTF-8"));
    }

    #[test]
    fn truncate_head_respects_the_line_limit() {
        let lines: Vec<String> = (0..(DEFAULT_MAX_LINES + 10)).map(|i| i.to_string()).collect();
        let refs: Vec<&str> = lines.iter().map(String::as_str).collect();

        let (kept, truncated, by) = truncate_head(&refs);

        assert!(truncated);
        assert_eq!(by, "lines");
        assert_eq!(kept.len(), DEFAULT_MAX_LINES);
    }
}
