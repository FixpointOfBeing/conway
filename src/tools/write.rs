//! `write` tool: creates or overwrites a file.

use serde_json::{Value, json};

use crate::tools::paths::resolve_for_mutation;
use crate::tools::{ToolContext, ToolOutput, ToolSpec};

pub struct WriteTool;

impl WriteTool {
    pub fn spec(&self) -> ToolSpec {
        ToolSpec {
            name: "write",
            snippet: "Create or overwrite files",
            description: "Write content to a file. Creates the file if it doesn't exist, overwrites if it does. Automatically creates parent directories.",
            parameters: json!({
                "type": "object",
                "properties": {
                    "path": {
                        "type": "string",
                        "description": "Path to the file to write (relative or absolute)"
                    },
                    "content": {
                        "type": "string",
                        "description": "Content to write to the file"
                    }
                },
                "required": ["path", "content"],
                "additionalProperties": false
            }),
            guidelines: &["Use write only for new files or complete rewrites"],
        }
    }

    pub async fn execute(&self, ctx: &ToolContext, args: Value) -> ToolOutput {
        let Some(path) = args.get("path").and_then(Value::as_str) else {
            return ToolOutput::err("write requires a string `path` argument");
        };
        let Some(content) = args.get("content").and_then(Value::as_str) else {
            return ToolOutput::err("write requires a string `content` argument");
        };

        let absolute = match resolve_for_mutation(&ctx.cwd, path) {
            Ok(path) => path,
            Err(error) => return ToolOutput::err(error),
        };

        if let Some(parent) = absolute.parent()
            && let Err(error) = tokio::fs::create_dir_all(parent).await
        {
            return ToolOutput::err(format!(
                "write: cannot create parent directory for {path}: {error}"
            ));
        }

        match tokio::fs::write(&absolute, content).await {
            Ok(()) => {
                ToolOutput::ok(format!("Successfully wrote {} byte(s) to {path}.", content.len()))
            },
            Err(error) => ToolOutput::err(format!("write: cannot write {path}: {error}")),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[tokio::test]
    async fn writes_a_new_file_creating_parent_directories() {
        let dir = tempfile::tempdir().expect("tempdir");
        let ctx = ToolContext::new(dir.path());

        let output = WriteTool
            .execute(&ctx, json!({ "path": "a/b/hello.txt", "content": "hi" }))
            .await;

        assert!(!output.is_error, "{}", output.text);
        assert_eq!(std::fs::read_to_string(dir.path().join("a/b/hello.txt")).expect("read"), "hi");
    }

    #[tokio::test]
    async fn overwrites_an_existing_file() {
        let dir = tempfile::tempdir().expect("tempdir");
        std::fs::write(dir.path().join("f.txt"), "old").expect("seed");
        let ctx = ToolContext::new(dir.path());

        let output = WriteTool
            .execute(&ctx, json!({ "path": "f.txt", "content": "new" }))
            .await;

        assert!(!output.is_error, "{}", output.text);
        assert_eq!(std::fs::read_to_string(dir.path().join("f.txt")).expect("read"), "new");
    }

    #[tokio::test]
    async fn rejects_a_path_that_escapes_cwd() {
        let dir = tempfile::tempdir().expect("tempdir");
        let ctx = ToolContext::new(dir.path());

        let output = WriteTool
            .execute(&ctx, json!({ "path": "../escape.txt", "content": "x" }))
            .await;

        assert!(output.is_error);
        assert!(output.text.contains("outside the project root"));
    }
}
