//! Builds the system prompt injected into every Responses API request.

use std::fs;
use std::path::{Path, PathBuf};

use crate::tools::ToolSpec;

const TEMPLATE: &str = include_str!("../prompts/system.md");

/// Renders the system prompt for the given working directory and tool set,
/// including the nearest `AGENTS.md` if one exists.
///
/// The tool list and rules are derived from `tools`, never hard-coded, so the
/// prompt always matches the tools actually declared to the model.
pub fn build_system_prompt(cwd: &Path, tools: &[ToolSpec]) -> String {
    let tools_block = if tools.is_empty() { no_tools_section() } else { tools_section(tools) };
    let tool_rules = tool_rules(tools);

    TEMPLATE
        .replace("{{TOOLS}}", tools_block.trim_end())
        .replace("{{TOOL_RULES}}", &tool_rules)
        .replace("{{PROJECT_CONTEXT}}", project_context(cwd).trim_end())
        .replace("{{CWD}}", &cwd.display().to_string())
}

fn tools_section(tools: &[ToolSpec]) -> String {
    let mut section = String::from("<tools>\n");
    for tool in tools {
        section.push_str(&format!("- {}: {}\n", tool.name, tool.snippet));
    }
    section.push_str(
        "\nIn addition to the tools above, you may have access to other custom tools depending on the project.\n</tools>",
    );
    section
}

fn no_tools_section() -> String {
    String::from(
        "You have no tools available in this session. You cannot read files, execute commands, or write files; work from the conversation only and say so when a task requires tools.",
    )
}

/// Renders the tool-specific portion of the `<rules>` block.
fn tool_rules(tools: &[ToolSpec]) -> String {
    if tools.is_empty() {
        return "- You have no tools in this session: do not claim to have read files, run commands, or modified code\n"
            .to_string();
    }

    let mut rules = String::new();
    let has_bash = tools.iter().any(|tool| tool.name == "bash");
    let has_search = tools.iter().any(|tool| matches!(tool.name, "grep" | "find" | "ls"));
    if has_bash && !has_search {
        rules.push_str("- Use bash for file operations like ls, rg, find\n");
    }
    for tool in tools {
        for guideline in tool.guidelines {
            rules.push_str(&format!("- {guideline}\n"));
        }
    }
    rules
}

fn project_context(cwd: &Path) -> String {
    let Some(path) = find_agents_md(cwd) else {
        return String::new();
    };
    let Ok(content) = fs::read_to_string(&path) else {
        return String::new();
    };

    format!(
        "<project_context>\nProject-specific instructions and guidelines:\n\n<project_instructions path=\"{}\">\n{}\n</project_instructions>\n</project_context>",
        path.display(),
        content.trim_end()
    )
}

/// Walks from `cwd` upwards and returns the nearest `AGENTS.md`.
fn find_agents_md(cwd: &Path) -> Option<PathBuf> {
    cwd.ancestors()
        .map(|dir| dir.join("AGENTS.md"))
        .find(|path| path.is_file())
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::time::{SystemTime, UNIX_EPOCH};

    fn unique_temp_dir(label: &str) -> PathBuf {
        let nanos = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .expect("system clock is before the unix epoch")
            .as_nanos();
        std::env::temp_dir().join(format!("conway-prompt-{label}-{}-{nanos}", std::process::id()))
    }

    fn read_spec() -> ToolSpec {
        ToolSpec {
            name: "read",
            snippet: "Read file contents",
            description: "Read a file",
            parameters: serde_json::json!({}),
            guidelines: &["Use read to examine files instead of cat or sed"],
        }
    }

    #[test]
    fn substitutes_every_placeholder_with_no_tools() {
        let prompt = build_system_prompt(Path::new("/home/example/project"), &[]);

        assert!(prompt.contains("/home/example/project"));
        assert!(!prompt.contains("{{"), "no placeholder should survive rendering");
    }

    #[test]
    fn declares_that_no_tools_are_available() {
        let prompt = build_system_prompt(Path::new("/home/example/project"), &[]);

        assert!(prompt.contains("no tools available"));
        assert!(!prompt.contains("<tools>"));
    }

    #[test]
    fn lists_tools_and_their_guidelines() {
        let prompt = build_system_prompt(Path::new("/home/example/project"), &[read_spec()]);

        assert!(prompt.contains("<tools>"));
        assert!(prompt.contains("- read: Read file contents"));
        assert!(prompt.contains("- Use read to examine files instead of cat or sed"));
    }

    #[test]
    fn includes_the_nearest_agents_md() {
        let root = unique_temp_dir("nearest");
        let nested = root.join("nested");
        fs::create_dir_all(&nested).expect("create temp dirs");
        fs::write(root.join("AGENTS.md"), "root rules").expect("write root AGENTS.md");
        fs::write(nested.join("AGENTS.md"), "nested rules").expect("write nested AGENTS.md");

        let prompt = build_system_prompt(&nested, &[read_spec()]);

        let _ = fs::remove_dir_all(&root);
        assert!(prompt.contains("<project_context>"));
        assert!(prompt.contains("nested rules"));
        assert!(!prompt.contains("root rules"), "the nearest AGENTS.md wins");
        assert!(prompt.contains(&nested.join("AGENTS.md").display().to_string()));
    }

    #[test]
    fn omits_project_context_without_agents_md() {
        let dir = unique_temp_dir("empty");
        fs::create_dir_all(&dir).expect("create temp dir");

        let prompt = build_system_prompt(&dir, &[read_spec()]);

        let _ = fs::remove_dir_all(&dir);
        assert!(!prompt.contains("<project_context>"));
    }
}
