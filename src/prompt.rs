//! Builds the system prompt injected into every Responses API request.

use std::fs;
use std::path::{Path, PathBuf};

const TEMPLATE: &str = include_str!("../prompts/system.md");

/// Tools announced to the model.
///
/// Empty for now: the agent loop does not execute tools yet, so the prompt
/// must not claim that files can be read or commands can be run. When tools
/// are implemented, register them here and the tool block and edit rules are
/// rendered automatically.
fn available_tools() -> &'static [(&'static str, &'static str)] {
    &[
        // ("read", "Read file contents"),
        // ("bash", "Execute bash commands (ls, rg, find, etc.)"),
        // ("edit", "Make precise file edits with exact text replacement, including multiple disjoint edits in one call"),
        // ("write", "Create or overwrite files"),
    ]
}

const TOOL_RULES: &str = "\
- Use bash for file operations like ls, rg, find
- Use read to examine files instead of cat or sed
- Use edit for precise changes (edits[].oldText must match exactly)
- When changing multiple separate locations in one file, use one edit call with multiple entries in edits[] instead of multiple edit calls
- Each edits[].oldText is matched against the original file, not after earlier edits are applied. Do not emit overlapping or nested edits. Merge nearby changes into one edit.
- Keep edits[].oldText as small as possible while still unique in the file. Do not pad with large unchanged regions.
- Use write only for new files or complete rewrites
";

const NO_TOOL_RULES: &str = "- You have no tools in this session: do not claim to have read files, run commands, or modified code\n";

/// Renders the system prompt for the given working directory, including the
/// nearest `AGENTS.md` if one exists.
pub fn build_system_prompt(cwd: &Path) -> String {
    let (rules, tools) = if available_tools().is_empty() {
        (NO_TOOL_RULES, no_tools_section())
    } else {
        (TOOL_RULES, tools_section())
    };

    TEMPLATE
        .replace("{{TOOLS}}", tools.trim_end())
        .replace("{{TOOL_RULES}}", rules)
        .replace("{{PROJECT_CONTEXT}}", project_context(cwd).trim_end())
        .replace("{{CWD}}", &cwd.display().to_string())
}

fn tools_section() -> String {
    let mut section = String::from("<tools>\n");
    for (name, description) in available_tools() {
        section.push_str(&format!("- {name}: {description}\n"));
    }
    section.push_str("\nIn addition to the tools above, you may have access to other custom tools depending on the project.\n</tools>");
    section
}

fn no_tools_section() -> String {
    String::from(
        "You have no tools available in this session. You cannot read files, execute commands, or write files; work from the conversation only and say so when a task requires tools.",
    )
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

    #[test]
    fn substitutes_every_placeholder() {
        let prompt = build_system_prompt(Path::new("/home/example/project"));

        assert!(prompt.contains("/home/example/project"));
        assert!(!prompt.contains("{{"), "no placeholder should survive rendering");
    }

    #[test]
    fn declares_that_no_tools_are_available() {
        let prompt = build_system_prompt(Path::new("/home/example/project"));

        assert!(prompt.contains("no tools available"));
        assert!(!prompt.contains("<tools>"));
        assert!(!prompt.contains("edits[].oldText"));
    }

    #[test]
    fn includes_the_nearest_agents_md() {
        let root = unique_temp_dir("nearest");
        let nested = root.join("nested");
        fs::create_dir_all(&nested).expect("create temp dirs");
        fs::write(root.join("AGENTS.md"), "root rules").expect("write root AGENTS.md");
        fs::write(nested.join("AGENTS.md"), "nested rules").expect("write nested AGENTS.md");

        let prompt = build_system_prompt(&nested);

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

        let prompt = build_system_prompt(&dir);

        let _ = fs::remove_dir_all(&dir);
        assert!(!prompt.contains("<project_context>"));
    }
}
