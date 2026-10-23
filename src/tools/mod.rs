//! Tool registry and shared types.
//!
//! A tool is a name, a JSON-schema parameter set, and an `execute` function
//! that turns parsed arguments plus a working directory into a model-facing
//! text result. Tools never throw: errors are returned as `ToolOutput` with
//! `is_error = true` so the model can recover.

pub mod edit;
pub mod paths;
pub mod read;
pub mod write;

use std::path::PathBuf;

use serde_json::Value;

use crate::llm::ToolDefinition;

/// Static metadata used both for the provider `tools` array and the system
/// prompt's tool listing.
pub struct ToolSpec {
    pub name: &'static str,
    /// One-line summary shown in the system prompt.
    pub snippet: &'static str,
    /// Longer description shown in the tool definition sent to the provider.
    pub description: &'static str,
    pub parameters: Value,
    /// Guideline bullets rendered into the system prompt's `<rules>` block.
    pub guidelines: &'static [&'static str],
}

/// Working-directory context passed to tool executions.
#[derive(Debug, Clone)]
pub struct ToolContext {
    pub cwd: PathBuf,
}

impl ToolContext {
    pub fn new(cwd: impl Into<PathBuf>) -> Self {
        Self { cwd: cwd.into() }
    }
}

/// Model-facing result of a tool execution.
pub struct ToolOutput {
    pub text: String,
    pub is_error: bool,
}

impl ToolOutput {
    pub fn ok(text: impl Into<String>) -> Self {
        Self { text: text.into(), is_error: false }
    }

    pub fn err(text: impl Into<String>) -> Self {
        Self { text: text.into(), is_error: true }
    }
}

/// The concrete set of tools conway knows how to execute.
pub enum Tool {
    Read(read::ReadTool),
    Edit(edit::EditTool),
    Write(write::WriteTool),
}

impl Tool {
    pub fn spec(&self) -> ToolSpec {
        match self {
            Tool::Read(tool) => tool.spec(),
            Tool::Edit(tool) => tool.spec(),
            Tool::Write(tool) => tool.spec(),
        }
    }

    pub async fn execute(&self, ctx: &ToolContext, args: Value) -> ToolOutput {
        match self {
            Tool::Read(tool) => tool.execute(ctx, args).await,
            Tool::Edit(tool) => tool.execute(ctx, args).await,
            Tool::Write(tool) => tool.execute(ctx, args).await,
        }
    }
}

/// The tools available to the agent loop.
pub struct ToolRegistry {
    tools: Vec<Tool>,
}

impl ToolRegistry {
    /// The default coding toolset.
    pub fn coding() -> Self {
        Self {
            tools: vec![
                Tool::Read(read::ReadTool),
                Tool::Edit(edit::EditTool),
                Tool::Write(write::WriteTool),
            ],
        }
    }

    /// Prompt metadata for every registered tool.
    pub fn specs(&self) -> Vec<ToolSpec> {
        self.tools.iter().map(Tool::spec).collect()
    }

    /// Provider-facing tool definitions.
    pub fn definitions(&self) -> Vec<ToolDefinition> {
        self.specs()
            .into_iter()
            .map(|spec| {
                ToolDefinition::function(spec.name, spec.description, spec.parameters.clone())
            })
            .collect()
    }

    pub fn get(&self, name: &str) -> Option<&Tool> {
        self.tools.iter().find(|tool| tool.spec().name == name)
    }

    pub fn is_empty(&self) -> bool {
        self.tools.is_empty()
    }
}
