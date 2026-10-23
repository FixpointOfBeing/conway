//! `edit` tool: precise multi-edit replacement against a file's original
//! content, preserving line endings and BOM.

use std::path::Path;

use serde_json::{Value, json};

use crate::tools::paths::resolve_for_mutation;
use crate::tools::{ToolContext, ToolOutput, ToolSpec};

pub struct EditTool;

/// One parsed replacement.
struct Edit {
    old_text: String,
    new_text: String,
}

/// A located replacement plus its display coordinates for the diff summary.
struct AppliedEdit {
    start_line: usize,
    old_lines: Vec<String>,
    new_lines: Vec<String>,
}

impl EditTool {
    pub fn spec(&self) -> ToolSpec {
        ToolSpec {
            name: "edit",
            snippet: "Make precise file edits with exact text replacement, including multiple disjoint edits in one call",
            description: "Edit a single file using exact text replacement. Every edits[].oldText must match a unique, non-overlapping region of the original file. If two changes affect the same block or nearby lines, merge them into one edit instead of emitting overlapping edits. Do not include large unchanged regions just to connect distant changes.",
            parameters: json!({
                "type": "object",
                "properties": {
                    "path": {
                        "type": "string",
                        "description": "Path to the file to edit (relative or absolute)"
                    },
                    "edits": {
                        "type": "array",
                        "description": "One or more targeted replacements. Each edit is matched against the original file, not incrementally.",
                        "items": {
                            "type": "object",
                            "properties": {
                                "oldText": {
                                    "type": "string",
                                    "description": "Exact text to replace. Must be unique in the file."
                                },
                                "newText": {
                                    "type": "string",
                                    "description": "Replacement text."
                                }
                            },
                            "required": ["oldText", "newText"],
                            "additionalProperties": false
                        }
                    }
                },
                "required": ["path", "edits"],
                "additionalProperties": false
            }),
            guidelines: &[
                "Use edit for precise changes (edits[].oldText must match exactly)",
                "When changing multiple separate locations in one file, use one edit call with multiple entries in edits[] instead of multiple edit calls",
                "Each edits[].oldText is matched against the original file, not after earlier edits are applied. Do not emit overlapping or nested edits. Merge nearby changes into one edit.",
                "Keep edits[].oldText as small as possible while still being unique in the file. Do not pad with large unchanged regions.",
            ],
        }
    }

    pub async fn execute(&self, ctx: &ToolContext, args: Value) -> ToolOutput {
        let Some(path) = args.get("path").and_then(Value::as_str) else {
            return ToolOutput::err("edit requires a string `path` argument");
        };

        let Some(edits) = parse_edits(args.get("edits")) else {
            return ToolOutput::err("edit requires an `edits` array (or a single edit object)");
        };
        if edits.is_empty() {
            return ToolOutput::err("edit: `edits` must contain at least one replacement");
        }

        let absolute = match resolve_for_mutation(&ctx.cwd, path) {
            Ok(path) => path,
            Err(error) => return ToolOutput::err(error),
        };

        let bytes = match tokio::fs::read(&absolute).await {
            Ok(bytes) => bytes,
            Err(error) => return ToolOutput::err(format!("edit: cannot read {path}: {error}")),
        };
        let raw = match String::from_utf8(bytes) {
            Ok(text) => text,
            Err(_) => {
                return ToolOutput::err(format!(
                    "edit: {path} is not valid UTF-8 text (binary file?)"
                ));
            },
        };

        let (bom, text) = split_bom(&raw);
        let ending = detect_line_ending(text);
        let content = to_lf(text);

        // Normalize the edits to LF too, so oldText copied from a CRLF file
        // (via `read`) still matches the LF-normalized content.
        let normalized_edits: Vec<(String, String)> = edits
            .iter()
            .map(|edit| (to_lf(&edit.old_text), to_lf(&edit.new_text)))
            .collect();

        let (new_content, applied) = match apply_edits(&content, &normalized_edits) {
            Ok(result) => result,
            Err(error) => return ToolOutput::err(error),
        };

        let mut final_content = String::with_capacity(new_content.len() + bom.len());
        final_content.push_str(bom);
        final_content.push_str(&restore_ending(&new_content, ending));

        if let Err(error) = write_atomic(&absolute, &final_content).await {
            return ToolOutput::err(format!("edit: cannot write {path}: {error}"));
        }

        ToolOutput::ok(format!(
            "Successfully replaced {} block(s) in {}.\n\n{}",
            applied.len(),
            path,
            diff_summary(&applied)
        ))
    }
}

/// Accepts an `edits` array or, for tolerant models, a single edit object.
fn parse_edits(value: Option<&Value>) -> Option<Vec<Edit>> {
    let objects: Vec<&Value> = match value {
        Some(Value::Array(items)) => items.iter().collect(),
        Some(object @ Value::Object(_)) => vec![object],
        _ => return None,
    };

    let mut edits = Vec::with_capacity(objects.len());
    for object in objects {
        let old_text = object.get("oldText")?.as_str()?.to_string();
        let new_text = object.get("newText")?.as_str()?.to_string();
        edits.push(Edit { old_text, new_text });
    }
    Some(edits)
}

fn split_bom(text: &str) -> (&str, &str) {
    if let Some(rest) = text.strip_prefix('\u{feff}') { ("\u{feff}", rest) } else { ("", text) }
}

/// Returns `\r\n` only when the first CRLF precedes any lone LF.
fn detect_line_ending(text: &str) -> &'static str {
    match (text.find('\n'), text.find("\r\n")) {
        (None, _) => "\n",
        (Some(_), None) => "\n",
        (Some(lf), Some(crlf)) => {
            if crlf < lf {
                "\r\n"
            } else {
                "\n"
            }
        },
    }
}

fn to_lf(text: &str) -> String {
    text.replace("\r\n", "\n").replace('\r', "\n")
}

fn restore_ending(text: &str, ending: &str) -> String {
    if ending == "\r\n" { text.replace('\n', "\r\n") } else { text.to_string() }
}

fn line_col(content: &str, offset: usize) -> usize {
    content[..offset].bytes().filter(|&byte| byte == b'\n').count() + 1
}

fn apply_edits(
    content: &str,
    edits: &[(String, String)],
) -> Result<(String, Vec<AppliedEdit>), String> {
    let mut located: Vec<(usize, &str, &str)> = Vec::with_capacity(edits.len());

    for (old_text, new_text) in edits {
        if old_text.is_empty() {
            return Err("edit: `oldText` must not be empty".to_string());
        }

        let Some(first) = content.find(old_text.as_str()) else {
            let preview = truncate(old_text, 80);
            return Err(format!("edit: `oldText` was not found in the file:\n{preview}"));
        };
        if content[first + old_text.len()..].contains(old_text.as_str()) {
            let preview = truncate(old_text, 80);
            return Err(format!(
                "edit: `oldText` is not unique in the file; add more surrounding context:\n{preview}"
            ));
        }

        located.push((first, old_text.as_str(), new_text.as_str()));
    }

    located.sort_by_key(|(offset, ..)| *offset);

    for pair in located.windows(2) {
        let (first_offset, first_old, _) = pair[0];
        let (second_offset, ..) = pair[1];
        if first_offset + first_old.len() > second_offset {
            return Err("edit: edits overlap; merge nearby changes into one edit".to_string());
        }
    }

    let applied: Vec<AppliedEdit> = located
        .iter()
        .map(|(offset, old, new)| AppliedEdit {
            start_line: line_col(content, *offset),
            old_lines: old.split('\n').map(str::to_string).collect(),
            new_lines: new.split('\n').map(str::to_string).collect(),
        })
        .collect();

    let mut result = content.to_string();
    for (offset, old, new) in located.iter().rev() {
        result.replace_range(*offset..*offset + old.len(), new);
    }

    Ok((result, applied))
}

/// Writes atomically via a temp file in the same directory, preserving the
/// original file's permissions.
async fn write_atomic(path: &Path, content: &str) -> std::io::Result<()> {
    let dir = path.parent().unwrap_or_else(|| Path::new("."));
    let name = path
        .file_name()
        .ok_or_else(|| std::io::Error::other("path has no file name"))?;
    let tmp = dir.join(format!(".{}.conway-tmp", name.to_string_lossy()));

    let permissions = tokio::fs::metadata(path).await.map(|meta| meta.permissions()).ok();

    tokio::fs::write(&tmp, content).await?;
    if let Some(permissions) = permissions {
        let _ = tokio::fs::set_permissions(&tmp, permissions).await;
    }
    if let Err(error) = tokio::fs::rename(&tmp, path).await {
        let _ = tokio::fs::remove_file(&tmp).await;
        return Err(error);
    }
    Ok(())
}

fn diff_summary(edits: &[AppliedEdit]) -> String {
    const MAX_DIFF_LINES: usize = 40;

    let mut out = String::new();
    let mut total = 0;
    for edit in edits {
        if total >= MAX_DIFF_LINES {
            out.push_str("…\n");
            break;
        }
        out.push_str(&format!(
            "@@ -{},{} +{},{} @@\n",
            edit.start_line,
            edit.old_lines.len(),
            edit.start_line,
            edit.new_lines.len()
        ));
        total += 1;
        for line in &edit.old_lines {
            if total >= MAX_DIFF_LINES {
                break;
            }
            out.push_str(&format!("- {line}\n"));
            total += 1;
        }
        for line in &edit.new_lines {
            if total >= MAX_DIFF_LINES {
                break;
            }
            out.push_str(&format!("+ {line}\n"));
            total += 1;
        }
    }
    out
}

fn truncate(text: &str, max: usize) -> String {
    let chars: Vec<char> = text.chars().collect();
    if chars.len() <= max {
        text.to_string()
    } else {
        format!("{}…", chars[..max].iter().collect::<String>())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[tokio::test]
    async fn applies_a_single_edit() {
        let dir = tempfile::tempdir().expect("tempdir");
        std::fs::write(dir.path().join("f.txt"), "fn main() {}\n").expect("seed");
        let ctx = ToolContext::new(dir.path());

        let output = EditTool
            .execute(
                &ctx,
                json!({ "path": "f.txt", "edits": [{ "oldText": "fn main() {}", "newText": "fn main() { todo!() }" }] }),
            )
            .await;

        assert!(!output.is_error, "{}", output.text);
        assert_eq!(
            std::fs::read_to_string(dir.path().join("f.txt")).expect("read"),
            "fn main() { todo!() }\n"
        );
        assert!(output.text.contains("1 block"));
        assert!(output.text.contains("- fn main() {}"));
        assert!(output.text.contains("+ fn main() { todo!() }"));
    }

    #[tokio::test]
    async fn applies_multiple_edits_against_the_original() {
        let dir = tempfile::tempdir().expect("tempdir");
        std::fs::write(dir.path().join("f.txt"), "a\nb\nc\n").expect("seed");
        let ctx = ToolContext::new(dir.path());

        // The two edits do not share text, so each must match the original.
        let output = EditTool
            .execute(
                &ctx,
                json!({ "path": "f.txt", "edits": [
                    { "oldText": "a", "newText": "A" },
                    { "oldText": "c", "newText": "C" }
                ] }),
            )
            .await;

        assert!(!output.is_error, "{}", output.text);
        assert_eq!(std::fs::read_to_string(dir.path().join("f.txt")).expect("read"), "A\nb\nC\n");
    }

    #[tokio::test]
    async fn rejects_a_non_unique_old_text() {
        let dir = tempfile::tempdir().expect("tempdir");
        std::fs::write(dir.path().join("f.txt"), "dup\ndup\n").expect("seed");
        let ctx = ToolContext::new(dir.path());

        let output = EditTool
            .execute(
                &ctx,
                json!({ "path": "f.txt", "edits": [{ "oldText": "dup", "newText": "x" }] }),
            )
            .await;

        assert!(output.is_error);
        assert!(output.text.contains("not unique"));
    }

    #[tokio::test]
    async fn rejects_overlapping_edits() {
        let dir = tempfile::tempdir().expect("tempdir");
        std::fs::write(dir.path().join("f.txt"), "hello world\n").expect("seed");
        let ctx = ToolContext::new(dir.path());

        let output = EditTool
            .execute(
                &ctx,
                json!({ "path": "f.txt", "edits": [
                    { "oldText": "hello world", "newText": "hi" },
                    { "oldText": "world", "newText": "there" }
                ] }),
            )
            .await;

        assert!(output.is_error);
        assert!(output.text.contains("overlap"));
    }

    #[tokio::test]
    async fn preserves_crlf_line_endings() {
        let dir = tempfile::tempdir().expect("tempdir");
        std::fs::write(dir.path().join("f.txt"), "a\r\nb\r\n").expect("seed");
        let ctx = ToolContext::new(dir.path());

        let output = EditTool
            .execute(
                &ctx,
                json!({ "path": "f.txt", "edits": [{ "oldText": "b", "newText": "B" }] }),
            )
            .await;

        assert!(!output.is_error, "{}", output.text);
        assert_eq!(std::fs::read_to_string(dir.path().join("f.txt")).expect("read"), "a\r\nB\r\n");
    }

    #[tokio::test]
    async fn preserves_a_utf8_bom() {
        let dir = tempfile::tempdir().expect("tempdir");
        std::fs::write(dir.path().join("f.txt"), "\u{feff}let x = 1;\n").expect("seed");
        let ctx = ToolContext::new(dir.path());

        let output = EditTool
            .execute(
                &ctx,
                json!({ "path": "f.txt", "edits": [{ "oldText": "let x = 1;", "newText": "let x = 2;" }] }),
            )
            .await;

        assert!(!output.is_error, "{}", output.text);
        assert_eq!(
            std::fs::read_to_string(dir.path().join("f.txt")).expect("read"),
            "\u{feff}let x = 2;\n"
        );
    }

    #[tokio::test]
    async fn accepts_a_single_edit_object_instead_of_an_array() {
        let dir = tempfile::tempdir().expect("tempdir");
        std::fs::write(dir.path().join("f.txt"), "a\n").expect("seed");
        let ctx = ToolContext::new(dir.path());

        let output = EditTool
            .execute(&ctx, json!({ "path": "f.txt", "edits": { "oldText": "a", "newText": "A" } }))
            .await;

        assert!(!output.is_error, "{}", output.text);
        assert_eq!(std::fs::read_to_string(dir.path().join("f.txt")).expect("read"), "A\n");
    }

    #[tokio::test]
    async fn reports_a_missing_file() {
        let dir = tempfile::tempdir().expect("tempdir");
        let ctx = ToolContext::new(dir.path());

        let output = EditTool
            .execute(
                &ctx,
                json!({ "path": "nope.txt", "edits": [{ "oldText": "a", "newText": "b" }] }),
            )
            .await;

        assert!(output.is_error);
        assert!(output.text.contains("nope.txt"));
    }
}
