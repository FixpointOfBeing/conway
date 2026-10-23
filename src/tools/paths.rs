//! Path resolution for tools that mutate the filesystem.

use std::ffi::OsString;
use std::fs;
use std::path::{Path, PathBuf};

/// Resolve `raw` against `cwd` for a mutating tool, confining the result to
/// the project root.
///
/// Absolute paths and `..` escapes that land outside `cwd` are rejected.
/// Symlinks are resolved on the nearest existing ancestor, so a link inside
/// the project cannot redirect a write to an unrelated location.
pub fn resolve_for_mutation(cwd: &Path, raw: &str) -> Result<PathBuf, String> {
    if raw.is_empty() {
        return Err("path must not be empty".to_string());
    }

    let candidate = if Path::new(raw).is_absolute() { PathBuf::from(raw) } else { cwd.join(raw) };

    let canonical_cwd = fs::canonicalize(cwd)
        .map_err(|error| format!("cannot resolve working directory {}: {error}", cwd.display()))?;

    // Walk up from the candidate to its nearest existing ancestor and
    // canonicalize it, so symlinks resolve to their real location. The
    // non-existing tail is re-attached afterwards.
    let mut tail: Vec<OsString> = Vec::new();
    let mut current = candidate.as_path();
    let base = loop {
        match fs::canonicalize(current) {
            Ok(real) => break real,
            Err(_) => {
                let name = current
                    .file_name()
                    .ok_or_else(|| format!("cannot resolve path: {raw}"))?;
                tail.push(name.to_os_string());
                match current.parent() {
                    Some(parent) if !parent.as_os_str().is_empty() => current = parent,
                    _ => return Err(format!("cannot resolve path: {raw}")),
                }
            },
        }
    };

    // Reattach the tail, collapsing any `.`/`..` components that slipped through.
    let mut resolved = base;
    for component in tail.iter().rev() {
        match component.to_str() {
            Some(".") => {},
            Some("..") => {
                resolved.pop();
            },
            _ => resolved.push(component),
        }
    }

    if !resolved.starts_with(&canonical_cwd) {
        return Err(format!("refusing to write outside the project root: {raw}"));
    }

    Ok(resolved)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn resolves_a_relative_path_inside_cwd() {
        let dir = tempfile::tempdir().expect("tempdir");
        std::fs::create_dir_all(dir.path().join("src")).expect("mkdir");

        let resolved = resolve_for_mutation(dir.path(), "src/main.rs").expect("resolves");

        assert_eq!(resolved, dir.path().join("src").join("main.rs"));
    }

    #[test]
    fn rejects_a_dot_dot_escape() {
        let dir = tempfile::tempdir().expect("tempdir");

        let err = resolve_for_mutation(dir.path(), "../outside.txt").expect_err("escapes cwd");

        assert!(err.contains("outside the project root"));
    }

    #[test]
    fn rejects_an_absolute_path_outside_cwd() {
        let dir = tempfile::tempdir().expect("tempdir");
        let outside = std::env::temp_dir().join("conway-absolute-escape-test.txt");

        let err = resolve_for_mutation(dir.path(), outside.to_str().expect("utf8"))
            .expect_err("absolute outside");

        assert!(err.contains("outside the project root"));
    }

    #[cfg(unix)]
    #[test]
    fn rejects_a_symlink_that_escapes_cwd() {
        use std::os::unix::fs::symlink;

        let dir = tempfile::tempdir().expect("tempdir");
        let target = tempfile::tempdir().expect("target dir");
        symlink(target.path(), dir.path().join("link")).expect("symlink");

        let err = resolve_for_mutation(dir.path(), "link/new.txt").expect_err("symlink escapes");

        assert!(err.contains("outside the project root"));
    }
}
