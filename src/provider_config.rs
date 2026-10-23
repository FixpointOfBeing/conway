use serde::{Deserialize, Serialize};
use std::collections::BTreeMap;
use std::fmt;
use std::fs;
use std::path::Path;

/// Mirrors the shape of `~/.codex/config.toml`.
///
/// Unknown keys are ignored on purpose: Codex configs carry many unrelated
/// settings (`approval_policy`, `sandbox_mode`, ...) and must still load here.
#[derive(Deserialize, Debug, PartialEq)]
pub struct Config {
    pub model: String,
    #[serde(default = "default_model_provider")]
    pub model_provider: String,
    #[serde(default)]
    pub model_reasoning_effort: Option<ReasoningEffort>,
    #[serde(default)]
    pub model_providers: BTreeMap<String, ModelProvider>,
}

#[derive(Deserialize, Serialize, Debug, PartialEq, Eq, Clone, Copy)]
#[serde(rename_all = "lowercase")]
pub enum ReasoningEffort {
    None,
    Minimal,
    Low,
    Medium,
    High,
    Max,
}

impl std::fmt::Display for ReasoningEffort {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            ReasoningEffort::None => write!(f, "none"),
            ReasoningEffort::Minimal => write!(f, "minimal"),
            ReasoningEffort::Low => write!(f, "low"),
            ReasoningEffort::Medium => write!(f, "medium"),
            ReasoningEffort::High => write!(f, "high"),
            ReasoningEffort::Max => write!(f, "max"),
        }
    }
}

#[derive(Deserialize, Debug, PartialEq)]
pub struct ModelProvider {
    pub name: String,
    pub base_url: String,
    #[serde(default)]
    pub env_key: Option<String>,
    #[serde(default)]
    pub wire_api: WireApi,
}

#[derive(Deserialize, Debug, Default, PartialEq, Eq, Clone, Copy)]
#[serde(rename_all = "lowercase")]
pub enum WireApi {
    #[default]
    #[serde(alias = "response")]
    Responses,
    Chat,
}

#[derive(Debug, PartialEq, Eq)]
pub enum ConfigError {
    UnknownModelProvider { id: String, available: Vec<String> },
}

impl fmt::Display for ConfigError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::UnknownModelProvider { id, available } if available.is_empty() => write!(
                f,
                "model provider `{id}` is not defined and no [model_providers.*] entries exist"
            ),
            Self::UnknownModelProvider { id, available } => write!(
                f,
                "model provider `{id}` is not defined; available providers: {}",
                available.join(", ")
            ),
        }
    }
}

impl std::error::Error for ConfigError {}

impl Config {
    /// Resolves the `model_provider` key against `[model_providers.*]`.
    pub fn provider(&self) -> Result<&ModelProvider, ConfigError> {
        self.model_providers.get(&self.model_provider).ok_or_else(|| {
            ConfigError::UnknownModelProvider {
                id: self.model_provider.clone(),
                available: self.model_providers.keys().cloned().collect(),
            }
        })
    }
}

fn default_model_provider() -> String {
    "openai".to_string()
}

const CONFIG_PATH: &str = ".conway/config.toml";

fn parse_config(content: &str) -> Result<Config, Box<dyn std::error::Error>> {
    let config: Config = toml::from_str(content)?;
    Ok(config)
}

fn read_config_from(path: impl AsRef<Path>) -> Result<Config, Box<dyn std::error::Error>> {
    let content = fs::read_to_string(path)?;
    parse_config(&content)
}

pub fn read_config() -> Result<Config, Box<dyn std::error::Error>> {
    let home = dirs::home_dir().expect("Could not read HOME directory");
    let config_file = home.join(CONFIG_PATH);
    read_config_from(config_file)
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::path::PathBuf;
    use std::time::{SystemTime, UNIX_EPOCH};

    /// The exact config shape the user supplied, including a second provider.
    const CODEX_CONFIG: &str = r#"
model = "gpt-5.6-luna"
model_provider = "justoken"
model_reasoning_effort = "medium"

[model_providers.justoken]
name = "JustToken"
base_url = "https://justoken.ai/v1"
env_key = "JUSTOKEN_API_KEY"
wire_api = "responses"

[model_providers.deepseek]
name = "DeepSeek"
base_url = "https://api.deepseek.com/v1"
env_key = "DEEPSEEK_API_KEY"
wire_api = "responses"
"#;

    fn provider(id: &str) -> String {
        format!(
            r#"
[model_providers.{id}]
name = "{id}"
base_url = "https://{id}.example/v1"
env_key = "{id}_API_KEY"
"#
        )
    }

    fn config_with_effort(effort: &str) -> String {
        format!(
            r#"
model = "gpt-5.6-luna"
model_provider = "justoken"
model_reasoning_effort = "{effort}"
{}
"#,
            provider("justoken")
        )
    }

    /// Returns a unique path in the temp directory so parallel tests never collide.
    fn unique_temp_path(label: &str) -> PathBuf {
        let nanos = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .expect("system clock is before the unix epoch")
            .as_nanos();
        std::env::temp_dir().join(format!("conway-{label}-{}-{nanos}.toml", std::process::id()))
    }

    #[test]
    fn parses_the_codex_config_format() {
        let config = parse_config(CODEX_CONFIG).expect("codex config should parse");

        assert_eq!(config.model, "gpt-5.6-luna");
        assert_eq!(config.model_provider, "justoken");
        assert_eq!(config.model_reasoning_effort, Some(ReasoningEffort::Medium));
        assert_eq!(config.model_providers.len(), 2);
        assert_eq!(
            config.model_providers["deepseek"],
            ModelProvider {
                name: "DeepSeek".to_string(),
                base_url: "https://api.deepseek.com/v1".to_string(),
                env_key: Some("DEEPSEEK_API_KEY".to_string()),
                wire_api: WireApi::Responses,
            }
        );
    }

    #[test]
    fn resolves_the_selected_provider() {
        let config = parse_config(CODEX_CONFIG).expect("codex config should parse");

        let provider = config.provider().expect("justoken is defined");
        assert_eq!(provider.name, "JustToken");
        assert_eq!(provider.base_url, "https://justoken.ai/v1");
        assert_eq!(provider.env_key.as_deref(), Some("JUSTOKEN_API_KEY"));
        assert_eq!(provider.wire_api, WireApi::Responses);
    }

    #[test]
    fn rejects_an_unknown_model_provider() {
        let content =
            CODEX_CONFIG.replace("model_provider = \"justoken\"", "model_provider = \"nope\"");
        let config = parse_config(&content).expect("the config itself is valid");

        let err = config.provider().expect_err("`nope` is not defined");

        assert_eq!(
            err,
            ConfigError::UnknownModelProvider {
                id: "nope".to_string(),
                available: vec!["deepseek".to_string(), "justoken".to_string()],
            }
        );
        assert_eq!(
            err.to_string(),
            "model provider `nope` is not defined; available providers: deepseek, justoken"
        );
    }

    #[test]
    fn reports_no_providers_when_the_table_is_missing() {
        let content = r#"
model = "gpt-5.6-luna"
model_provider = "openai"
"#;
        let config = parse_config(content).expect("the config itself is valid");

        let err = config.provider().expect_err("no providers are defined");

        assert_eq!(
            err.to_string(),
            "model provider `openai` is not defined and no [model_providers.*] entries exist"
        );
    }

    #[test]
    fn defaults_model_provider_and_effort_like_codex() {
        let content = format!(
            r#"
model = "gpt-5.6-luna"
{}
"#,
            provider("openai")
        );

        let config = parse_config(&content).expect("the config should parse");

        assert_eq!(config.model_provider, "openai");
        assert_eq!(config.model_reasoning_effort, None);
    }

    #[test]
    fn parses_every_reasoning_effort_in_lowercase() {
        let cases = [
            ("none", ReasoningEffort::None),
            ("minimal", ReasoningEffort::Minimal),
            ("low", ReasoningEffort::Low),
            ("medium", ReasoningEffort::Medium),
            ("high", ReasoningEffort::High),
            ("max", ReasoningEffort::Max),
        ];

        for (raw, expected) in cases {
            let config = parse_config(&config_with_effort(raw))
                .unwrap_or_else(|err| panic!("`{raw}` should be valid: {err}"));
            assert_eq!(config.model_reasoning_effort, Some(expected));
        }
    }

    #[test]
    fn rejects_capitalized_or_unknown_reasoning_effort() {
        for raw in ["Medium", "ultra"] {
            assert!(
                parse_config(&config_with_effort(raw)).is_err(),
                "`{raw}` must not deserialize"
            );
        }
    }

    #[test]
    fn ignores_unrelated_codex_settings() {
        let content = r#"
model = "gpt-5.6-luna"
model_provider = "justoken"
model_reasoning_effort = "low"
approval_policy = "on-request"
sandbox_mode = "workspace-write"
model_context_window = 272000

[model_providers.justoken]
name = "JustToken"
base_url = "https://justoken.ai/v1"
env_key = "JUSTOKEN_API_KEY"
wire_api = "chat"
request_max_retries = 4
stream_idle_timeout_ms = 300000
"#;

        let config = parse_config(content).expect("unknown keys must be ignored");

        assert_eq!(config.provider().expect("provider is defined").wire_api, WireApi::Chat);
    }

    #[test]
    fn defaults_wire_api_to_responses_and_allows_a_missing_env_key() {
        let content = r#"
model = "local-model"
model_provider = "ollama"

[model_providers.ollama]
name = "Ollama"
base_url = "http://localhost:11434/v1"
"#;
        let config = parse_config(content).expect("the config should parse");

        let provider = config.provider().expect("ollama is defined");
        assert_eq!(provider.env_key, None);
        assert_eq!(provider.wire_api, WireApi::Responses);
    }

    #[test]
    fn rejects_an_unknown_wire_api() {
        let content = CODEX_CONFIG.replace("wire_api = \"responses\"", "wire_api = \"grpc\"");

        assert!(parse_config(&content).is_err(), "unknown wire_api must fail");
    }

    #[test]
    fn rejects_missing_required_fields() {
        let missing_model = CODEX_CONFIG.replace("model = \"gpt-5.6-luna\"\n", "");
        let missing_name = CODEX_CONFIG.replace("name = \"JustToken\"\n", "");
        let missing_base_url = CODEX_CONFIG.replace("base_url = \"https://justoken.ai/v1\"\n", "");

        for (label, content) in [
            ("model", missing_model),
            ("name", missing_name),
            ("base_url", missing_base_url),
        ] {
            assert!(parse_config(&content).is_err(), "`{label}` is required");
        }
    }

    #[test]
    fn rejects_a_field_of_the_wrong_type() {
        let content = CODEX_CONFIG.replace("model = \"gpt-5.6-luna\"", "model = 5");

        assert!(parse_config(&content).is_err(), "`model` must be a string");
    }

    #[test]
    fn rejects_malformed_toml() {
        let result = parse_config("model = \"gpt-5.6-luna");

        assert!(result.is_err(), "unterminated string is invalid TOML");
    }

    #[test]
    fn reads_and_parses_a_config_file() {
        let path = unique_temp_path("valid");
        fs::write(&path, CODEX_CONFIG).expect("failed to write fixture");

        let result = read_config_from(&path);

        let _ = fs::remove_file(&path);
        assert_eq!(result.expect("config file should parse").model, "gpt-5.6-luna");
    }

    /// End-to-end check against the real `~/.conway/config.toml` on this machine.
    ///
    /// Ignored by default because it depends on local state. Run it with:
    /// `cargo test parses_the_real_home_config -- --ignored --nocapture`
    #[test]
    #[ignore = "reads the real ~/.conway/config.toml; run with --ignored"]
    fn parses_the_real_home_config() {
        let home = dirs::home_dir().expect("HOME should be available");
        let path = home.join(CONFIG_PATH);
        assert!(path.is_file(), "expected a config file at {}", path.display());

        let config = read_config().expect("the real config file should load");
        let provider = config
            .provider()
            .unwrap_or_else(|err| panic!("{} in {}", err, path.display()));

        assert!(!config.model.is_empty(), "model must not be empty");
        assert!(!provider.name.is_empty(), "provider.name must not be empty");
        assert!(
            provider.base_url.starts_with("http://") || provider.base_url.starts_with("https://"),
            "base_url should be an http(s) URL, got `{}`",
            provider.base_url
        );
        assert!(
            provider.env_key.as_deref().is_some_and(|key| !key.is_empty()),
            "env_key should name a non-empty environment variable"
        );

        eprintln!(
            "loaded {} from {}: model={} provider={} base_url={} env_key={} wire_api={:?}",
            path.display(),
            home.display(),
            config.model,
            provider.name,
            provider.base_url,
            provider.env_key.as_deref().unwrap_or("<none>"),
            provider.wire_api,
        );
    }

    #[test]
    fn reports_a_missing_config_file() {
        let path = unique_temp_path("missing");

        assert!(read_config_from(&path).is_err(), "missing file must surface an error");
    }

    #[test]
    fn config_path_is_relative_to_home() {
        assert_eq!(
            PathBuf::from("/home/example").join(CONFIG_PATH),
            PathBuf::from("/home/example/.conway/config.toml")
        );
    }
}
