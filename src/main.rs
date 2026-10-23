pub mod agent;
pub mod llm;
pub mod prompt;
pub mod provider_config;
pub mod repl;
pub mod tools;

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    repl::run().await
}
