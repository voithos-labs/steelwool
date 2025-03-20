#[cfg(all(test, feature = "tokio-runtime"))]
mod tests {
    use super::*;
    use tokio;

    #[cfg(feature = "ollama")]
    use steelwool::providers::ollama::create_ollama_adapter;
    #[cfg(feature = "ollama")]
    use steelwool::{ContentType, ContextBuilder, Message, MessageRole};

    #[tokio::test]
    #[cfg(feature = "ollama")]
    async fn test_ollama_integration() {
        let model_name = "llama3.2".to_string();
        let adapter = create_ollama_adapter(model_name);

        let context = ContextBuilder { history: vec![] }.add_message(Message {
            role: MessageRole::User,
            content: "Explain quantum computing in 3 simple sentences.".to_string(),
            content_type: ContentType::Text,
        });

        let system_message =
            "You are a helpful, concise assistant. Keep your answers brief.".to_string();

        let response = context
            .send(adapter, system_message, 1000, None)
            .await
            .resolve_without()
            .await;

        assert!(!response.history.is_empty());
        for message in response.history {
            let role = match message.role {
                MessageRole::User => "User",
                MessageRole::Model => "Assistant",
                MessageRole::System => "System",
                MessageRole::Function => "Function",
                MessageRole::Tool => "Tool",
            };
            println!("{}: {}", role, message.content);
        }
    }
}
