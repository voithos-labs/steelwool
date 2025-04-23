#[cfg(all(test, feature = "tokio-runtime"))]
mod tests {
    use futures::StreamExt;
    use std::sync::{Arc, Mutex};
    use tokio;

    #[cfg(feature = "ollama")]
    use steelwool::providers::ollama::{ollama_adapter_factory, ollama_streaming_adapter_factory};
    #[cfg(feature = "ollama")]
    use steelwool::{ContentType, ContextBuilder, Message, MessageRole};

    #[tokio::test]
    #[cfg(feature = "ollama")]
    async fn test_ollama_integration() {
        let model_name = "llama3.2".to_string();
        let system_message =
            "You are a helpful, concise assistant. Keep your answers brief.".to_string();
        let adapter = ollama_adapter_factory(model_name, None);

        let context = ContextBuilder { history: vec![] }
            .add_message(Message {
                role: MessageRole::System,
                content: system_message,
                content_type: ContentType::Text,
            })
            .add_message(Message {
                role: MessageRole::User,
                content: "Explain quantum computing in 3 simple sentences.".to_string(),
                content_type: ContentType::Text,
            });

        let response = context.send(adapter, 1000).await.resolve_without();

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

    #[tokio::test]
    #[cfg(feature = "ollama")]
    async fn test_ollama_streaming_integration() {
        let model_name = "llama3.2".to_string();
        let system_message =
            "You are a helpful, concise assistant. Keep your answers brief.".to_string();
        let streaming_adapter = ollama_streaming_adapter_factory(model_name.clone(), None);

        let context = ContextBuilder { history: vec![] }
            .add_message(Message {
                role: MessageRole::System,
                content: system_message.clone(),
                content_type: ContentType::Text,
            })
            .add_message(Message {
                role: MessageRole::User,
                content: "Explain quantum computing in 3 simple sentences.".to_string(),
                content_type: ContentType::Text,
            });

        // Test 1: Basic streaming with DIRECT stream consumption
        println!("Testing direct stream consumption:");
        let mut stream = context
            .clone()
            .send_streaming(streaming_adapter.clone(), 1000);

        // Collect and display streamed content for debug
        let mut streamed_content = String::new();
        while let Some(result) = stream.next().await {
            match result {
                Ok(delta) => {
                    print!("{}", delta.content);
                    streamed_content.push_str(&delta.content);

                    if delta.stop_reason.is_some() {
                        println!("\n[Stream completed]");
                    }
                }
                Err(e) => {
                    println!("\nError: {}", e);
                    assert!(false, "Streaming error: {}", e);
                }
            }
        }

        // Verify we got smth
        assert!(
            !streamed_content.is_empty(),
            "Stream should produce content"
        );

        // Test 2: Streaming with callback
        println!("\nTesting streaming with callback:");
        let callback_counter = Arc::new(Mutex::new(0));
        let callback_content = Arc::new(Mutex::new(String::new()));

        let counter_clone = callback_counter.clone();
        let content_clone = callback_content.clone();

        let _result = context
            .clone()
            .send_streaming_with_callback(streaming_adapter.clone(), 1000, move |result| {
                if let Ok(delta) = result {
                    print!("{}", delta.content);

                    let mut counter = counter_clone.lock().unwrap();
                    *counter += 1;

                    let mut content = content_clone.lock().unwrap();
                    content.push_str(&delta.content);

                    if delta.stop_reason.is_some() {
                        println!("\n[Callback stream completed]");
                    }
                }
            })
            .await;

        // Verify callback was called multiple times (streaming should produce multiple chunks)
        let final_count = *callback_counter.lock().unwrap();
        let final_content = callback_content.lock().unwrap().clone();

        assert!(
            final_count > 1,
            "Callback should be called multiple times, got {}",
            final_count
        );
        assert!(!final_content.is_empty(), "Callback should collect content");
        println!("Callback was called {} times", final_count);
    }
}
