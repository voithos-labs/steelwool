#[cfg(all(test, feature = "tokio-runtime"))]
mod tests {
    use futures::StreamExt;
    use std::sync::{Arc, Mutex};
    use tokio;

    #[cfg(feature = "openai")]
    use steelwool::providers::openai::{openai_adapter_factory, openai_streaming_adapter_factory};
    #[cfg(feature = "openai")]
    use steelwool::{ContentType, ContextBuilder, Message, MessageRole};

    #[tokio::test]
    #[cfg(feature = "openai")]
    async fn test_openai_integration() {
        let model_name = "gpt-3.5-turbo".to_string();
        let system_message =
            "You are a helpful, concise assistant. Keep your answers brief.".to_string();
        let adapter = openai_adapter_factory(model_name, system_message, None);

        let context = ContextBuilder { history: vec![] }.add_message(Message {
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
    #[cfg(feature = "openai")]
    async fn test_openai_streaming_integration() {
        let model_name = "gpt-3.5-turbo".to_string();
        let system_message =
            "You are a helpful, concise assistant. Keep your answers brief.".to_string();
        let streaming_adapter =
            openai_streaming_adapter_factory(model_name.clone(), system_message.clone(), None);

        let context = ContextBuilder { history: vec![] }.add_message(Message {
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
        let mut saw_tokens = false; // Track if we ever see tokens

        while let Some(result) = stream.next().await {
            match result {
                Ok(delta) => {
                    print!("{}", delta.content);
                    streamed_content.push_str(&delta.content);

                    if delta.cumulative_tokens > 0 {
                        saw_tokens = true;
                        println!("\n[Tokens used so far: {}]", delta.cumulative_tokens);
                    }

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

        // Verify we got something and saw tokens
        assert!(
            !streamed_content.is_empty(),
            "Stream should produce content"
        );
        assert!(saw_tokens, "Stream should report token usage at least once");

        // Test 2: Streaming with callback
        println!("\nTesting streaming with callback:");
        let callback_counter = Arc::new(Mutex::new(0));
        let callback_content = Arc::new(Mutex::new(String::new()));
        let saw_tokens = Arc::new(Mutex::new(false)); // Track tokens in callback

        let counter_clone = callback_counter.clone();
        let content_clone = callback_content.clone();
        let tokens_clone = saw_tokens.clone();

        let _result = context
            .clone()
            .send_streaming_with_callback(streaming_adapter.clone(), 1000, move |result| {
                if let Ok(delta) = result {
                    print!("{}", delta.content);

                    let mut counter = counter_clone.lock().unwrap();
                    *counter += 1;

                    let mut content = content_clone.lock().unwrap();
                    content.push_str(&delta.content);

                    if delta.cumulative_tokens > 0 {
                        let mut tokens = tokens_clone.lock().unwrap();
                        *tokens = true;
                        println!("\n[Tokens used so far: {}]", delta.cumulative_tokens);
                    }

                    if delta.stop_reason.is_some() {
                        println!("\n[Callback stream completed]");
                    }
                }
            })
            .await;

        // Verify callback was called multiple times and reported tokens
        let final_count = *callback_counter.lock().unwrap();
        let final_content = callback_content.lock().unwrap().clone();
        let final_saw_tokens = *saw_tokens.lock().unwrap();

        assert!(
            final_count > 1,
            "Callback should be called multiple times, got {}",
            final_count
        );
        assert!(!final_content.is_empty(), "Callback should collect content");
        assert!(
            final_saw_tokens,
            "Callback should report token usage at least once"
        );
        println!("Callback was called {} times", final_count);
    }

    #[tokio::test]
    #[cfg(feature = "openai")]
    async fn test_openai_tool_calling() {
        let model_name = "gpt-3.5-turbo".to_string();
        let system_message = "You are a helpful assistant. When asked about the weather, use the get_weather function.".to_string();

        // Define a mock weather tool
        let weather_tool = steelwool::ToolDescriptor {
            name: "get_weather".to_string(),
            description: "Get the current weather for a location".to_string(),
            schema: serde_json::json!({
                "type": "object",
                "properties": {
                    "location": {
                        "type": "string",
                        "description": "The city and state, e.g., 'San Francisco, CA'"
                    }
                },
                "required": ["location"],
                "additionalProperties": false
            }),
            required: true,
        };

        let tools = Some(vec![weather_tool]);
        let adapter = openai_adapter_factory(model_name, system_message, tools);

        let context = ContextBuilder { history: vec![] }.add_message(Message {
            role: MessageRole::User,
            content: "What's the weather like in Seattle?".to_string(),
            content_type: ContentType::Text,
        });

        let response = context.send(adapter, 1000).await;

        // assert!(matches!(response.prompt_response.stop_reason, StopReason::ToolCalls), "Expected to stop for a tool call");
        assert!(
            response.prompt_response.tool_calls.is_some(),
            "Expected to have some tool calls"
        );

        println!("Tool calling test completed successfully");

        let resolved_response = response.resolve_without();

        for message in resolved_response.history {
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
    #[cfg(feature = "openai")]
    async fn test_openai_tool_calling_streaming() {
        println!("\nTesting streaming tool calling integration:");

        let model_name = "gpt-3.5-turbo".to_string();
        let system_message = 
            "You are a helpful assistant. When asked about the weather, use the get_weather function.".to_string();

        // Define a mock weather tool
        let weather_tool = steelwool::ToolDescriptor {
            name: "get_weather".to_string(),
            description: "Get the current weather for a location".to_string(),
            schema: serde_json::json!({
                "type": "object",
                "properties": {
                    "location": {
                        "type": "string",
                        "description": "The city and state, e.g., 'San Francisco, CA'"
                    }
                },
                "required": ["location"],
                "additionalProperties": false
            }),
            required: true,
        };

        let tools = Some(vec![weather_tool]);
        let streaming_adapter =
            openai_streaming_adapter_factory(model_name.clone(), system_message.clone(), tools);

        let context = ContextBuilder { history: vec![] }.add_message(Message {
            role: MessageRole::User,
            content: "What's the weather like in Seattle?".to_string(),
            content_type: ContentType::Text,
        });

        // Test 1: Basic streaming with DIRECT stream consumption
        println!("Testing direct stream consumption:");
        let mut stream = context
            .clone()
            .send_streaming(streaming_adapter.clone(), 1000);

        let mut saw_tokens = false;
        let mut saw_tool_call = false;

        while let Some(result) = stream.next().await {
            match result {
                Ok(delta) => {
                    if delta.cumulative_tokens > 0 {
                        saw_tokens = true;
                        println!("\n[Tokens used so far: {}]", delta.cumulative_tokens);
                    }

                    if let Some(tool_calls) = delta.tool_calls {
                        saw_tool_call = true;
                        println!("\n[Tool Call detected]");
                        for tool_call in tool_calls {
                            println!("Tool Name: {}", tool_call.name);
                            println!("Arguments: {}", tool_call.arguments);
                        }
                    }

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

        // Verify tokens and tool calls only
        assert!(saw_tokens, "Stream should report token usage at least once");
        assert!(
            saw_tool_call,
            "Stream should include at least one tool call"
        );

        // Test 2: Streaming with callback
        println!("\nTesting streaming with callback:");
        let callback_counter = Arc::new(Mutex::new(0));
        let saw_tokens = Arc::new(Mutex::new(false));
        let saw_tool_call = Arc::new(Mutex::new(false));

        let counter_clone = callback_counter.clone();
        let tokens_clone = saw_tokens.clone();
        let tool_clone = saw_tool_call.clone();

        let _result = context
            .clone()
            .send_streaming_with_callback(streaming_adapter.clone(), 1000, move |result| {
                if let Ok(delta) = result {
                    let mut counter = counter_clone.lock().unwrap();
                    *counter += 1;

                    if delta.cumulative_tokens > 0 {
                        let mut tokens = tokens_clone.lock().unwrap();
                        *tokens = true;
                        println!("\n[Tokens used so far: {}]", delta.cumulative_tokens);
                    }

                    if let Some(tool_calls) = delta.tool_calls {
                        let mut tool_flag = tool_clone.lock().unwrap();
                        *tool_flag = true;
                        println!("\n[Tool Call in callback]");
                        for tool_call in tool_calls {
                            println!("Tool Name: {}", tool_call.name);
                            println!("Arguments: {}", tool_call.arguments);
                        }
                    }

                    if delta.stop_reason.is_some() {
                        println!("\n[Callback stream completed]");
                    }
                }
            })
            .await;

        // Verify callback behavior without content checks
        let final_count = *callback_counter.lock().unwrap();
        let final_saw_tokens = *saw_tokens.lock().unwrap();
        let final_saw_tool_call = *saw_tool_call.lock().unwrap();

        assert!(
            final_count > 1,
            "Callback should be called multiple times, got {}",
            final_count
        );
        assert!(final_saw_tokens, "Callback should report token usage");
        assert!(final_saw_tool_call, "Callback should report tool calls");
        println!("Callback was called {} times", final_count);
    }
}
