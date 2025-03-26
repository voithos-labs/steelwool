use futures::stream::{self, BoxStream, StreamExt};
use ollama_rs::Ollama;
use ollama_rs::generation::completion::request::GenerationRequest;
use std::sync::Arc;

use crate::{
    ContentType, ContextBuilder, Message, MessageRole, PromptResponse, PromptResponseDelta,
    ProviderAdapter, StopReason, StreamProviderAdapter, Tool, ToolCall,
};

/// Format a prompt for Ollama in the OpenAI-compatible format
/// TODO:
/// - should be default, allowing users to pass in their own formatter -> String
pub fn format_ollama_prompt(
    context: &ContextBuilder,
    system_message: &str,
    tools: &Option<Vec<Tool>>,
) -> String {
    let mut prompt = String::new();

    // Add system message if provided
    if !system_message.is_empty() {
        prompt.push_str(&format!("system\n{}\n\n", system_message));
    }

    // Include tool definitions if available
    if let Some(tools) = tools {
        prompt.push_str("Available tools:\n");
        for tool in tools {
            let schema_str =
                serde_json::to_string_pretty(&tool.schema).unwrap_or_else(|_| "{}".to_string());
            prompt.push_str(&format!(
                "Tool name: {}\nDescription: {}\nParameters: {}\nRequired: {}\n\n",
                tool.name, tool.description, schema_str, tool.required
            ));
        }
    }

    // Add conversation history
    for msg in &context.history {
        let role = match msg.role {
            MessageRole::User => "user",
            MessageRole::Model => "assistant",
            MessageRole::System => "system",
            MessageRole::Function | MessageRole::Tool => "function",
        };
        prompt.push_str(&format!("{role}\n{}\n\n", msg.content));
    }

    // Add final role marker for assistant's response
    prompt.push_str("assistant\n");

    prompt
}

// Non-streaming adapter factory
pub fn ollama_adapter_factory(model_name: String) -> ProviderAdapter {
    Arc::new(
        move |context: ContextBuilder,
              system_message: String,
              max_tokens: u32,
              tools: Option<Vec<Tool>>| {
            let model = model_name.clone();

            Box::pin(async move {
                let ollama = Ollama::default();

                // Use the generalized formatting function
                let prompt = format_ollama_prompt(&context, &system_message, &tools);

                let request = GenerationRequest::new(model, prompt);
                let result = ollama.generate(request).await;

                match result {
                    Ok(generation) => Ok(PromptResponse {
                        message: Message {
                            role: MessageRole::Model,
                            content: generation.response,
                            content_type: ContentType::Text,
                        },
                        stop_reason: StopReason::Stop,
                        token_usage: 0,
                        tool_calls: None,
                    }),
                    Err(e) => Err(format!("Ollama generation error: {:?}", e)),
                }
            })
        },
    )
}

// Streaming adapter factory
pub fn ollama_streaming_adapter_factory(model_name: String) -> StreamProviderAdapter {
    Arc::new(
        move |context: ContextBuilder,
              system_message: String,
              _max_tokens: u32,
              tools: Option<Vec<Tool>>| {
            let model = model_name.clone();

            // Use the generalized formatting function
            let prompt = format_ollama_prompt(&context, &system_message, &tools);

            // Create a boxed stream that will contain our PromptResponseDelta items
            let stream = async move {
                let ollama = Ollama::default();
                let request = GenerationRequest::new(model, prompt);

                match ollama.generate_stream(request).await {
                    Ok(mut response_stream) => {
                        // Map the Ollama response stream to our PromptResponseDelta stream
                        Box::pin(stream::poll_fn(move |cx| {
                            response_stream.poll_next_unpin(cx).map(|opt| {
                                match opt {
                                    Some(Ok(responses)) => {
                                        // Process all responses in the batch
                                        if responses.is_empty() {
                                            return None;
                                        }

                                        // Get the last response to check if it's done
                                        let last_response = responses.last().unwrap();
                                        let is_done = last_response.done;

                                        // Combine all response texts from this batch
                                        let content = responses
                                            .iter()
                                            .map(|r| r.response.clone())
                                            .collect::<Vec<String>>()
                                            .join("");

                                        // Create a delta with the content and stop reason if done
                                        let delta = PromptResponseDelta {
                                            content,
                                            stop_reason: if is_done {
                                                Some(StopReason::Stop)
                                            } else {
                                                None
                                            },
                                            tool_call: None, // Ollama doesn't support tool calls in streaming mode yet
                                        };

                                        Some(Ok(delta))
                                    }
                                    Some(Err(e)) => {
                                        Some(Err(format!("Ollama streaming error: {:?}", e)))
                                    }
                                    None => None,
                                }
                            })
                        }))
                            as BoxStream<'static, Result<PromptResponseDelta, String>>
                    }
                    Err(e) => {
                        // Return a stream with a single error if we cant start streaming (String)
                        Box::pin(stream::once(async move {
                            Err(format!("Failed to start Ollama stream: {:?}", e))
                        }))
                            as BoxStream<'static, Result<PromptResponseDelta, String>>
                    }
                }
            };

            // Return a boxed stream that will resolve to our real stream
            Box::pin(stream::once(stream).flatten())
                as BoxStream<'static, Result<PromptResponseDelta, String>>
        },
    )
}
