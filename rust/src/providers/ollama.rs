use futures::stream::{self, BoxStream, StreamExt};
use ollama_rs::Ollama;
use ollama_rs::generation::completion::request::GenerationRequest;
use serde_json::json;
use std::sync::Arc;

use crate::{
    ContentType, ContextBuilder, Message, MessageRole, PromptResponse, PromptResponseDelta,
    ProviderAdapter, StopReason, StreamProviderAdapter, ToolDescriptor,
};

/// Format a prompt for Ollama using standard JSON format
pub fn format_ollama_prompt(
    context: &ContextBuilder,
    tools: &Option<Vec<ToolDescriptor>>,
) -> String {
    // Create a JSON object with the messages from context history
    let messages = context
        .history
        .iter()
        .map(|msg| {
            let role = match msg.role {
                MessageRole::User => "user",
                MessageRole::Model => "assistant",
                MessageRole::System => "system",
                MessageRole::Function | MessageRole::Tool => "tool",
            };

            json!({
                "role": role,
                "content": msg.content
            })
        })
        .collect::<Vec<_>>();

    // Add tools if available
    let mut request = json!({
        "messages": messages
    });

    if let Some(tools_list) = tools {
        request["tools"] = json!(tools_list);
    }

    // Return the JSON as a string
    serde_json::to_string(&request).unwrap_or_default()
}

// Non-streaming adapter factory
pub fn ollama_adapter_factory(
    model_name: String,
    tools: Option<Vec<ToolDescriptor>>,
) -> ProviderAdapter {
    Arc::new(move |context: ContextBuilder, _: u32| {
        let model = model_name.clone();
        let tools_clone = tools.clone();

        Box::pin(async move {
            let ollama = Ollama::default();

            // Use the generalized formatting function
            let prompt = format_ollama_prompt(&context, &tools_clone);

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
    })
}

// Streaming adapter factory
pub fn ollama_streaming_adapter_factory(
    model_name: String,
    tools: Option<Vec<ToolDescriptor>>,
) -> StreamProviderAdapter {
    Arc::new(move |context: ContextBuilder, _: u32| {
        let model = model_name.clone();
        let tools_clone = tools.clone();

        // Use the generalized formatting function
        let prompt = format_ollama_prompt(&context, &tools_clone);

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
                                        tool_calls: None, // *it does support this; todo
                                        cumulative_tokens: 0,
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
    })
}
