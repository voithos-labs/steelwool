use async_openai::Client;
use async_openai::types::{
    ChatCompletionRequestAssistantMessageArgs, ChatCompletionRequestMessage,
    ChatCompletionRequestSystemMessageArgs, ChatCompletionRequestToolMessageArgs,
    ChatCompletionRequestUserMessageArgs, ChatCompletionStreamOptions, ChatCompletionTool,
    CreateChatCompletionRequest, CreateChatCompletionRequestArgs, FinishReason,
};
use futures::StreamExt;
use futures::stream::{self, BoxStream};
use std::sync::Arc;

use crate::{
    ContentType, ContextBuilder, Message, MessageRole, PromptResponse, PromptResponseDelta,
    ProviderAdapter, StopReason, StreamProviderAdapter, ToolCall, ToolDescriptor,
};

pub fn build_chat_completion_message_history(
    context: &ContextBuilder,
    system_message: &str,
) -> Vec<ChatCompletionRequestMessage> {
    let mut msg_vec: Vec<ChatCompletionRequestMessage> = vec![];

    // Push the system message into the msg_vec
    msg_vec.push(
        ChatCompletionRequestSystemMessageArgs::default()
            .content(system_message)
            .build()
            .unwrap()
            .into(),
    );

    for msg in &context.history {
        msg_vec.push(match msg.role {
            MessageRole::User => ChatCompletionRequestUserMessageArgs::default()
                .content(msg.content.to_string())
                .build()
                .unwrap()
                .into(),
            MessageRole::Model => {
                // TODO: This is where you would add tool calls when they happen
                ChatCompletionRequestAssistantMessageArgs::default()
                    .content(msg.content.to_string())
                    .build()
                    .unwrap()
                    .into()
            }
            MessageRole::Function => ChatCompletionRequestToolMessageArgs::default()
                .content(msg.content.to_string())
                .build()
                .unwrap()
                .into(),
            MessageRole::System => ChatCompletionRequestSystemMessageArgs::default()
                .content(msg.content.to_string())
                .build()
                .unwrap()
                .into(),
            MessageRole::Tool => ChatCompletionRequestToolMessageArgs::default()
                .content(msg.content.to_string())
                .build()
                .unwrap()
                .into(),
        });
    }

    return msg_vec;
}

pub fn convert_steelwool_tools_to_openai(tools: Vec<ToolDescriptor>) -> Vec<ChatCompletionTool> {
    return tools
        .iter()
        .map(|td| ChatCompletionTool {
            r#type: async_openai::types::ChatCompletionToolType::Function,
            function: async_openai::types::FunctionObject {
                name: td.name.clone(),
                description: Some(td.description.clone()),
                parameters: Some(td.schema.clone()),
                strict: Some(true),
            },
        })
        .collect();
}

// Non-streaming adapter factory
pub fn openai_adapter_factory(
    model_name: String,
    system_message: String,
    tools: Option<Vec<ToolDescriptor>>,
) -> ProviderAdapter {
    Arc::new(move |context: ContextBuilder, max_tokens : u32| -> std::pin::Pin<Box<dyn Future<Output = Result<PromptResponse, String>> + Send>> {

        let model = model_name.clone();
        let system_msg = system_message.clone();
        let tools_clone = tools.clone();

        Box::pin(async move {

            // Build the openai client
            let openai_client = Client::new();
            
            // Format the message history into the openai lib's one
            let request_msgs = build_chat_completion_message_history(
                &context, &system_msg);

            // Build the request body
            let mut binding = CreateChatCompletionRequestArgs::default();
            let mut request_body = binding
                .max_tokens(max_tokens)
                .model(model)
                .messages(request_msgs);

            // Add tools if provided
            if let Some(tools_vec) = tools_clone {
                request_body = request_body.tools(
                    convert_steelwool_tools_to_openai(tools_vec)
                );
            }

            // Build the rest of the request from the builder
            let request = request_body
                .build()
                .unwrap();

            // Get the response
            let response = openai_client
                .chat()
                .create(request)
                .await
                .unwrap();

            let choice = &response.choices[0];

            return Ok(
                PromptResponse {
                    message: Message { 
                        role: MessageRole::Model, 
                        content: match &choice.message.content {
                            Some(content) => content.clone(),
                            None => "".to_string(),
                        }, 
                        content_type: ContentType::Text 
                    },
                    stop_reason: match choice.finish_reason {
                        Some(reason) => match reason {
                            async_openai::types::FinishReason::Stop => StopReason::Stop,
                            async_openai::types::FinishReason::Length => StopReason::Length,
                            async_openai::types::FinishReason::ToolCalls => StopReason::ToolCalls,
                            async_openai::types::FinishReason::ContentFilter => StopReason::ContentFilter,
                            async_openai::types::FinishReason::FunctionCall => StopReason::ToolCalls,
                        },
                        None => StopReason::Stop,
                    },
                    token_usage: response.usage.unwrap().total_tokens,
                    tool_calls: match choice.message.tool_calls.as_ref() {
                        Some(tool_calls) => Some(
                            tool_calls
                                .iter()                                                    
                                .map(|tc| {
                                    ToolCall {
                                        id: tc.id.clone(),
                                        name: tc.function.name.clone(),
                                        arguments: serde_json::Value::from(tc.function.arguments.clone())
                                    }
                                })
                                .collect()
                        ),
                        None => None,
                    },
                }
            )
        })
    })
}

// Streaming adapter factory
pub fn openai_streaming_adapter_factory(
    model_name: String,
    system_message: String,
    tools: Option<Vec<ToolDescriptor>>,
) -> StreamProviderAdapter {
    Arc::new(move |context: ContextBuilder, max_tokens: u32| {
        let model = model_name.clone();
        let system_msg = system_message.clone();
        let tools_clone = tools.clone();

        // Format the message history into the openai lib's one
        let request_msgs = build_chat_completion_message_history(&context, &system_msg);

        // Build the request body
        let mut binding = CreateChatCompletionRequestArgs::default();
        let mut request_body = binding
            .max_tokens(max_tokens)
            .model(model)
            .messages(request_msgs)
            .stream_options(ChatCompletionStreamOptions {
                include_usage: true,
            });

        // Add tools if provided
        if let Some(tools_vec) = tools_clone {
            request_body = request_body.tools(convert_steelwool_tools_to_openai(tools_vec));
        }

        let request = request_body.build().unwrap();

        let stream = async move {
            let openai_client = Client::new();

            let req_stream = openai_client.chat().create_stream(request).await;

            let mut prepared_tool_call: Option<ToolCall> = None;
            let mut tool_call_buffer: Option<ToolCall> = None;
            let mut arguments_buffer = String::new();

            match req_stream {
                Ok(mut response_stream) => {
                    Box::pin(stream::poll_fn(move |cx| {
                        response_stream.poll_next_unpin(cx).map(|opt| {
                            match opt {
                                Some(Ok(response)) => {

                                    if response.choices.is_empty() {
                                        return response.usage.map_or(None, |usage| {
                                            Some(Ok(PromptResponseDelta {
                                                content: "".to_string(),
                                                stop_reason: Some(StopReason::Stop),
                                                tool_calls: None,
                                                cumulative_tokens: usage.completion_tokens,
                                            }))
                                        });
                                    }

                                    let first_choice = &response.choices[0];

                                    // Handling the concatenation of tool calls & their args
                                    if let Some(tool_chunks) = &first_choice.delta.tool_calls {
                                        for chunk in tool_chunks {

                                            let function = chunk.function.as_ref().unwrap();

                                            // If a tool call is already in the building buffer but a NEW tool call name
                                            // pops up, assume that there was another tool call and swap the old buffer to
                                            // "prepared" to be sent off next round
                                            if let Some(tool_call) = &tool_call_buffer {
                                                if function.name.is_some() {

                                                    // Move the buffers to "prepared"
                                                    prepared_tool_call = Some(ToolCall { 
                                                        arguments: serde_json::Value::from(arguments_buffer.to_string()), ..tool_call.clone() 
                                                    });

                                                    // Reset the buffers
                                                    tool_call_buffer = None;
                                                    arguments_buffer = "".to_string();
                                                }
                                            }

                                            // If there is no tool call in the buffer then start a new tool call
                                            // and start a running 
                                            if tool_call_buffer.is_none() {
                                                tool_call_buffer = Some(
                                                    ToolCall { 
                                                        id: chunk.id.clone().unwrap(), 
                                                        name: function.name.as_ref().unwrap().to_string(), 
                                                        arguments: serde_json::Value::Null
                                                    });
                                                continue;
                                            }

                                            // Push additional arguments onto the arguments buffer
                                            let tmp_args_content = arguments_buffer.clone();
                                            arguments_buffer = tmp_args_content + &function.arguments.as_ref().cloned().unwrap_or_else(|| "".to_string());

                                        }
                                    }

                                    if let Some(FinishReason::ToolCalls) = first_choice.finish_reason {
                                        if let Some(tool_call) = &tool_call_buffer {
                                            // Move the buffers to "prepared"
                                            prepared_tool_call = Some(ToolCall { 
                                                arguments: serde_json::Value::from(arguments_buffer.to_string()), ..tool_call.clone() 
                                            });
    
                                            // Reset the buffers
                                            tool_call_buffer = None;
                                            arguments_buffer = "".to_string();
                                        }
                                    }

                                    return Some(Ok(PromptResponseDelta {
                                        content: first_choice.delta.content.clone().unwrap_or_default(),
                                        stop_reason: match first_choice.finish_reason {
                                            Some(reason) => match reason {
                                                async_openai::types::FinishReason::Stop => Some(StopReason::Stop),
                                                async_openai::types::FinishReason::Length => Some(StopReason::Length),
                                                async_openai::types::FinishReason::ToolCalls => Some(StopReason::ToolCalls),
                                                async_openai::types::FinishReason::ContentFilter => Some(StopReason::ContentFilter),
                                                async_openai::types::FinishReason::FunctionCall => Some(StopReason::ToolCalls),
                                            },
                                            None => None,
                                        },

                                        tool_calls: match prepared_tool_call.take() { // take tool calls if they're prepared and return them
                                            Some(tool_call) => Some(vec![tool_call]),
                                            None => None,
                                        },

                                        // async-openai only ships tokens on the final delta with an empty response (fml)
                                        cumulative_tokens: 0,
                                    }));
                                    
                                },
                                Some(Err(e)) => {
                                    Some(Err(format!("OpenAI streaming error: {:?}", e)))
                                }
                                None => None,
                            }
                        })
                    }))
                        as BoxStream<'static, Result<PromptResponseDelta, String>>
                }
                Err(e) => Box::pin(stream::once(async move {
                    Err(format!("Failed to start OpenAI stream: {:?}", e))
                }))
                    as BoxStream<'static, Result<PromptResponseDelta, String>>,
            }
        };

        return Box::pin(stream::once(stream).flatten())
            as BoxStream<'static, Result<PromptResponseDelta, String>>;
    })
}
