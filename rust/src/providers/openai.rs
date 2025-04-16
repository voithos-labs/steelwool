use async_openai::Client;
use async_openai::types::{
    ChatCompletionRequestAssistantMessageArgs, ChatCompletionRequestMessage,
    ChatCompletionRequestSystemMessageArgs, ChatCompletionRequestToolMessageArgs,
    ChatCompletionRequestUserMessageArgs, ChatCompletionStreamOptions, CreateChatCompletionRequest,
    CreateChatCompletionRequestArgs, FinishReason,
};
use futures::StreamExt;
use futures::stream::{self, BoxStream};
use std::sync::Arc;

use crate::{
    ContentType, ContextBuilder, Message, MessageRole, PromptResponse, PromptResponseDelta,
    ProviderAdapter, StopReason, StreamProviderAdapter, ToolDescriptor,
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

// Non-streaming adapter factory
pub fn openai_adapter_factory(
    model_name: String,
    system_message: String,
    tools: Option<Vec<ToolDescriptor>>,
) -> ProviderAdapter {
    Arc::new(move |context: ContextBuilder, max_tokens : u32| -> std::pin::Pin<Box<dyn Future<Output = Result<PromptResponse, String>> + Send>> {

        let model = model_name.clone();
        let system_message = system_message.clone();
        let tools_clone = tools.clone();

        Box::pin(async move {

            // Build the openai client
            let openai_client = Client::new();
            
            // Format the message history into the openai lib's one
            let request_msgs = build_chat_completion_message_history(
                &context, &system_message);

            // Build the request body
            let request_body = CreateChatCompletionRequestArgs::default()
                .max_tokens(max_tokens)
                .model(model)
                .messages(request_msgs)
                .build()
                .unwrap();

            let response = openai_client
                .chat()
                .create(request_body)
                .await
                .unwrap();

            let choice = &response.choices[0];

            return Ok(
                PromptResponse {
                    message: Message { 
                        role: MessageRole::Model, 
                        content: String::from(choice.message.content.as_ref().unwrap()), 
                        content_type: ContentType::Text 
                    },
                    stop_reason: StopReason::Stop,
                    token_usage: response.usage.unwrap().total_tokens,
                    tool_calls: None,
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
        let request_msgs = build_chat_completion_message_history(&context, &system_message);

        // Build the request body
        let request_body = CreateChatCompletionRequestArgs::default()
            .max_tokens(max_tokens)
            .model(model)
            .messages(request_msgs)
            .stream_options(ChatCompletionStreamOptions {
                include_usage: true,
            })
            .build()
            .unwrap();

        let stream = async move {
            let openai_client = Client::new();

            let req_stream = openai_client.chat().create_stream(request_body).await;

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
                                                tool_call: None,
                                                cumulative_tokens: usage.completion_tokens,
                                            }))
                                        });
                                    }

                                    let first_choice = &response.choices[0];

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

                                        // todo
                                        tool_call: None,

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
