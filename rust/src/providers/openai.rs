use std::{default, env};
use std::sync::Arc;
use async_openai::types::{
    ChatCompletionRequestAssistantMessageArgs, ChatCompletionRequestMessage, ChatCompletionRequestSystemMessageArgs, ChatCompletionRequestToolMessageArgs, ChatCompletionRequestUserMessageArgs, CreateChatCompletionRequest, CreateChatCompletionRequestArgs
};
use async_openai::Client;

use crate::{
    ContentType, ContextBuilder, Message, MessageRole, PromptResponse, PromptResponseDelta,
    ProviderAdapter, StopReason, StreamProviderAdapter, ToolDescriptor,
};

pub fn build_chat_completion_request(
    context: &ContextBuilder,
    system_message: &str,
    max_tokens : u32,
    model : &String
) -> CreateChatCompletionRequest {

    let mut msg_vec : Vec<ChatCompletionRequestMessage> = vec![];

    // Push the system message into the msg_vec
    msg_vec.push(
        ChatCompletionRequestSystemMessageArgs::default()
            .content(system_message)
            .build()
            .unwrap()
            .into()
    );

    for msg in &context.history {

        msg_vec.push(
            match msg.role {
                MessageRole::User => {
                    ChatCompletionRequestUserMessageArgs::default()
                        .content(msg.content.to_string())
                        .build()
                        .unwrap()
                        .into()
                },
                MessageRole::Model => {
                    // TODO: This is where you would add tool calls when they happen
                    ChatCompletionRequestAssistantMessageArgs::default()
                        .content(msg.content.to_string())
                        .build()
                        .unwrap()
                        .into()
                },
                MessageRole::Function => {
                    ChatCompletionRequestToolMessageArgs::default()
                        .content(msg.content.to_string())
                        .build()
                        .unwrap()
                        .into()
                },
                MessageRole::System => {
                    ChatCompletionRequestSystemMessageArgs::default()
                        .content(msg.content.to_string())
                        .build()
                        .unwrap()
                        .into()
                },
                MessageRole::Tool => {
                    ChatCompletionRequestToolMessageArgs::default()
                        .content(msg.content.to_string())
                        .build()
                        .unwrap()
                        .into()
                },
            }
        );

    }

    // Finally, return the request 
    return CreateChatCompletionRequestArgs::default()
        .max_tokens(max_tokens)
        .model(model)
        .messages(msg_vec)
        .build()
        .unwrap();
}

// Non-streaming adapter factory
pub fn openai_adapter_factory(
    model_name : String,
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
            let request_body = build_chat_completion_request(
                &context, &system_message, max_tokens, &model);


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
    
) -> StreamProviderAdapter {
    todo!()
}