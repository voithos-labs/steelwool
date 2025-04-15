use std::env;
use std::sync::Arc;
use openai_api_rs::v1::api::OpenAIClient;
use openai_api_rs::v1::chat_completion::{self, ChatCompletionMessage};

use crate::{
    ContentType, ContextBuilder, Message, MessageRole, PromptResponse, PromptResponseDelta,
    ProviderAdapter, StopReason, StreamProviderAdapter, ToolDescriptor,
};

pub fn format_ctx_to_openai_history(
    context: &ContextBuilder,
    system_message: &str,
) -> Vec<ChatCompletionMessage> {

    // Create a vec to hold the openai chat completion messages
    let mut msg_vec: Vec<ChatCompletionMessage> = Vec::new();

    // Push the system message first
    msg_vec.push(
        ChatCompletionMessage { 
            role: chat_completion::MessageRole::system, 
            content: chat_completion::Content::Text(String::from(system_message)), 
            name: None, 
            tool_calls: None, 
            tool_call_id: None 
        }
    );

    // Format all of the messages
    for msg in &context.history {
        
        // Convert the role
        let role = match msg.role {
            MessageRole::User => chat_completion::MessageRole::user,
            MessageRole::Model => chat_completion::MessageRole::assistant,
            MessageRole::Function => chat_completion::MessageRole::function,
            MessageRole::System => chat_completion::MessageRole::system,
            MessageRole::Tool => chat_completion::MessageRole::tool,
        };

        // TODO: Converting the tool call stuff also needs to be done in here

        // Push the converted chat completion message into the vec
        msg_vec.push(
            ChatCompletionMessage {
                role: role,
                content: chat_completion::Content::Text(msg.content.to_string()),
                name: None,
                tool_calls: None,
                tool_call_id: None 
            }
        );
    }

    // Finally, return the message vector 
    return msg_vec;
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
            
            let api_key = env::var("OPENAI_API_KEY")
                .expect("Failed to fetch OPENAI_API_KEY from env");

            // Build the openai client
            let mut openai_client = OpenAIClient::builder()
                .with_api_key(api_key)
                .build()
                .expect("Failed to build OpenAI client");
            
            // Format the message history into the openai lib's one
            let message_history = format_ctx_to_openai_history(
                &context, &system_message);

            // Build the request struct
            // TODO: This is where tools are passed via `.tools`, needs to be done
            let request = chat_completion::ChatCompletionRequest::new(
                model, message_history)
                .max_tokens(max_tokens as i64);

            let response = openai_client
                .chat_completion(request)
                .await
                .expect("Failed to get response from OpenAI");

            let choice = &response.choices[0];

            return Ok(
                PromptResponse {
                    message: Message { 
                        role: MessageRole::Model, 
                        content: String::from(choice.message.content.as_ref().unwrap()), 
                        content_type: ContentType::Text 
                    },
                    stop_reason: StopReason::Stop,
                    token_usage: response.usage.total_tokens as u32,
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