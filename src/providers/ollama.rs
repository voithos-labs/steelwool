use ollama_rs::Ollama;
use ollama_rs::generation::completion::request::GenerationRequest;
use ollama_rs::generation::options::GenerationOptions;

use crate::{
    ContentType, ContextBuilder, Message, MessageRole, PromptResponse, ProviderAdapter, StopReason,
    Tool,
};
use std::sync::Arc;

pub fn ollama_adapter_factory(model_name: String) -> ProviderAdapter {
    Arc::new(
        move |context: ContextBuilder,
              system_message: String,
              _max_tokens: u32,
              _tools: Option<Vec<Tool>>| {
            let model = model_name.clone();
            let history = context.history.clone();

            Box::pin(async move {
                let ollama = Ollama::default();

                let mut prompt = String::new();
                if !system_message.is_empty() {
                    prompt.push_str(&format!("System: {}\n\n", system_message));
                }

                for msg in history {
                    let role_prefix = match msg.role {
                        MessageRole::User => "User",
                        MessageRole::Model => "Assistant",
                        MessageRole::System => "System",
                        MessageRole::Function => "Function",
                        MessageRole::Tool => "Tool",
                    };
                    prompt.push_str(&format!("{}: {}\n\n", role_prefix, msg.content));
                }
                prompt.push_str("Assistant: ");

                let request = GenerationRequest::new(model, prompt.clone())
                    .options(GenerationOptions::default());

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
                        tool_calls: None, // todo
                    }),
                    Err(e) => Err(format!("Ollama generation error: {:?}", e)),
                }
            })
        },
    )
}
