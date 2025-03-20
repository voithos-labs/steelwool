/// Authors:
/// Alex & Finn
///
/// steelwool is a lightweight library for interacting with LLMs
/* -------------------------------------------------------------------------- */
/*                                  STEELWOOL                                 */
/* -------------------------------------------------------------------------- */

/* ------------------------------ Dependencies ------------------------------ */
use std::collections::HashMap;
use std::future::Future;
use std::pin::Pin;
use std::sync::Arc;

// use futures::StreamExt;
use futures::stream::BoxStream;

/* -------------------------------- Features -------------------------------- */

pub mod providers {
    #[cfg(feature = "ollama")]
    pub mod ollama;
}

/* ------------------------------- Signatures ------------------------------- */

pub type PromptFuture = Pin<Box<dyn Future<Output = Result<PromptResponse, String>> + Send>>;
pub type ProviderAdapter =
    Arc<dyn Fn(ContextBuilder, String, u32, Option<Vec<Tool>>) -> PromptFuture + Send + Sync>;
// for streaming prob want to use a SteamingDelta or smth to represent the individual steaming chunks instead of PromptResponse
pub type StreamProviderAdapter = Arc<
    dyn Fn(
            ContextBuilder,
            String,
            u32,
            Option<Vec<Tool>>,
        ) -> BoxStream<'static, Result<PromptResponse, String>>
        + Send
        + Sync,
>;

pub type ToolResolver = Arc<dyn Fn(ToolCall) -> String + Send + Sync>;

/* ------------------------------ Data Structs ------------------------------ */

// Message

#[derive(Clone)]
pub struct Message {
    pub role: MessageRole,
    pub content: String,
    pub content_type: ContentType,
}

// Responses

pub struct PromptResponse {
    pub message: Message,
    pub stop_reason: StopReason,
    pub token_usage: u32,
    pub tool_calls: Option<Vec<ToolCall>>,
}

// Tools

#[derive(Clone)]
pub struct Tool {
    pub name: String,
    pub description: String,
    pub parameters: HashMap<String, Parameter>,
}

#[derive(Clone)]
pub struct ToolCall {
    pub id: String,
    pub name: String,
    pub arguments: HashMap<String, ArgumentValue>,
}

// #[derive(Clone, Serialize, Deserialize)]
// pub struct ToolResult {
//     pub tool_call_id: String,
//     pub content: String,
// }

#[derive(Clone)]
pub struct Parameter {
    pub return_type: String,
    // ... todo
}

/* ---------------------------------- Enums --------------------------------- */

#[derive(Clone)]
pub enum ArgumentValue {
    String(String),
    Integer(i32),
    Float(f64),
    Boolean(bool),
    Vector(Vec<ArgumentValue>),
    Object(HashMap<String, ArgumentValue>),
}

#[derive(PartialEq, Clone)]
pub enum MessageRole {
    User,
    Model,
    Function,
    System,
    Tool,
}

#[derive(PartialEq, Clone)]
pub enum ContentType {
    Text,
}

#[derive(PartialEq, Clone)]
pub enum StopReason {
    Stop,
    Length,
    ContentFilter,
    ToolCalls,
    Null,
}

/* ---------------------- ContextBuilder Implementation --------------------- */

#[derive(Clone)]
pub struct ContextBuilder {
    pub history: Vec<Message>,
}

impl ContextBuilder {
    pub fn transform<F>(self, transformer: F) -> Self
    where
        F: FnOnce(Self) -> Self, // pass ownership down the chain
    {
        transformer(self)
    }

    pub fn add_message(mut self, msg: Message) -> ContextBuilder {
        self.history.push(msg);
        self
    }

    pub async fn send(
        self,
        adapter: ProviderAdapter, // Accept a boxed Send adapter
        system_message: String,
        max_tokens: u32,
        tools: Option<Vec<Tool>>,
    ) -> UnresolvedResponse {
        let prompt_response = adapter(
            self.clone(),
            system_message.clone(),
            max_tokens,
            tools.clone(),
        )
        .await;
        UnresolvedResponse {
            prompt_response: prompt_response
                .unwrap_or_else(|e| panic!("Failed to get PromptResponse: {}", e)),
            context_builder: self,
            tools,
            adapter,
            system_message,
        }
    }

    // todo: send_stream
}

/* -------------------- UnresolvedResponse Implementation ------------------- */
pub struct UnresolvedResponse {
    pub prompt_response: PromptResponse,
    pub context_builder: ContextBuilder,
    pub adapter: ProviderAdapter,
    pub tools: Option<Vec<Tool>>,
    pub system_message: String,
}

impl UnresolvedResponse {
    pub fn resolve_tool_calls(
        self,
        tool_resolver: ToolResolver,
        tool_reprompt_depth: usize,
        max_tokens: u32,
    ) -> Pin<Box<dyn Future<Output = ContextBuilder> + Send>> {
        Box::pin(async move {
            // Base case: stop if depth is 0 or tokens are exhausted
            if tool_reprompt_depth == 0 || max_tokens <= 0 {
                return self
                    .context_builder
                    .add_message(self.prompt_response.message);
            }

            // Handle stop reasons
            match self.prompt_response.stop_reason {
                StopReason::Stop | StopReason::Length | StopReason::ContentFilter => self
                    .context_builder
                    .add_message(self.prompt_response.message),
                StopReason::ToolCalls => {
                    // Resolve tool calls building tool_res
                    // Todo: this should probably be generalized so adapters can format tool calls as needed

                    let mut tool_res = String::new();
                    if let Some(tool_calls) = self.prompt_response.tool_calls {
                        for tool_call in tool_calls {
                            let result = tool_resolver(tool_call.clone());
                            tool_res.push_str(&result);
                            tool_res.push('\n'); // Add a newline after each result
                        }
                    } else {
                        return self.context_builder;
                    }

                    // Unravel the steelwool
                    self.context_builder
                        .add_message(self.prompt_response.message)
                        .add_message(Message {
                            role: MessageRole::Tool,
                            content_type: ContentType::Text,
                            content: tool_res,
                        })
                        .send(self.adapter, self.system_message, max_tokens, self.tools)
                        .await
                        .resolve_tool_calls(
                            tool_resolver,
                            tool_reprompt_depth - 1,
                            max_tokens - self.prompt_response.token_usage,
                        )
                        .await
                }
                StopReason::Null => {
                    // TODO: handle Null reason
                    self.context_builder
                }
            }
        })
    }

    pub async fn resolve_with<F, Fut>(self, resolver: F) -> ContextBuilder
    where
        F: FnOnce(UnresolvedResponse) -> Fut + Send,
        Fut: Future<Output = ContextBuilder> + Send,
    {
        resolver(self).await
    }

    pub async fn resolve_without(self) -> ContextBuilder {
        self.context_builder
            .add_message(self.prompt_response.message)
    }
}
