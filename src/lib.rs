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

/// ## `PromptFuture`
/// **Type Alias**: Describes a typed future `PromptResponse` that represents the returned data
/// from any given LLM `send` interaction
pub type PromptFuture = Pin<Box<dyn Future<Output = Result<PromptResponse, String>> + Send>>;

/// ## `ProviderAdapter`
///
/// **Use:** for creating adapter function factories, e.g.:
///
/// ```rust,ignore
/// pub fn ollama_adapter_factory(model_name: String) -> ProviderAdapter {
///     Arc::new(
///         move |context: ContextBuilder,
///               system_message: String,
///               _max_tokens: u32,
///               _tools: Option<Vec<Tool>>| {
///             let model = model_name.clone();
///             let history = context.history.clone();
///
///             Box::pin(async move {
///                 // Ollama interaction...
///             })
///         }
///     )
/// }
///
/// ```
///
pub type ProviderAdapter =
    Arc<dyn Fn(ContextBuilder, String, u32, Option<Vec<Tool>>) -> PromptFuture + Send + Sync>;

/// ## `StreamProviderAdapter`
/// **Type Alias**: Function adapter for streaming responses from LLM providers.
///
/// Similar to ProviderAdapter but returns a stream of response chunks instead of a single future.
/// Enables processing partial responses as they arrive from the model.
///
/// **TODO:**
///
/// Use a `PromptResponseDelta` struct or smth to capture values instead of `PromptResponse`
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

/// ## `ToolResolver`
/// Function for executing tool calls and returning their results as strings.
///
/// Serves as the implementation bridge between model-requested tool operations
/// and the actual business logic that performs those operations
pub type ToolResolver = Arc<dyn Fn(ToolCall) -> String + Send + Sync>;

/* ------------------------------ Data Structs ------------------------------ */

// Message

/// Atomic message type with a specific role, content, and content type.
#[derive(Clone)]
pub struct Message {
    pub role: MessageRole,
    pub content: String,
    pub content_type: ContentType,
}

// Responses

/// Prompt response content
/// ## Fields
/// - `message`: The main message content of the response.
/// - `stop_reason`: The reason why the response generation was stopped.
/// - `token_usage`: The number of tokens used in generating the response.
/// - `tool_calls`: An optional list of tool calls made during the response generation.
pub struct PromptResponse {
    pub message: Message,
    pub stop_reason: StopReason,
    pub token_usage: u32,
    pub tool_calls: Option<Vec<ToolCall>>,
}

// Tools

/// Describes a tool available to a model
#[derive(Clone)]
pub struct Tool {
    pub name: String,
    pub description: String,
    pub parameters: HashMap<String, Parameter>,
}

/// Describes a parsed tool-call
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

/* ----------------------------- ContextBuilder ----------------------------- */
/// ## `ContextBuilder`
/// _steelwool entry point_
///
/// This struct provides methods to modify the context by chaining calls
/// in a functional style. For example:
///
/// ```rust,ignore
/// let context_builder = ContextBuilder::new()
///     .add_message(message) // -> ContextBuilder
///     .your_transformer() // your custom transformer -> ContextBuilder
///     .self_rag(vector_interface) // -> ContextBuilder
///     .transform_with(|ctx| {
///         // Perform custom transformation (for small in-line stuff, e.g. character stripping)
///         // otherwise implement a standalone transformer for more advanced tasks
///         // E.g. 'impl CustomTransformers for ContextBuilder { ... }'
///         // see .your_transformer()
///         ctx
///     })
///     .send(adapter, "System message".to_string(), 1000, tools); // async send; non-streaming
///     .await
///     .resolve_without(); // RESOLVER
///     // See `UnresolvedResponse` for dealing with unresolved
///     // states returned by send opertaions
/// ```
/// ---
/// ## Methods
///
/// - `transform_with`: Allows applying a custom transformation function to the builder.
/// - `add_message`: Adds a message to the context's history.
/// - `send`: Sends the context using a specified adapter and returns an `UnresolvedResponse`.
#[derive(Clone)]
pub struct ContextBuilder {
    pub history: Vec<Message>,
}

impl ContextBuilder {
    pub fn transform_with<F>(self, transformer: F) -> Self
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

/* --------------------------- UnresolvedResponse --------------------------- */
/// ## `UnresolvedResponse`
/// _intermediate state after LLM interaction_
///
/// This struct represents the state after sending a prompt to an LLM but before
/// resolving any tool calls or further processing. It provides methods to handle
/// the model's response in different ways. For example:
///
/// ```rust,ignore
/// let resolved_context = context_builder
///     .send(adapter, "System message".to_string(), 1000, tools)
///     .await
///     // Choose one of these resolution methods:
///     .resolve_without() // Simply add response to context
///     // OR
///     .resolve_tool_calls(tool_resolver, 5, 1000).await // Handle tool calls recursively
///     // OR
///     .resolve_with(|ur| async move {
///         // Custom async resolution logic
///         ur.context_builder.add_message(ur.prompt_response.message)
///     }).await
///     // OR
///     .resolve_with_sync(|ur| {
///         // Custom synchronous resolution logic
///         ur.context_builder.add_message(ur.prompt_response.message)
///     });
/// ```
///
/// ### Extending
/// You can also make your own transformer that matches the pattern;
/// UnresolvedResponse (self) -> ContextBuilder
///
/// e.g.:
///
/// ```rust,ignore
/// impl CustomResolvers for UnresolvedResponse {
///     fn resolve_and_log(self) -> ContextBuilder {
///         println!("Response resolved, token usage: {}", self.prompt_response.token_usage);
///         self.context_builder.add_message(self.prompt_response.message)
///     }
/// }
/// ```
/// ---
/// ## Methods
///
/// - `resolve_tool_calls`: Recursively resolves tool calls, feeding results back to the model.
/// - `resolve_with`: Applies a custom async transformation function to resolve the response.
/// - `resolve_with_sync`: Applies a custom synchronous transformation function to resolve the response.
/// - `resolve_without`: Simply adds the response message to the context without additional processing.
pub struct UnresolvedResponse {
    pub prompt_response: PromptResponse,
    pub context_builder: ContextBuilder,
    pub adapter: ProviderAdapter,
    pub tools: Option<Vec<Tool>>,
    pub system_message: String,
}

impl UnresolvedResponse {
    /// Recursively resolves tool calls in a model response and feeds results back to the model
    ///
    /// Takes a tool resolver function that handles executing tool calls and converting results to strings.
    /// Implements an agent pattern with controlled recursion depth and token budget management.
    ///
    /// # Arguments
    /// * `tool_resolver` - Function that processes `ToolCall` objects and returns string results
    /// * `tool_reprompt_depth` - Maximum recursion depth for tool-model interaction cycles
    /// * `max_tokens` - Token budget for all subsequent model calls (decremented with each call)
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

    /// Resolve to ContextBuilder with async resolver, good for simple custom resolver usecases
    pub async fn resolve_with<F, Fut>(self, resolver: F) -> ContextBuilder
    where
        F: FnOnce(UnresolvedResponse) -> Fut + Send,
        Fut: Future<Output = ContextBuilder> + Send,
    {
        resolver(self).await
    }

    /// Resolve `UnresolvedResponse` instance with a synchronous resolver that returns a new `ContextBuilder`
    pub fn resolve_with_sync<F>(self, resolver: F) -> ContextBuilder
    where
        F: FnOnce(UnresolvedResponse) -> ContextBuilder + Send,
    {
        resolver(self)
    }

    /// Resolves the response without handling tool calls or additional processing
    ///
    /// This simply adds the prompt response message to the context builder
    pub fn resolve_without(self) -> ContextBuilder {
        self.context_builder
            .add_message(self.prompt_response.message)
    }
}
