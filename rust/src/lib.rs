/// Authors:
/// Alex & Finn
///
/// steelwool is a lightweight library for interacting with LLMs
/* -------------------------------------------------------------------------- */
/*                                  STEELWOOL                                 */
/* -------------------------------------------------------------------------- */

/* ------------------------------ Dependencies ------------------------------ */
use std::future::Future;
use std::pin::Pin;
use std::sync::Arc;

use futures::StreamExt;
use futures::stream::BoxStream;
use serde::{Deserialize, Serialize};

/* -------------------------------- Features -------------------------------- */

pub mod providers {
    #[cfg(feature = "ollama")]
    pub mod ollama;
    #[cfg(feature = "openai")]
    pub mod openai;
}

/* ------------------------------- Signatures ------------------------------- */

/// ## `PromptFuture`
/// **Type Alias**: Describes a typed future returning a `PromptResponse` that represents
/// the result of an LLM interaction
pub type PromptFuture = Pin<Box<dyn Future<Output = Result<PromptResponse, String>> + Send>>;

/// ## `ProviderAdapter`
///
/// **Type Alias**: Function adapter for sending requests to LLM providers.
///
/// **Parameters**:
/// - `context: ContextBuilder` - The context containing message history
/// - `max_tokens: u32` - Maximum number of tokens for the response
///
/// **Returns**: A `PromptFuture` containing the model's response
///
/// **Example**:
/// ```rust,ignore
/// pub fn ollama_adapter_factory(model_name: String) -> ProviderAdapter {
///     Arc::new(
///         move |context: ContextBuilder,
///               max_tokens: u32| {
///             let model = model_name.clone();
///             let history = context.history.clone();
///
///             Box::pin(async move {
///                 // Ollama interaction...
///             })
///         }
///     )
/// }
/// ```
pub type ProviderAdapter = Arc<dyn Fn(ContextBuilder, u32) -> PromptFuture + Send + Sync>;

/// ## `StreamProviderAdapter`
/// **Type Alias**: Function adapter for streaming responses from LLM providers.
///
/// Similar to ProviderAdapter but returns a stream of response chunks instead of a single future.
/// Enables processing partial responses as they arrive from the model.
pub type StreamProviderAdapter = Arc<
    dyn Fn(ContextBuilder, u32) -> BoxStream<'static, Result<PromptResponseDelta, String>>
        + Send
        + Sync,
>;

/// ## `ToolExecuter`
/// Function for executing tool calls and returning their results as strings.
///
/// Serves as the implementation bridge between model-requested tool operations
/// and the actual business logic that performs those operations
pub type ToolExecuter = Arc<
    dyn Fn(ToolCall) -> Pin<Box<dyn Future<Output = Result<String, String>> + Send>> + Send + Sync,
>;

/* ------------------------------ Data Structs ------------------------------ */

// Message

/// Atomic message type with a specific role, content, and content type.
#[derive(Serialize, Deserialize, Clone, PartialEq)]
pub struct Message {
    pub role: MessageRole,
    pub content: String,
    pub content_type: ContentType,
    // pub tool_results: Option<Vec<ToolResult>>,
}

// Responses

/// Prompt response content
#[derive(Serialize, Deserialize, Clone)]
pub struct PromptResponse {
    pub message: Message,
    pub stop_reason: StopReason,
    pub token_usage: u32,
    pub tool_calls: Option<Vec<ToolCall>>,
}

/// Prompt response delta for streaming
#[derive(Serialize, Deserialize, Clone)]
pub struct PromptResponseDelta {
    pub content: String,
    pub stop_reason: Option<StopReason>,
    pub tool_calls: Option<Vec<ToolCall>>,
    pub cumulative_tokens: u32,
}

// Tools

/// Describes a tool available to a model
#[derive(Serialize, Deserialize, Clone)]
pub struct ToolDescriptor {
    pub name: String,
    pub description: String,
    pub schema: serde_json::Value,
    pub required: bool,
}

/// Describes a parsed tool-call
#[derive(Serialize, Deserialize, Clone)]
pub struct ToolCall {
    pub id: String,
    pub name: String,
    pub arguments: serde_json::Value,
}

#[derive(Clone, Serialize, Deserialize)]
pub struct ToolResult {
    pub tool_call_id: String,
    pub result: String,
    pub error: bool,
}

/* ---------------------------------- Enums --------------------------------- */

#[derive(PartialEq, Clone, Deserialize, Serialize)]
pub enum MessageRole {
    User,
    Model,
    Function,
    System,
    Tool,
}
#[derive(PartialEq, Clone, Deserialize, Serialize)]
pub enum ContentType {
    Text,
}
#[derive(PartialEq, Clone, Deserialize, Serialize)]
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
///     .transform_with(|ctx| {
///         // Perform custom transformation
///         ctx
///     })
///     .send(adapter, 1000) // async send; non-streaming
///     .await
///     .resolve_without(); // Add response to history
/// ```
///
/// ## Methods
///
/// - `transform_with`: Applies a custom transformation function to the builder
/// - `add_message`: Adds a message to the context's history
/// - `send`: Sends the context to an LLM and returns an `UnresolvedResponse`
/// - `send_streaming`: Sends the context and returns a stream of response deltas
/// - `send_streaming_with_callback`: Streams with a callback for each delta
#[derive(Serialize, Deserialize, Clone)]
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

    pub fn add_message(mut self, msg: Message) -> Self {
        self.history.push(msg);
        self
    }

    pub async fn send(
        self,
        adapter: ProviderAdapter, // Accept a boxed Send adapter
        max_tokens: u32,
    ) -> UnresolvedResponse {
        let prompt_response = adapter(self.clone(), max_tokens).await;
        UnresolvedResponse {
            prompt_response: prompt_response
                .unwrap_or_else(|e| panic!("Failed to get PromptResponse: {}", e)),
            context_builder: self,
        }
    }

    /// Stream a response from a provider, returning the raw stream for custom handling
    pub fn send_streaming(
        self,
        adapter: StreamProviderAdapter,
        max_tokens: u32,
    ) -> BoxStream<'static, Result<PromptResponseDelta, String>> {
        adapter(self.clone(), max_tokens)
    }

    /// Stream a response from a provider with a callback for each delta
    pub async fn send_streaming_with_callback<F>(
        self,
        adapter: StreamProviderAdapter,
        max_tokens: u32,
        callback: F,
    ) -> Result<UnresolvedResponse, String>
    where
        F: Fn(Result<PromptResponseDelta, String>) + Send + Sync + 'static,
    {
        let stream = adapter(self.clone(), max_tokens);

        // Collect the stream into a complete PromptResponse
        let mut content = String::new();
        let mut final_stop_reason = StopReason::Null;
        let mut tool_calls: Option<Vec<ToolCall>> = None;

        // Process each delta
        let mut stream = Box::pin(stream);
        while let Some(delta_result) = stream.next().await {
            // 1. Callback before the rest can break
            callback(delta_result.clone());

            if let Ok(delta) = delta_result {
                // Append content
                content.push_str(&delta.content);

                // Update tool call if provided
                if let Some(tool_call) = delta.tool_calls {
                    match &mut tool_calls {
                        Some(calls) => calls.extend(tool_call),
                        None => tool_calls = Some(tool_call),
                    }
                }

                // Stop
                if let Some(reason) = delta.stop_reason {
                    final_stop_reason = reason;
                }
            } else if let Err(err) = delta_result {
                return Err(err);
            }
        }

        // Create the final message and response
        let message = Message {
            role: MessageRole::Model,
            content,
            content_type: ContentType::Text,
        };

        let prompt_response = PromptResponse {
            message,
            stop_reason: final_stop_reason,
            token_usage: 0,
            tool_calls,
        };

        // Return as UnresolvedResponse for consistent API
        Ok(UnresolvedResponse {
            prompt_response,
            context_builder: self,
        })
    }
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
///     .send(adapter, 1000)
///     .await
///     // Choose one of these resolution methods:
///     .resolve_without() // Simply add response to context
///     // OR
///     .resolve(tool_executer).await // Execute tool calls and add to context
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
/// *also supports chaining, see `exec_tool_calls` and `transform_with`
///
/// ## Methods
///
/// - `resolve`: Executes any tool calls and returns the updated context
/// - `resolve_with_retry`: Handles tool calls with retry logic
/// - `resolve_without`: Adds the response to context without handling tool calls
/// - `exec_tool_calls`: Executes tool calls and adds results to context
/// - `resolve_with`/`resolve_with_sync`: Custom resolution with async/sync functions
/// - `transform_with`/`transform_with_sync`: Custom transformations returning `Self`
#[derive(Serialize, Deserialize, Clone)]
pub struct UnresolvedResponse {
    pub prompt_response: PromptResponse,
    pub context_builder: ContextBuilder,
}

impl UnresolvedResponse {
    pub async fn resolve(self, tool_executer: ToolExecuter) -> ContextBuilder {
        let unresolved_response = self.exec_tool_calls(tool_executer).await;
        unresolved_response.context_builder
    }

    pub async fn resolve_with_retry(
        self,
        tool_executer: ToolExecuter,
        retry_depth: Option<usize>,
    ) -> ContextBuilder {
        // todo
        self.context_builder
    }

    pub fn resolve_without(self) -> ContextBuilder {
        self.context_builder
            .add_message(self.prompt_response.message)
    }

    pub async fn exec_tool_calls(self, tool_executer: ToolExecuter) -> Self {
        let mut unresolved_response = self.clone();

        // Add the current message to the context
        unresolved_response.context_builder = self
            .context_builder
            .add_message(self.prompt_response.message.clone());

        // If there are no tool calls or the stop reason isn't ToolCalls, just return
        if self.prompt_response.stop_reason != StopReason::ToolCalls
            || self.prompt_response.tool_calls.is_none()
        {
            return unresolved_response;
        }

        // Resolve tool calls
        let mut tool_res = String::new();

        if let Some(tool_calls) = self.prompt_response.tool_calls.clone() {
            for tool_call in tool_calls {
                let result = tool_executer(tool_call.clone()).await;
                match result {
                    Ok(output) => tool_res.push_str(&output),
                    Err(err) => tool_res.push_str(&format!(
                        "Error in tool call {} of {}: {}",
                        tool_call.id, tool_call.name, err
                    )),
                }
                tool_res.push('\n'); // Add a newline after each result
            }
        }

        // Add tool response message to the context
        unresolved_response.context_builder =
            unresolved_response.context_builder.add_message(Message {
                role: MessageRole::Tool,
                content_type: ContentType::Text,
                content: tool_res,
            });

        unresolved_response
    }

    // Silly methods for lazy extensions

    pub async fn resolve_with<F, Fut>(self, resolver: F) -> ContextBuilder
    where
        F: FnOnce(Self) -> Fut + Send,
        Fut: Future<Output = ContextBuilder> + Send,
    {
        resolver(self).await
    }

    pub fn resolve_with_sync<F>(self, resolver: F) -> ContextBuilder
    where
        F: FnOnce(Self) -> ContextBuilder + Send,
    {
        resolver(self)
    }

    pub async fn transform_with<F, Fut>(self, resolver: F) -> Self
    where
        F: FnOnce(Self) -> Fut + Send,
        Fut: Future<Output = Self> + Send,
    {
        resolver(self).await
    }

    pub fn transform_with_sync<F>(self, resolver: F) -> Self
    where
        F: FnOnce(Self) -> Self + Send,
    {
        resolver(self)
    }
}
