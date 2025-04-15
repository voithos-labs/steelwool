import { Observable } from "rxjs";

/* --------------------------- Function Signatures -------------------------- */

/**
 * PromptFuture
 * Type Alias: Describes a Promise that resolves to a PromptResponse
 */
export type PromptFuture = Promise<PromptResponse>;

/**
 * ProviderAdapter
 *
 * Function type for creating LLM provider adapters
 *
 * Example:
 * ```typescript
 * export function ollamaAdapterFactory(modelName: string): ProviderAdapter {
 *   return async (
 *     context: ContextBuilder,
 *     maxTokens: number
 *   ) => {
 *     const history = context.history;
 *
 *     // Ollama interaction...
 *     return promptResponse;
 *   };
 * }
 * ```
 */
export type ProviderAdapter = (
  context: ContextBuilder,
  maxTokens: number
) => PromptFuture;

/**
 * StreamProviderAdapter
 *
 * Function type for streaming responses from LLM providers
 * Uses Observable for streaming pattern
 */
export type StreamProviderAdapter = (
  context: ContextBuilder,
  maxTokens: number
) => Observable<PromptResponseDelta>;

/**
 * ToolExecuter
 *
 * Function for executing tool calls and returning their results as strings
 * Serves as the implementation bridge between model-requested tool operations
 * and the actual business logic that performs those operations
 */
export type ToolExecuter = (toolCall: ToolCall) => Promise<string>;

/* ------------------------------ Data Interfaces --------------------------- */

/**
 * Message
 *
 * Atomic message type with a specific role, content, and content type
 */
export interface Message {
  role: MessageRole;
  content: string;
  contentType: ContentType;
}

/**
 * PromptResponse
 *
 * Response content from an LLM
 */
export interface PromptResponse {
  message: Message;
  stopReason: StopReason;
  tokenUsage: number;
  toolCalls?: ToolCall[];
}

/**
 * PromptResponseDelta
 *
 * Partial response for streaming
 */
export interface PromptResponseDelta {
  content: string;
  stopReason?: StopReason;
  toolCall?: ToolCall;
}

/**
 * Tool
 *
 * Describes a tool available to a model
 */
export interface Tool {
  name: string;
  description: string;
  schema: any; // JSON schema
  required: boolean;
}

/**
 * ToolCall
 *
 * Describes a parsed tool call
 */
export interface ToolCall {
  id: string;
  name: string;
  arguments: any; // JSON
}

/**
 * ToolResult
 *
 * Result of a tool call execution
 */
export interface ToolResult {
  toolCallId: string;
  result: string;
  error: boolean;
}

/* ---------------------------------- Enums --------------------------------- */

/**
 * ArgumentValue
 *
 * Union type representing different possible argument values
 */
export type ArgumentValue =
  | { type: "string"; value: string }
  | { type: "integer"; value: number }
  | { type: "float"; value: number }
  | { type: "boolean"; value: boolean }
  | { type: "array"; value: ArgumentValue[] }
  | { type: "object"; value: Record<string, ArgumentValue> };

/**
 * MessageRole
 */
export enum MessageRole {
  User = "User",
  Model = "Model",
  Function = "Function",
  System = "System",
  Tool = "Tool",
}

/**
 * ContentType
 */
export enum ContentType {
  Text = "Text",
}

/**
 * StopReason
 */
export enum StopReason {
  Stop = "stop",
  Length = "length",
  ContentFilter = "content_filter",
  ToolCalls = "tool_calls",
  Null = "null",
}

/* ----------------------------- ContextBuilder ----------------------------- */

/**
 * ContextBuilder
 *
 * Main entry point for the steelwool library
 *
 * This class provides methods to modify the context by chaining calls
 * in a functional style. For example:
 *
 * ```typescript
 * const contextBuilder = new ContextBuilder()
 *   .addMessage(message)
 *   .transformWith(ctx => {
 *     // Custom transformation
 *     return ctx;
 *   });
 *
 * // Send to LLM and get response
 * const response = await contextBuilder.send(adapter, 1000);
 *
 * // Simply add the response to history
 * const newContext = response.resolveWithout();
 * ```
 */
export class ContextBuilder {
  public history: Message[];

  constructor() {
    this.history = [];
  }

  /**
   * Apply a custom transformation function to the builder
   */
  transformWith(
    transformer: (context: ContextBuilder) => ContextBuilder
  ): ContextBuilder {
    return transformer(this);
  }

  /**
   * Add a message to the context's history
   */
  addMessage(msg: Message): ContextBuilder {
    const newContext = new ContextBuilder();
    newContext.history = [...this.history, msg];
    return newContext;
  }

  /**
   * Send the context using a specified adapter and return an UnresolvedResponse
   */
  async send(
    adapter: ProviderAdapter,
    maxTokens: number
  ): Promise<UnresolvedResponse> {
    try {
      const promptResponse = await adapter(this.clone(), maxTokens);
      return new UnresolvedResponse(promptResponse, this);
    } catch (e) {
      throw new Error(`Failed to get PromptResponse: ${e}`);
    }
  }

  /**
   * Stream a response from a provider, returning an Observable for custom handling
   */
  sendStreaming(
    adapter: StreamProviderAdapter,
    maxTokens: number
  ): Observable<PromptResponseDelta> {
    return adapter(this.clone(), maxTokens);
  }

  /**
   * Stream a response from a provider with a callback for each delta
   */
  async sendStreamingWithCallback(
    adapter: StreamProviderAdapter,
    maxTokens: number,
    callback: (delta: PromptResponseDelta) => void
  ): Promise<UnresolvedResponse> {
    const stream = adapter(this.clone(), maxTokens);

    // Collect the stream into a complete PromptResponse
    let content = "";
    let finalStopReason = StopReason.Null;
    let toolCalls: ToolCall[] = [];

    return new Promise((resolve, reject) => {
      const subscription = stream.subscribe({
        next: (delta) => {
          // Execute callback
          callback(delta);

          // Append content
          content += delta.content;

          // Update tool call if provided
          if (delta.toolCall) {
            toolCalls.push(delta.toolCall);
          }

          // Update stop reason
          if (delta.stopReason) {
            finalStopReason = delta.stopReason;
          }
        },
        error: (err) => {
          reject(err);
        },
        complete: () => {
          // Create the final message and response
          const message: Message = {
            role: MessageRole.Model,
            content,
            contentType: ContentType.Text,
          };

          const promptResponse: PromptResponse = {
            message,
            stopReason: finalStopReason,
            tokenUsage: 0,
            toolCalls: toolCalls.length > 0 ? toolCalls : undefined,
          };

          // Return as UnresolvedResponse for consistent API
          resolve(new UnresolvedResponse(promptResponse, this));

          // Clean up subscription
          subscription.unsubscribe();
        },
      });
    });
  }

  /**
   * Create a clone of the current context
   */
  clone(): ContextBuilder {
    const newContext = new ContextBuilder();
    newContext.history = [...this.history];
    return newContext;
  }
}

/* --------------------------- UnresolvedResponse --------------------------- */

/**
 * UnresolvedResponse
 *
 * Intermediate state after LLM interaction
 *
 * This class represents the state after sending a prompt to an LLM but before
 * resolving any tool calls or further processing. It provides methods to handle
 * the model's response in different ways. For example:
 *
 * ```typescript
 * const resolvedContext = await contextBuilder
 *   .send(adapter, 1000)
 *   .then(unresolvedResponse => {
 *     // Choose one of these resolution methods:
 *     return unresolvedResponse.resolveWithout() // Simply add response to context
 *     // OR
 *     return unresolvedResponse.resolve(toolExecuter) // Execute tool calls and add to context
 *     // OR
 *     return unresolvedResponse.resolveWith(async (ur) => {
 *       // Custom async resolution logic
 *       return ur.contextBuilder.addMessage(ur.promptResponse.message);
 *     });
 *   });
 * ```
 */
export class UnresolvedResponse {
  constructor(
    public promptResponse: PromptResponse,
    public contextBuilder: ContextBuilder
  ) {}

  /**
   * Resolve by executing tool calls and adding results to context
   */
  async resolve(toolExecuter: ToolExecuter): Promise<ContextBuilder> {
    const result = await this.execToolCalls(toolExecuter);
    return result.contextBuilder;
  }

  /**
   * Resolve with retry logic when tool calls fail
   * (Currently a placeholder for future implementation)
   */
  async resolveWithRetry(
    toolExecuter: ToolExecuter,
    retryDepth?: number
  ): Promise<ContextBuilder> {
    // todo
    return this.contextBuilder;
  }

  /**
   * Execute any tool calls and add the results to context
   */
  async execToolCalls(toolExecuter: ToolExecuter): Promise<UnresolvedResponse> {
    // First add the model's message to the context
    let updatedContextBuilder = this.contextBuilder.addMessage(
      this.promptResponse.message
    );

    // If there are no tool calls or the stop reason isn't ToolCalls, just return
    if (
      this.promptResponse.stopReason !== StopReason.ToolCalls ||
      !this.promptResponse.toolCalls
    ) {
      return this;
    }

    // Process tool calls
    let toolResults = "";

    for (const toolCall of this.promptResponse.toolCalls) {
      try {
        const result = await toolExecuter(toolCall);
        toolResults += result;
      } catch (err) {
        toolResults += `Error in tool call ${toolCall.id} of ${toolCall.name}: ${err}`;
      }
      toolResults += "\n";
    }

    // Add tool results as a Tool message
    updatedContextBuilder = updatedContextBuilder.addMessage({
      role: MessageRole.Tool,
      contentType: ContentType.Text,
      content: toolResults,
    });

    // Return a new UnresolvedResponse with the updated context
    return new UnresolvedResponse(this.promptResponse, updatedContextBuilder);
  }

  /**
   * Resolve to ContextBuilder with async resolver
   */
  async resolveWith(
    resolver: (
      unresolvedResponse: UnresolvedResponse
    ) => Promise<ContextBuilder>
  ): Promise<ContextBuilder> {
    return resolver(this);
  }

  /**
   * Resolve with a synchronous resolver
   */
  resolveWithSync(
    resolver: (unresolvedResponse: UnresolvedResponse) => ContextBuilder
  ): ContextBuilder {
    return resolver(this);
  }

  /**
   * Transform with async transformer that returns UnresolvedResponse
   */
  async transformWith(
    transformer: (
      unresolvedResponse: UnresolvedResponse
    ) => Promise<UnresolvedResponse>
  ): Promise<UnresolvedResponse> {
    return transformer(this);
  }

  /**
   * Transform with synchronous transformer that returns UnresolvedResponse
   */
  transformWithSync(
    transformer: (unresolvedResponse: UnresolvedResponse) => UnresolvedResponse
  ): UnresolvedResponse {
    return transformer(this);
  }

  /**
   * Resolves the response without handling tool calls or additional processing
   */
  resolveWithout(): ContextBuilder {
    return this.contextBuilder.addMessage(this.promptResponse.message);
  }
}
