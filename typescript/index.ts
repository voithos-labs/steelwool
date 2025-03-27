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
 *     systemMessage: string,
 *     maxTokens: number,
 *     tools?: Tool[]
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
  systemMessage: string,
  maxTokens: number,
  tools?: Tool[]
) => PromptFuture;

/**
 * StreamProviderAdapter
 *
 * Function type for streaming responses from LLM providers
 * Uses Observable for streaming pattern
 */
export type StreamProviderAdapter = (
  context: ContextBuilder,
  systemMessage: string,
  maxTokens: number,
  tools?: Tool[]
) => Observable<PromptResponseDelta>;

/**
 * ToolResolver
 *
 * Function for executing tool calls and returning their results as strings
 */
export type ToolResolver = (toolCall: ToolCall) => string;

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
 * Parameter
 */
export interface Parameter {
  returnType: string;
  // ... todo
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
 *   })
 *   .send(adapter, "System message", 1000, tools)
 *   .then(unresolvedResponse => unresolvedResponse.resolveWithout());
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
    systemMessage: string,
    maxTokens: number,
    tools?: Tool[]
  ): Promise<UnresolvedResponse> {
    try {
      const promptResponse = await adapter(
        this.clone(),
        systemMessage,
        maxTokens,
        tools
      );

      return new UnresolvedResponse(promptResponse, this, tools, systemMessage);
    } catch (e) {
      throw new Error(`Failed to get PromptResponse: ${e}`);
    }
  }

  /**
   * Stream a response from a provider, returning an Observable for custom handling
   */
  sendStreaming(
    adapter: StreamProviderAdapter,
    systemMessage: string,
    maxTokens: number,
    tools?: Tool[]
  ): Observable<PromptResponseDelta> {
    return adapter(this.clone(), systemMessage, maxTokens, tools);
  }

  /**
   * Stream a response from a provider with a callback for each delta
   */
  async sendStreamingWithCallback(
    adapter: StreamProviderAdapter,
    systemMessage: string,
    maxTokens: number,
    callback: (delta: PromptResponseDelta) => void,
    tools?: Tool[]
  ): Promise<UnresolvedResponse> {
    const stream = adapter(this.clone(), systemMessage, maxTokens, tools);

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
          resolve(
            new UnresolvedResponse(promptResponse, this, tools, systemMessage)
          );
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
 *   .send(adapter, "System message", 1000, tools)
 *   .then(unresolvedResponse => {
 *     // Choose one of these resolution methods:
 *     return unresolvedResponse.resolveWithout() // Simply add response to context
 *     // OR
 *     return unresolvedResponse.resolveToolCalls(toolResolver, adapter, 5, 1000) // Handle tool calls recursively
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
    public contextBuilder: ContextBuilder,
    public tools: Tool[] | undefined,
    public systemMessage: string
  ) {}

  /**
   * Resolve tool calls recursively provided an adapter
   */
  async resolveToolCallsRecurse(
    toolResolver: ToolResolver,
    adapter: ProviderAdapter,
    toolRepromptDepth: number,
    maxTokens: number
  ): Promise<ContextBuilder> {
    // Base case: stop if depth is 0 or tokens are exhausted
    if (toolRepromptDepth === 0 || maxTokens === 0) {
      return this.contextBuilder.addMessage(this.promptResponse.message);
    }

    switch (this.promptResponse.stopReason) {
      case StopReason.Stop:
      case StopReason.Length:
      case StopReason.ContentFilter:
        return this.contextBuilder.addMessage(this.promptResponse.message);

      case StopReason.ToolCalls:
        // Resolve tool calls
        let toolResults = "";
        if (
          this.promptResponse.toolCalls &&
          this.promptResponse.toolCalls.length > 0
        ) {
          for (const toolCall of this.promptResponse.toolCalls) {
            const result = toolResolver(toolCall);
            toolResults += result + "\n";
          }
        } else {
          return this.contextBuilder;
        }

        // Add model response and tool results to context
        const updatedContext = this.contextBuilder
          .addMessage(this.promptResponse.message)
          .addMessage({
            role: MessageRole.Tool,
            contentType: ContentType.Text,
            content: toolResults,
          });

        // Send updated context back to the model
        const nextResponse = await updatedContext.send(
          adapter,
          this.systemMessage,
          maxTokens,
          this.tools
        );

        // Recursively resolve next response
        return nextResponse.resolveToolCallsRecurse(
          toolResolver,
          adapter,
          toolRepromptDepth - 1,
          maxTokens - this.promptResponse.tokenUsage
        );

      case StopReason.Null:
      default:
        throw new Error(
          `Unhandled stop reason: ${this.promptResponse.stopReason}`
        );
    }
  }

  async resolveToolCalls(toolResolver: ToolResolver) {}

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
   * Resolves the response without handling tool calls or additional processing
   */
  resolveWithout(): ContextBuilder {
    return this.contextBuilder.addMessage(this.promptResponse.message);
  }
}
