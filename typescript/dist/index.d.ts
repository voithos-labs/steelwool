import { Observable } from "rxjs";
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
export type ProviderAdapter = (context: ContextBuilder, maxTokens: number) => PromptFuture;
/**
 * StreamProviderAdapter
 *
 * Function type for streaming responses from LLM providers
 * Uses Observable for streaming pattern
 */
export type StreamProviderAdapter = (context: ContextBuilder, maxTokens: number) => Observable<PromptResponseDelta>;
/**
 * ToolExecuter
 *
 * Function for executing tool calls and returning their results as strings
 * Serves as the implementation bridge between model-requested tool operations
 * and the actual business logic that performs those operations
 */
export type ToolExecuter = (toolCall: ToolCall) => Promise<string>;
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
    cumulative_tokens: number;
}
/**
 * Tool
 *
 * Describes a tool available to a model
 */
export interface Tool {
    name: string;
    description: string;
    schema: any;
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
    arguments: any;
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
/**
 * ArgumentValue
 *
 * Union type representing different possible argument values
 */
export type ArgumentValue = {
    type: "string";
    value: string;
} | {
    type: "integer";
    value: number;
} | {
    type: "float";
    value: number;
} | {
    type: "boolean";
    value: boolean;
} | {
    type: "array";
    value: ArgumentValue[];
} | {
    type: "object";
    value: Record<string, ArgumentValue>;
};
/**
 * MessageRole
 */
export declare enum MessageRole {
    User = "User",
    Model = "Model",
    Function = "Function",
    System = "System",
    Tool = "Tool"
}
/**
 * ContentType
 */
export declare enum ContentType {
    Text = "Text"
}
/**
 * StopReason
 */
export declare enum StopReason {
    Stop = "stop",
    Length = "length",
    ContentFilter = "content_filter",
    ToolCalls = "tool_calls",
    Null = "null"
}
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
export declare class ContextBuilder {
    history: Message[];
    constructor();
    /**
     * Apply a custom transformation function to the builder
     */
    transformWith(transformer: (context: ContextBuilder) => ContextBuilder): ContextBuilder;
    /**
     * Add a message to the context's history
     */
    addMessage(msg: Message): ContextBuilder;
    /**
     * Send the context using a specified adapter and return an UnresolvedResponse
     */
    send(adapter: ProviderAdapter, maxTokens: number): Promise<UnresolvedResponse>;
    /**
     * Stream a response from a provider, returning an Observable for custom handling
     */
    sendStreaming(adapter: StreamProviderAdapter, maxTokens: number): Observable<PromptResponseDelta>;
    /**
     * Stream a response from a provider with a callback for each delta
     */
    sendStreamingWithCallback(adapter: StreamProviderAdapter, maxTokens: number, callback: (delta: PromptResponseDelta) => void): Promise<UnresolvedResponse>;
    /**
     * Create a clone of the current context
     */
    clone(): ContextBuilder;
}
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
export declare class UnresolvedResponse {
    promptResponse: PromptResponse;
    contextBuilder: ContextBuilder;
    constructor(promptResponse: PromptResponse, contextBuilder: ContextBuilder);
    /**
     * Resolve by executing tool calls and adding results to context
     */
    resolve(toolExecuter: ToolExecuter): Promise<ContextBuilder>;
    /**
     * Resolve with retry logic when tool calls fail
     * (Currently a placeholder for future implementation)
     */
    resolveWithRetry(toolExecuter: ToolExecuter, retryDepth?: number): Promise<ContextBuilder>;
    /**
     * Execute any tool calls and add the results to context
     */
    execToolCalls(toolExecuter: ToolExecuter): Promise<UnresolvedResponse>;
    /**
     * Resolve to ContextBuilder with async resolver
     */
    resolveWith(resolver: (unresolvedResponse: UnresolvedResponse) => Promise<ContextBuilder>): Promise<ContextBuilder>;
    /**
     * Resolve with a synchronous resolver
     */
    resolveWithSync(resolver: (unresolvedResponse: UnresolvedResponse) => ContextBuilder): ContextBuilder;
    /**
     * Transform with async transformer that returns UnresolvedResponse
     */
    transformWith(transformer: (unresolvedResponse: UnresolvedResponse) => Promise<UnresolvedResponse>): Promise<UnresolvedResponse>;
    /**
     * Transform with synchronous transformer that returns UnresolvedResponse
     */
    transformWithSync(transformer: (unresolvedResponse: UnresolvedResponse) => UnresolvedResponse): UnresolvedResponse;
    /**
     * Resolves the response without handling tool calls or additional processing
     */
    resolveWithout(): ContextBuilder;
}
