"use strict";
Object.defineProperty(exports, "__esModule", { value: true });
exports.UnresolvedResponse = exports.ContextBuilder = exports.StopReason = exports.ContentType = exports.MessageRole = void 0;
/**
 * MessageRole
 */
var MessageRole;
(function (MessageRole) {
    MessageRole["User"] = "User";
    MessageRole["Model"] = "Model";
    MessageRole["Function"] = "Function";
    MessageRole["System"] = "System";
    MessageRole["Tool"] = "Tool";
})(MessageRole || (exports.MessageRole = MessageRole = {}));
/**
 * ContentType
 */
var ContentType;
(function (ContentType) {
    ContentType["Text"] = "Text";
})(ContentType || (exports.ContentType = ContentType = {}));
/**
 * StopReason
 */
var StopReason;
(function (StopReason) {
    StopReason["Stop"] = "stop";
    StopReason["Length"] = "length";
    StopReason["ContentFilter"] = "content_filter";
    StopReason["ToolCalls"] = "tool_calls";
    StopReason["Null"] = "null";
})(StopReason || (exports.StopReason = StopReason = {}));
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
class ContextBuilder {
    constructor() {
        this.history = [];
    }
    /**
     * Apply a custom transformation function to the builder
     */
    transformWith(transformer) {
        return transformer(this);
    }
    /**
     * Add a message to the context's history
     */
    addMessage(msg) {
        const newContext = new ContextBuilder();
        newContext.history = [...this.history, msg];
        return newContext;
    }
    /**
     * Send the context using a specified adapter and return an UnresolvedResponse
     */
    async send(adapter, maxTokens) {
        try {
            const promptResponse = await adapter(this.clone(), maxTokens);
            return new UnresolvedResponse(promptResponse, this);
        }
        catch (e) {
            throw new Error(`Failed to get PromptResponse: ${e}`);
        }
    }
    /**
     * Stream a response from a provider, returning an Observable for custom handling
     */
    sendStreaming(adapter, maxTokens) {
        return adapter(this.clone(), maxTokens);
    }
    /**
     * Stream a response from a provider with a callback for each delta
     */
    async sendStreamingWithCallback(adapter, maxTokens, callback) {
        const stream = adapter(this.clone(), maxTokens);
        // Collect the stream into a complete PromptResponse
        let content = "";
        let finalStopReason = StopReason.Null;
        let toolCalls = [];
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
                    const message = {
                        role: MessageRole.Model,
                        content,
                        contentType: ContentType.Text,
                    };
                    const promptResponse = {
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
    clone() {
        const newContext = new ContextBuilder();
        newContext.history = [...this.history];
        return newContext;
    }
}
exports.ContextBuilder = ContextBuilder;
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
class UnresolvedResponse {
    constructor(promptResponse, contextBuilder) {
        this.promptResponse = promptResponse;
        this.contextBuilder = contextBuilder;
    }
    /**
     * Resolve by executing tool calls and adding results to context
     */
    async resolve(toolExecuter) {
        const result = await this.execToolCalls(toolExecuter);
        return result.contextBuilder;
    }
    /**
     * Resolve with retry logic when tool calls fail
     * (Currently a placeholder for future implementation)
     */
    async resolveWithRetry(toolExecuter, retryDepth) {
        // todo
        return this.contextBuilder;
    }
    /**
     * Execute any tool calls and add the results to context
     */
    async execToolCalls(toolExecuter) {
        // First add the model's message to the context
        let updatedContextBuilder = this.contextBuilder.addMessage(this.promptResponse.message);
        // If there are no tool calls or the stop reason isn't ToolCalls, just return
        if (this.promptResponse.stopReason !== StopReason.ToolCalls ||
            !this.promptResponse.toolCalls) {
            return this;
        }
        // Process tool calls
        let toolResults = "";
        for (const toolCall of this.promptResponse.toolCalls) {
            try {
                const result = await toolExecuter(toolCall);
                toolResults += result;
            }
            catch (err) {
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
    async resolveWith(resolver) {
        return resolver(this);
    }
    /**
     * Resolve with a synchronous resolver
     */
    resolveWithSync(resolver) {
        return resolver(this);
    }
    /**
     * Transform with async transformer that returns UnresolvedResponse
     */
    async transformWith(transformer) {
        return transformer(this);
    }
    /**
     * Transform with synchronous transformer that returns UnresolvedResponse
     */
    transformWithSync(transformer) {
        return transformer(this);
    }
    /**
     * Resolves the response without handling tool calls or additional processing
     */
    resolveWithout() {
        return this.contextBuilder.addMessage(this.promptResponse.message);
    }
}
exports.UnresolvedResponse = UnresolvedResponse;
