from pydantic import BaseModel, Field
from typing import Any, AsyncIterator, Awaitable, List, Callable, Self, Optional
from ._types import (
    Message,
    PromptResponseDelta,
    StreamProviderAdapter, 
    Tool,
    ToolCall,
    PromptResponse,
    ProviderAdapter,
)
from ._enums import (
    ContentType,
    StopReason,
    MessageRole
)
from ._unresolved_response import UnresolvedResponse

class ContextBuilder(BaseModel):

    history: List[Message] = Field(default=[])

    def transform_with(
        self, 
        transformer: Callable[[Self], Self]
    ) -> Self:
        """
            Applies `transformer` onto the builder.
        """
        return transformer(self)
    
    def add_message(self, message : Message) -> Self:
        """
            Pushes a message into the history
        """
        self.history.append(
            message
        )
        return self
    
    async def send(
        self,
        adapter : ProviderAdapter,
        system_message : str,
        max_tokens: int,
        tools: Optional[List[Tool]]
    ) -> UnresolvedResponse:
        """
            Send the context to an LLM provider.
        """
        
        response : PromptResponse = await adapter(
            self,
            system_message,
            max_tokens,
            tools
        )

        return UnresolvedResponse(
            prompt_response = response,
            context_builder = self,
            tools = tools,
            system_message = system_message
        )
    
    async def send_streaming(
        self,
        adapter : StreamProviderAdapter,
        system_message : str,
        max_tokens : int,
        tools : Optional[List[Tool]]
    ) -> AsyncIterator[PromptResponseDelta]:
        """
            Streams a response back from a LLM provider.

            This returns the raw `AsyncIterator[PromptResponseDelta]` data
            from the adapter itself.
        """
        return adapter(
            self,
            system_message,
            max_tokens,
            tools
        )
    
    async def send_streaming_with_callback(
        self,
        adapter: StreamProviderAdapter,
        system_message : str,
        max_tokens : int,
        tools : Optional[List[Tool]],
        callback: Callable[[PromptResponseDelta], Awaitable[Any]]
    ) -> UnresolvedResponse:
        
        stream = adapter(
            self,
            system_message,
            max_tokens,
            tools
        )

        response_content : str = ""
        response_stop_reason : StopReason = StopReason.NULL
        response_token_usage : int = 0
        response_tool_calls : List["ToolCall"] = []

        async for delta in stream:

            await callback(delta)

            response_content += delta.content

            if delta.tool_call:
                response_tool_calls.append(
                    delta.tool_call
                )

            if delta.stop_reason:
                response_stop_reason = delta.stop_reason

        response_message = Message(
            role = MessageRole.MODEL,
            content = response_content,
            content_type = ContentType.TEXT
        )

        prompt_response = PromptResponse(
            message = response_message,
            stop_reason = response_stop_reason,
            token_usage = response_token_usage,
            tool_calls = response_tool_calls
        )

        return UnresolvedResponse(
            prompt_response = prompt_response,
            context_builder = self,
            tools = tools,
            system_message = system_message
        )