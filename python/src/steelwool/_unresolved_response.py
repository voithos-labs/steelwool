from typing import Awaitable, Callable, List, Optional
from pydantic import BaseModel, Field
from ._context_builder import ContextBuilder
from ._types import (
    PromptResponse,
    ProviderAdapter,
    Tool,
    Message,
    ToolResolver,
)
from ._enums import (
    ContentType,
    MessageRole,
    StopReason
)

class UnresolvedResponse(BaseModel):
    """
        Intermediate state after an LLM interaction.

        This class represents the state after sending a prompt to an LLM
        but prior to resolving tool calls or doing any additional processing.
    """

    prompt_response : PromptResponse
    context_builder : ContextBuilder
    tools : Optional[List[Tool]] = Field(default=None)
    system_message : str

    async def resolve_tool_calls(
        self,
        tool_resolver: ToolResolver
    ) -> ContextBuilder:
        
        self.context_builder.add_message(
            self.prompt_response.message
        )

        tool_response : str = ""

        if self.prompt_response.stop_reason != StopReason.TOOL_CALLS:
            for tc in self.prompt_response.tool_calls:
                
                # Call the tool resolver on the tool call
                result = await tool_resolver(tc)

                # If there is a non-null result, append it to the tool_response
                # string, and add a new-line afterwards.
                if result:
                    tool_response += result + "\n"

        self.context_builder.add_message(
            Message(
                role = MessageRole.TOOL,
                content_type = ContentType.TEXT,
                content = tool_response
            )
        )

        return self
    
    async def resolve_tool_calls_recursively(
        self,
        tool_resolver : ToolResolver,
        tool_reprompt_death : int,
        adapter : ProviderAdapter,
        max_tokens : int
    ) -> ContextBuilder:
        """
            Recursively resolves tool calls in a model response and feeds results back to the model.
        """
        
        # Base case
        if (
            tool_reprompt_death == 0 
            or max_tokens <= 0
        ):
            return self.context_builder.add_message(
                self.prompt_response.message
            )
        
        # If the completion stops for any normal reason (e.g., naturally, content filter, etc.)
        # then return the response. Because we don't need to call any tools.
        if (self.prompt_response.stop_reason in (StopReason.STOP, StopReason.LENGTH, StopReason.CONTENT_FILTER)):

            return self.context_builder.add_message(
                self.prompt_response.message
            )
        
        # If the completion stops because there is a tool call, resolve tool call and re-prompt
        # keep reprompting until prompt death == 0, max_tokens <= 0, or there is no more tools needed to be called!
        elif self.prompt_response.stop_reason == StopReason.TOOL_CALLS:

            if not self.prompt_response.tool_calls:
                return self.context_builder

            tool_response : str = ""

            for tc in self.prompt_response.tool_calls:
                result = await tool_resolver(tc)

                if result is not None:
                    tool_response += result + "\n"

                unresolved_response = await (
                    self.context_builder
                        .add_message(self.prompt_response.message)
                        .add_message(Message(
                            role = MessageRole.TOOL,
                            content_type = ContentType.TEXT,
                            content = tool_response
                        ))
                        .send(adapter, self.system_message, max_tokens, self.tools))
                
                return await (
                    unresolved_response
                        .resolve_tool_calls_recursively(
                            tool_resolver=tool_resolver,
                            tool_reprompt_death=tool_reprompt_death - 1,
                            adapter = adapter,
                            max_tokens = max_tokens - self.prompt_response.token_usage
                        ))


        elif self.prompt_response.stop_reason == StopReason.NULL:

            raise Exception("Unexpected StopReason.NULL encountered during tool call resolution")
        
    async def resolve_with(
        self,
        resolver : Callable[["UnresolvedResponse"], Awaitable[ContextBuilder]]
    ) -> ContextBuilder:
        """
            Resolve to ContextBuilder using a custom async resolver
        """
        return await resolver(self)
    
    def resolve_with_sync(
        self,
        resolver : Callable[["UnresolvedResponse"], ContextBuilder]
    ) -> ContextBuilder:
        """
            Resolve to ContextBuilder using a custom sync resolver
        """
        return resolver(self)
    
    def resolve_without(self) -> ContextBuilder:
        """
            Resolves the response without handling tool calls or additional processing.

            All this does is add the prompt response message to the context builder
        """
        return self.context_builder.add_message(self.prompt_response.message)

__all__ = [
    UnresolvedResponse
]