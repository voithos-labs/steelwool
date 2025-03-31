from pydantic import BaseModel, Field
from typing import (
    AsyncIterator, 
    List, 
    Optional, 
    Any, 
    Dict, 
    Awaitable, 
    Callable, 
    TYPE_CHECKING
)
from ._enums import (
    MessageRole, 
    ContentType, 
    StopReason
)

if TYPE_CHECKING:
    from ._context_builder import ContextBuilder

# -------------------- Shorthand Types --------------------

# Translates and attempts to complete a conversation had between a model and a user actor
ProviderAdapter = Callable[
    ["ContextBuilder", str, int, Optional[List["Tool"]]], 
    Awaitable["PromptResponse"]
]

# Like above, but works with an async iterator (i.e., a yield-er thing)
StreamProviderAdapter = Callable[
    ["ContextBuilder", str, int, Optional[List["Tool"]]], 
    AsyncIterator["PromptResponseDelta"]
]

# Takes a tool call and resolves it
ToolResolver = Callable[["ToolCall"], Awaitable[str]]

# -------------------- Messages --------------------

# Represents a single message from an actor in a conversation
class Message(BaseModel):
    role: MessageRole
    content: str
    content_type: ContentType

# Represents a full response
class PromptResponse(BaseModel):
    message: Message
    stop_reason: StopReason
    token_usage: int
    tool_calls: Optional["ToolCall"] = Field(default=None)

# Represents a chunk of a streaming response
class PromptResponseDelta(BaseModel):
    content : str
    stop_reason: Optional[StopReason]
    tool_call: Optional["ToolCall"]

# -------------------- Tools --------------------

# Defines and describes a tool
class Tool(BaseModel):
    name: str
    description: str
    schema: dict
    required: bool

# A call to... call a tool!
class ToolCall(BaseModel):
    id: str
    name: str
    arguments: Dict[str, Any]

__all__ = [
    ProviderAdapter,
    StreamProviderAdapter,
    ToolResolver,

    Message,
    PromptResponse,
    PromptResponseDelta,

    Tool,
    ToolCall
]