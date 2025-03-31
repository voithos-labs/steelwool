from enum import Enum

class MessageRole(Enum):
    USER = "user"
    MODEL = "model"
    FUNCTION = "function"
    SYSTEM = "system"
    TOOL = "tool"

class ContentType(Enum):
    TEXT = "text"

class StopReason(Enum):
    STOP = "stop"
    LENGTH = "length"
    CONTENT_FILTER = "content_filter"
    TOOL_CALLS = "tool_calls"
    NULL = "null"

__all__ = [
    MessageRole,
    ContentType,
    StopReason
]