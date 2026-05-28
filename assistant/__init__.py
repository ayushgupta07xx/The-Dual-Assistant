"""Dual-assistant package: a single assistant core with swappable backends."""
from .config import Settings, settings
from .core import Assistant, AssistantResponse, build_provider
from .observability import Observer

__all__ = [
    "Assistant",
    "AssistantResponse",
    "Settings",
    "settings",
    "Observer",
    "build_provider",
]
