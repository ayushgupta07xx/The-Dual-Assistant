"""Provider abstraction.

The Assistant talks to models only through this interface, so swapping the
open-source backend for the frontier one (or vice-versa) is a one-line config
change. Each provider fully encapsulates its own tool-calling mechanics:
the frontier provider uses Anthropic's native tool use, while the OSS provider
uses a ReAct-style text protocol (because a 0.5B model has no reliable native
tool calling). Both return the same `ChatResult`.
"""
from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional


@dataclass
class ChatResult:
    text: str
    input_tokens: int = 0
    output_tokens: int = 0
    model: str = ""
    provider: str = ""
    tool_trace: List[Dict[str, Any]] = field(default_factory=list)  # [{tool, args, result}]
    raw: Any = None


class Provider(ABC):
    name: str = "base"
    model: str = ""

    @abstractmethod
    def chat(
        self,
        system: str,
        messages: List[dict],
        tools: Optional["ToolRegistry"] = None,  # noqa: F821
        temperature: float = 0.7,
        max_tokens: int = 640,
    ) -> ChatResult:
        """Run one assistant turn over `messages`, optionally using tools."""
        raise NotImplementedError
