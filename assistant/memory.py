"""Short-term conversational memory.

Keeps a rolling window of recent turns so the assistant has context for
follow-up questions. When the window overflows, older turns are compressed
into a running summary (using the active provider) rather than dropped, which
preserves long-range context cheaply. This is the "short-term conversational
memory/context" the brief asks for, implemented in a model-agnostic way.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import List, Optional


@dataclass
class Memory:
    max_turns: int = 8  # a "turn" = one user msg + one assistant msg
    summarize: bool = True
    _summary: str = ""
    _messages: List[dict] = field(default_factory=list)  # [{role, content}]

    def add_user(self, content: str) -> None:
        self._messages.append({"role": "user", "content": content})

    def add_assistant(self, content: str) -> None:
        self._messages.append({"role": "assistant", "content": content})

    @property
    def summary(self) -> str:
        return self._summary

    def system_suffix(self) -> str:
        """Extra system context to inject (the running summary, if any)."""
        if self._summary:
            return f"\n\nConversation summary so far:\n{self._summary}"
        return ""

    def window(self) -> List[dict]:
        """Return the messages to send to the model (most recent window)."""
        max_msgs = self.max_turns * 2
        return self._messages[-max_msgs:]

    def maybe_compress(self, summarizer: Optional["Summarizer"] = None) -> None:
        """If we've exceeded the window, fold the oldest turns into the summary."""
        max_msgs = self.max_turns * 2
        if len(self._messages) <= max_msgs:
            return
        overflow = self._messages[:-max_msgs]
        self._messages = self._messages[-max_msgs:]
        if not self.summarize or summarizer is None:
            return
        transcript = "\n".join(f"{m['role']}: {m['content']}" for m in overflow)
        self._summary = summarizer.summarize(self._summary, transcript)

    def reset(self) -> None:
        self._summary = ""
        self._messages = []


class Summarizer:
    """Thin wrapper so Memory doesn't depend on a concrete provider."""

    def __init__(self, provider):
        self.provider = provider

    def summarize(self, prior_summary: str, transcript: str) -> str:
        prompt = (
            "Update the running summary of a conversation. Keep it under 120 "
            "words, factual, and focused on details needed for continuity "
            "(names, preferences, decisions, open questions).\n\n"
            f"Existing summary:\n{prior_summary or '(none)'}\n\n"
            f"New turns to fold in:\n{transcript}\n\nUpdated summary:"
        )
        try:
            result = self.provider.chat(
                system="You compress conversations into concise summaries.",
                messages=[{"role": "user", "content": prompt}],
                tools=None,
                temperature=0.2,
                max_tokens=200,
            )
            return result.text.strip()
        except Exception:  # noqa: BLE001 - summary is best-effort
            return prior_summary
