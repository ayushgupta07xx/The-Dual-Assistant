"""The Assistant: one orchestrator, two interchangeable brains.

Pipeline for every turn:

    user text
      -> input guardrail (jailbreak / harmful-request screen)
      -> memory (rolling window + running summary)
      -> provider.chat (frontier OR oss, with tools)
      -> output guardrail (harmful-output block / PII redaction)
      -> observability log (latency, tokens, cost, tools, guardrail hits)
      -> response

Because the provider is the only thing that changes between the two
assistants, the *experience and capabilities are identical* -- which is
exactly what the brief asks for and what makes the comparison fair.
"""
from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import List, Optional

from .config import Settings, settings as default_settings
from .guardrails import check_input, check_output, make_llm_moderation
from .memory import Memory, Summarizer
from .observability import (
    Observer,
    TurnRecord,
    frontier_cost,
    new_conversation_id,
    oss_cost,
)
from .providers.base import Provider
from .providers.frontier import FrontierProvider
from .providers.oss import OSSProvider
from .tools import ToolRegistry


def build_frontier_provider(s: Settings) -> Provider:
    """Build the frontier provider for the configured vendor (Claude or Gemini)."""
    if s.frontier_vendor == "gemini":
        from .providers.gemini import GeminiProvider

        return GeminiProvider(api_key=s.gemini_api_key, model=s.gemini_model)
    if s.frontier_vendor == "groq":
        from .providers.groq import GroqProvider

        return GroqProvider(api_key=s.groq_api_key, model=s.groq_model)
    return FrontierProvider(api_key=s.anthropic_api_key, model=s.frontier_model)


def build_provider(s: Settings, backend: str) -> Provider:
    if backend == "frontier":
        return build_frontier_provider(s)
    return OSSProvider(
        backend=s.oss_backend,
        model=s.oss_model,
        ollama_host=s.ollama_host,
        endpoint_url=s.oss_endpoint_url,
        endpoint_key=s.oss_endpoint_key,
    )


@dataclass
class AssistantResponse:
    text: str
    provider: str
    model: str
    latency_ms: float
    input_tokens: int
    output_tokens: int
    cost_usd: float
    tools_used: List[str] = field(default_factory=list)
    guardrail_input: str = "allow"
    guardrail_output: str = "allow"
    blocked: bool = False


class Assistant:
    def __init__(
        self,
        s: Optional[Settings] = None,
        backend: Optional[str] = None,
        guardrails_enabled: Optional[bool] = None,
        tools_enabled: Optional[bool] = None,
        observer: Optional[Observer] = None,
        label: Optional[str] = None,
    ):
        self.s = s or default_settings
        self.backend = backend or self.s.backend
        self.guardrails_enabled = (
            self.s.guardrails_enabled if guardrails_enabled is None else guardrails_enabled
        )
        self.tools_enabled = self.s.tools_enabled if tools_enabled is None else tools_enabled
        self.label = label or f"{self.backend}{'+guard' if self.guardrails_enabled else ''}"

        self.provider = build_provider(self.s, self.backend)
        self.tools = ToolRegistry() if self.tools_enabled else None
        self.memory = Memory(max_turns=self.s.memory_max_turns, summarize=self.s.memory_summarize)
        self.summarizer = Summarizer(self.provider) if self.s.memory_summarize else None
        self.observer = observer or Observer(self.s.log_dir)
        self.conversation_id = new_conversation_id()
        # Optional LLM moderation reuses the active provider.
        self._llm_mod = (
            make_llm_moderation(self.provider)
            if (self.guardrails_enabled and self.s.guardrails_llm_check)
            else None
        )

    def reset(self) -> None:
        self.memory.reset()
        self.conversation_id = new_conversation_id()

    def chat(self, user_message: str) -> AssistantResponse:
        start = time.perf_counter()
        self.memory.add_user(user_message)
        gi_action = "allow"
        go_action = "allow"
        error = ""
        tools_used: List[str] = []

        # --- input guardrail ---
        if self.guardrails_enabled:
            gi = check_input(user_message, self._llm_mod)
            gi_action = gi.action
            if gi.blocked:
                text = gi.text or ""
                self.memory.add_assistant(text)
                latency_ms = (time.perf_counter() - start) * 1000
                resp = AssistantResponse(
                    text=text, provider=self.provider.name, model=self.provider.model,
                    latency_ms=latency_ms, input_tokens=0, output_tokens=0, cost_usd=0.0,
                    guardrail_input=gi_action, blocked=True,
                )
                self._log(user_message, resp, error)
                return resp

        # --- generate ---
        system = self.s.system_prompt + self.memory.system_suffix()
        try:
            result = self.provider.chat(
                system=system,
                messages=self.memory.window(),
                tools=self.tools,
                temperature=self.s.temperature,
                max_tokens=self.s.max_tokens,
            )
            text = result.text
            in_tok, out_tok = result.input_tokens, result.output_tokens
            tools_used = [t["tool"] for t in result.tool_trace]
        except Exception as exc:  # noqa: BLE001
            error = str(exc)
            text = f"(provider error: {exc})"
            in_tok = out_tok = 0

        # --- output guardrail ---
        if self.guardrails_enabled and not error:
            go = check_output(text, self._llm_mod)
            go_action = go.action
            if go.action in {"block", "sanitize"}:
                text = go.text or text

        latency_ms = (time.perf_counter() - start) * 1000
        if self.provider.name == "frontier":
            cost = frontier_cost(self.provider.model, in_tok, out_tok)
        else:
            cost = oss_cost(latency_ms / 1000.0, self.s.oss_gpu_usd_per_sec)

        self.memory.add_assistant(text)
        self.memory.maybe_compress(self.summarizer)

        resp = AssistantResponse(
            text=text, provider=self.provider.name, model=self.provider.model,
            latency_ms=latency_ms, input_tokens=in_tok, output_tokens=out_tok,
            cost_usd=cost, tools_used=tools_used, guardrail_input=gi_action,
            guardrail_output=go_action, blocked=False,
        )
        self._log(user_message, resp, error)
        return resp

    def _log(self, user_message: str, resp: AssistantResponse, error: str) -> None:
        self.observer.log(
            TurnRecord(
                conversation_id=self.conversation_id,
                provider=resp.provider, model=resp.model,
                user_message=user_message, assistant_message=resp.text,
                latency_ms=resp.latency_ms, input_tokens=resp.input_tokens,
                output_tokens=resp.output_tokens, cost_usd=resp.cost_usd,
                tools_used=resp.tools_used, guardrail_input=resp.guardrail_input,
                guardrail_output=resp.guardrail_output, error=error,
            )
        )
