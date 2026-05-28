"""Frontier provider: Anthropic Claude with native tool use.

Implements a standard tool-use loop: send the conversation + tool specs, and
while Claude returns `tool_use` blocks, execute them and feed `tool_result`
blocks back until it produces a final text answer. Token counts come straight
from the API's usage object, so cost accounting is exact.
"""
from __future__ import annotations

from typing import List, Optional

from .base import ChatResult, Provider


class FrontierProvider(Provider):
    name = "frontier"

    def __init__(self, api_key: str, model: str = "claude-sonnet-4-6"):
        try:
            from anthropic import Anthropic
        except ImportError as exc:  # pragma: no cover
            raise ImportError(
                "The 'anthropic' package is required for the frontier provider. "
                "Install it with: pip install anthropic"
            ) from exc
        if not api_key:
            raise ValueError("ANTHROPIC_API_KEY is not set.")
        self.client = Anthropic(api_key=api_key)
        self.model = model

    def chat(
        self,
        system: str,
        messages: List[dict],
        tools=None,
        temperature: float = 0.7,
        max_tokens: int = 640,
    ) -> ChatResult:
        # Convert our simple {role, content:str} history into Anthropic format.
        convo: List[dict] = [{"role": m["role"], "content": m["content"]} for m in messages]
        tool_specs = tools.anthropic_specs() if tools else []
        tool_trace: List[dict] = []
        in_tokens = 0
        out_tokens = 0

        for _ in range(6):  # cap tool-use iterations to avoid runaway loops
            kwargs = dict(
                model=self.model,
                system=system,
                messages=convo,
                temperature=temperature,
                max_tokens=max_tokens,
            )
            if tool_specs:
                kwargs["tools"] = tool_specs
            resp = self.client.messages.create(**kwargs)
            in_tokens += resp.usage.input_tokens
            out_tokens += resp.usage.output_tokens

            # Collect any tool calls and any text from this response.
            tool_uses = [b for b in resp.content if getattr(b, "type", None) == "tool_use"]
            text_parts = [b.text for b in resp.content if getattr(b, "type", None) == "text"]

            if resp.stop_reason == "tool_use" and tool_uses and tools is not None:
                # Echo the assistant's tool-call turn back, then answer each tool.
                convo.append({"role": "assistant", "content": resp.content})
                results_block = []
                for tu in tool_uses:
                    out = tools.run(tu.name, dict(tu.input))
                    tool_trace.append({"tool": tu.name, "args": dict(tu.input), "result": out})
                    results_block.append(
                        {"type": "tool_result", "tool_use_id": tu.id, "content": out}
                    )
                convo.append({"role": "user", "content": results_block})
                continue

            return ChatResult(
                text="\n".join(text_parts).strip(),
                input_tokens=in_tokens,
                output_tokens=out_tokens,
                model=self.model,
                provider=self.name,
                tool_trace=tool_trace,
                raw=resp,
            )

        return ChatResult(
            text="(stopped: too many tool iterations)",
            input_tokens=in_tokens,
            output_tokens=out_tokens,
            model=self.model,
            provider=self.name,
            tool_trace=tool_trace,
        )
