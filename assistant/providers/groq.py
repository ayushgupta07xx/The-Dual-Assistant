"""Frontier provider: Groq (free tier) via the OpenAI-compatible API.

Selected with FRONTIER_VENDOR=groq. Groq serves open-weight models (e.g.
Llama 3.3 70B) on its LPU hardware at high speed, with a no-credit-card free
tier. It speaks the OpenAI Chat Completions format, including native tool
calling, so this provider mirrors the Anthropic/Gemini ones and returns the
same `ChatResult`.

Free-tier notes (2026): 30 RPM, ~1,000 RPD, and a 6,000 tokens-per-minute cap
shared across the org. The TPM cap is the real constraint, so this provider
throttles requests (default ~8s gap) and retries on 429/5xx with backoff that
honours Groq's `retry-after` header.

Design choice for this project: the frontier ARM uses a large model
(llama-3.3-70b-versatile) while the JUDGE can use a smaller one
(llama-3.1-8b-instant) — different models, which also reduces the judge's
self-preference bias and spreads usage across model quotas.
"""
from __future__ import annotations

import os
import threading
import time
from typing import Any, Dict, List, Optional

import requests

from .base import ChatResult, Provider

_URL = "https://api.groq.com/openai/v1/chat/completions"

# Global throttle shared across all Groq calls (assistant + judge) to respect
# the org-level 6,000 TPM / 30 RPM free-tier limits. ~8s gap => ~7 req/min,
# comfortably under 30 RPM and gentle on the token-per-minute cap.
_ENV_INTERVAL = os.getenv("GROQ_MIN_INTERVAL_S")
_MIN_INTERVAL = float(_ENV_INTERVAL) if _ENV_INTERVAL else 8.0
_throttle_lock = threading.Lock()
_last_call = [0.0]


def _throttle() -> None:
    with _throttle_lock:
        now = time.monotonic()
        wait = _MIN_INTERVAL - (now - _last_call[0])
        if wait > 0:
            time.sleep(wait)
        _last_call[0] = time.monotonic()


class GroqProvider(Provider):
    # Named "frontier" so arm logic / cost branch / reporting treat it as the
    # frontier arm. The model field disambiguates in logs.
    name = "frontier"

    def __init__(self, api_key: str, model: str = "llama-3.3-70b-versatile"):
        if not api_key:
            raise ValueError("GROQ_API_KEY is not set.")
        self.api_key = api_key
        self.model = model

    def _post(self, payload: dict, max_retries: int = 6) -> dict:
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
        }
        delay = 8.0
        last_exc: Optional[Exception] = None
        last_status = None
        last_text = ""
        for _ in range(max_retries):
            _throttle()
            try:
                r = requests.post(_URL, headers=headers, json=payload, timeout=90)
            except requests.RequestException as exc:
                last_exc = exc
                time.sleep(delay)
                delay = min(delay * 2, 60)
                continue
            if r.status_code == 429 or r.status_code >= 500:
                last_status, last_text = r.status_code, r.text[:200]
                # Honour Retry-After header if present, else back off.
                ra = r.headers.get("retry-after")
                try:
                    wait = float(ra) if ra else delay
                except ValueError:
                    wait = delay
                time.sleep(max(wait, delay))
                delay = min(delay * 2, 90)
                continue
            if r.status_code >= 400:
                raise RuntimeError(f"Groq API {r.status_code}: {r.text[:300]}")
            return r.json()
        if last_exc:
            raise last_exc
        raise RuntimeError(
            f"Groq API: exhausted {max_retries} retries (last {last_status}: {last_text})"
        )

    def chat(
        self,
        system: str,
        messages: List[dict],
        tools=None,
        temperature: float = 0.7,
        max_tokens: int = 640,
    ) -> ChatResult:
        convo: List[dict] = [{"role": "system", "content": system}]
        for m in messages:
            convo.append({"role": m["role"], "content": m["content"]})

        tool_specs = self._openai_tools(tools) if tools else None
        tool_trace: List[dict] = []
        in_tok = out_tok = 0

        for _ in range(6):  # cap tool-use iterations
            payload: Dict[str, Any] = {
                "model": self.model,
                "messages": convo,
                "temperature": temperature,
                "max_tokens": max_tokens,
            }
            if tool_specs:
                payload["tools"] = tool_specs
                payload["tool_choice"] = "auto"

            data = self._post(payload)
            usage = data.get("usage", {}) or {}
            in_tok += usage.get("prompt_tokens", 0)
            out_tok += usage.get("completion_tokens", 0)

            choices = data.get("choices", [])
            if not choices:
                return ChatResult(text="", input_tokens=in_tok, output_tokens=out_tok,
                                  model=self.model, provider=self.name,
                                  tool_trace=tool_trace, raw=data)
            msg = choices[0].get("message", {}) or {}
            calls = msg.get("tool_calls") or []

            if calls and tools is not None:
                # Echo the assistant tool-call turn, then append a tool result
                # message per call (OpenAI format).
                convo.append({
                    "role": "assistant",
                    "content": msg.get("content") or "",
                    "tool_calls": calls,
                })
                import json as _json
                for c in calls:
                    fn = (c.get("function") or {})
                    name = fn.get("name", "")
                    try:
                        args = _json.loads(fn.get("arguments") or "{}")
                    except _json.JSONDecodeError:
                        args = {}
                    result = tools.run(name, args)
                    tool_trace.append({"tool": name, "args": args, "result": result})
                    convo.append({
                        "role": "tool",
                        "tool_call_id": c.get("id", ""),
                        "content": result,
                    })
                continue

            return ChatResult(
                text=(msg.get("content") or "").strip(),
                input_tokens=in_tok, output_tokens=out_tok,
                model=self.model, provider=self.name,
                tool_trace=tool_trace, raw=data,
            )

        return ChatResult(text="(stopped: too many tool iterations)",
                          input_tokens=in_tok, output_tokens=out_tok,
                          model=self.model, provider=self.name, tool_trace=tool_trace)

    @staticmethod
    def _openai_tools(tools) -> List[Dict[str, Any]]:
        specs = []
        for s in tools.anthropic_specs():
            specs.append({
                "type": "function",
                "function": {
                    "name": s["name"],
                    "description": s.get("description", ""),
                    "parameters": s.get("input_schema", {"type": "object", "properties": {}}),
                },
            })
        return specs
