"""Frontier provider: Google Gemini (free tier) via the REST API.

A drop-in alternative to the Anthropic frontier provider, selected with
FRONTIER_VENDOR=gemini. It uses Gemini's native function calling for tools and
mirrors the Anthropic provider's tool-use loop, returning the same `ChatResult`
so nothing downstream changes.

Design notes for the free tier:
  * Auth uses the `x-goog-api-key` header (not a URL query param), so the key
    never lands in a URL/log.
  * The free tier is rate-limited (~15 requests/min), so every call retries with
    exponential backoff on HTTP 429. This makes the eval self-throttle instead
    of crashing with rate-limit errors.
  * If a request fails with 400 while tools are attached (e.g. a schema quirk),
    it retries once WITHOUT tools so the arm still produces an answer rather
    than failing outright.
  * Cost is $0 on the free tier (see observability.frontier_cost), but token
    counts are still read from usageMetadata for the report.
"""
from __future__ import annotations

import os
import re
import threading
import time
from typing import Any, Dict, List, Optional

import requests

from .base import ChatResult, Provider

_BASE = "https://generativelanguage.googleapis.com/v1beta/models"

# Global throttle shared across ALL GeminiProvider instances (assistant + judge),
# so the COMBINED request rate stays under the free-tier per-minute limit.
# Free-tier RPM: Flash = 10, Flash-Lite = 15. We default to a conservative
# 6.5s gap (~9 req/min) which is safe under BOTH, and let a provider tighten it
# to 4.5s when it knows the model is Flash-Lite. An explicit env var always wins.
_ENV_INTERVAL = os.getenv("GEMINI_MIN_INTERVAL_S")
_MIN_INTERVAL = float(_ENV_INTERVAL) if _ENV_INTERVAL else 6.5
_throttle_lock = threading.Lock()
_last_call = [0.0]


def _set_min_interval(seconds: float) -> None:
    """Let a provider adjust the global pace (only if no env override is set)."""
    global _MIN_INTERVAL
    if not _ENV_INTERVAL:
        _MIN_INTERVAL = seconds


def _parse_retry_delay(body_text: str) -> Optional[float]:
    """Pull Google's suggested retryDelay (e.g. '27s') out of a 429 body."""
    m = re.search(r'"retryDelay"\s*:\s*"(\d+(?:\.\d+)?)s"', body_text)
    return float(m.group(1)) if m else None


def _throttle() -> None:
    with _throttle_lock:
        now = time.monotonic()
        wait = _MIN_INTERVAL - (now - _last_call[0])
        if wait > 0:
            time.sleep(wait)
        _last_call[0] = time.monotonic()


def _upper_types(schema: Any) -> Any:
    """Gemini's schema enum expects upper-case types (STRING/OBJECT/...)."""
    if isinstance(schema, dict):
        out = {}
        for k, v in schema.items():
            if k == "type" and isinstance(v, str):
                out[k] = v.upper()
            else:
                out[k] = _upper_types(v)
        return out
    if isinstance(schema, list):
        return [_upper_types(x) for x in schema]
    return schema


def _to_gemini_tools(tools) -> List[Dict[str, Any]]:
    decls = []
    for spec in tools.anthropic_specs():
        decls.append(
            {
                "name": spec["name"],
                "description": spec.get("description", ""),
                "parameters": _upper_types(
                    spec.get("input_schema", {"type": "object", "properties": {}})
                ),
            }
        )
    return [{"functionDeclarations": decls}]


class GeminiProvider(Provider):
    # Named "frontier" so the rest of the system (arm logic, cost branch,
    # reporting) treats it exactly like the frontier arm. The `model` field
    # (e.g. "gemini-2.5-flash") is what disambiguates it in logs.
    name = "frontier"

    def __init__(self, api_key: str, model: str = "gemini-2.5-flash"):
        if not api_key:
            raise ValueError("GEMINI_API_KEY is not set.")
        self.api_key = api_key
        self.model = model
        # Flash-Lite allows 15 RPM, so a 4.5s gap (~13/min) is safe; everything
        # else (Flash=10 RPM) keeps the conservative 6.5s default.
        if "lite" in model.lower():
            _set_min_interval(4.5)

    def _post(self, body: dict, max_retries: int = 6) -> dict:
        url = f"{_BASE}/{self.model}:generateContent"
        headers = {"x-goog-api-key": self.api_key, "Content-Type": "application/json"}
        delay = 3.0
        last_exc: Optional[Exception] = None
        last_status = None
        last_text = ""
        for attempt in range(max_retries):
            _throttle()  # stay under the per-minute free-tier limit
            try:
                r = requests.post(url, headers=headers, json=body, timeout=90)
            except requests.RequestException as exc:  # network blip -> retry
                last_exc = exc
                time.sleep(delay)
                delay = min(delay * 2, 60)
                continue
            # Retry transient conditions: 429 (rate limit) and any 5xx
            # (500/502/503/504 = server overloaded / temporarily unavailable).
            if r.status_code == 429:
                # Respect Google's own suggested retry delay when present;
                # otherwise wait a conservative window. Rejected calls can
                # re-consume quota, so we wait rather than hammer.
                last_status, last_text = r.status_code, r.text[:200]
                suggested = _parse_retry_delay(r.text)
                time.sleep(suggested + 1.0 if suggested else max(delay, 30.0))
                delay = min(delay * 2, 90)
                continue
            if r.status_code >= 500:
                last_status, last_text = r.status_code, r.text[:200]
                time.sleep(delay)
                delay = min(delay * 2, 60)
                continue
            if r.status_code == 400:
                # Surface as an exception the caller can choose to handle
                # (used to retry without tools).
                raise _BadRequest(r.text)
            r.raise_for_status()
            return r.json()
        if last_exc:
            raise last_exc
        raise RuntimeError(
            f"Gemini API: exhausted {max_retries} retries "
            f"(last status {last_status}: {last_text})"
        )

    def chat(
        self,
        system: str,
        messages: List[dict],
        tools=None,
        temperature: float = 0.7,
        max_tokens: int = 640,
    ) -> ChatResult:
        # Map our {user, assistant} history to Gemini's {user, model} roles.
        contents: List[dict] = []
        for m in messages:
            role = "model" if m["role"] == "assistant" else "user"
            contents.append({"role": role, "parts": [{"text": m["content"]}]})

        gen_cfg = {"temperature": temperature, "maxOutputTokens": max_tokens}
        use_tools = tools is not None
        tool_trace: List[dict] = []
        in_tok = out_tok = 0

        for _ in range(6):  # cap tool-use iterations
            body: Dict[str, Any] = {
                "systemInstruction": {"parts": [{"text": system}]},
                "contents": contents,
                "generationConfig": gen_cfg,
            }
            if use_tools:
                body["tools"] = _to_gemini_tools(tools)

            try:
                data = self._post(body)
            except _BadRequest:
                if use_tools:  # a tool-schema quirk: drop tools and try again
                    use_tools = False
                    continue
                raise

            usage = data.get("usageMetadata", {})
            in_tok += usage.get("promptTokenCount", 0)
            out_tok += usage.get("candidatesTokenCount", 0)

            candidates = data.get("candidates", [])
            if not candidates:
                return ChatResult(
                    text="", input_tokens=in_tok, output_tokens=out_tok,
                    model=self.model, provider=self.name, tool_trace=tool_trace, raw=data,
                )
            parts = (candidates[0].get("content", {}) or {}).get("parts", []) or []
            text_parts = [p["text"] for p in parts if "text" in p]
            fcalls = [p["functionCall"] for p in parts if "functionCall" in p]

            if fcalls and tools is not None:
                # Echo the model's tool-call turn, then send tool results back.
                contents.append({"role": "model", "parts": parts})
                resp_parts = []
                for fc in fcalls:
                    fname = fc.get("name", "")
                    fargs = dict(fc.get("args", {}) or {})
                    result = tools.run(fname, fargs)
                    tool_trace.append({"tool": fname, "args": fargs, "result": result})
                    resp_parts.append(
                        {"functionResponse": {"name": fname, "response": {"result": result}}}
                    )
                contents.append({"role": "user", "parts": resp_parts})
                continue

            return ChatResult(
                text="\n".join(text_parts).strip(),
                input_tokens=in_tok, output_tokens=out_tok,
                model=self.model, provider=self.name, tool_trace=tool_trace, raw=data,
            )

        return ChatResult(
            text="(stopped: too many tool iterations)",
            input_tokens=in_tok, output_tokens=out_tok,
            model=self.model, provider=self.name, tool_trace=tool_trace,
        )


class _BadRequest(Exception):
    """Internal: HTTP 400 from the Gemini API (often a tool-schema issue)."""
