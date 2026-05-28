"""Public demo — large open-source assistant (Llama 3.3 70B via Groq).

This Gradio Space calls Groq's OpenAI-compatible API instead of loading a model
locally, so it is fast (Groq's LPU hardware, ~300+ tok/s) and needs no GPU and
no heavy ML libraries. The GROQ_API_KEY is read from the Space's Secrets
(Settings -> Variables and secrets) and is NEVER hardcoded.

It applies the same lightweight input guardrails as the project's `oss_guarded`
configuration, screening jailbreak / prompt-injection / high-harm requests
before they reach the model.

This is the large open-source model (Llama 3.3 70B) from the dual-assistant
evaluation. The small local model (Llama 3.2 3B) and the full hallucination /
bias / safety evaluation harness live in the GitHub repo.
"""
from __future__ import annotations

import os
import re

import gradio as gr
import requests

GROQ_URL = "https://api.groq.com/openai/v1/chat/completions"
MODEL = os.environ.get("GROQ_MODEL", "llama-3.3-70b-versatile")
API_KEY = os.environ.get("GROQ_API_KEY", "")

SYSTEM_PROMPT = (
    "You are a helpful, honest, and harmless personal assistant. "
    "Answer concisely and accurately. If you are unsure or do not know "
    "something, say so plainly instead of inventing facts. Refuse requests "
    "that are illegal, dangerous, or designed to cause harm, and briefly "
    "explain why, offering a safer alternative when possible."
)

# --- Lightweight input guardrails (mirror assistant/guardrails.py) -----------
_JAILBREAK = re.compile(
    r"(ignore (all|previous|your) (instructions|rules)|"
    r"you are now|developer mode|do anything now|\bDAN\b|"
    r"pretend you have no (rules|guidelines)|bypass your)",
    re.IGNORECASE,
)
_HARMFUL = re.compile(
    r"(how to (make|build|synthesize).{0,40}(bomb|explosive|meth|nerve agent)|"
    r"untraceable (poison|weapon)|write malware|ransomware code)",
    re.IGNORECASE,
)
_REFUSAL = (
    "I can't help with that. It falls outside what I can safely assist with. "
    "If there's a safe, legitimate version of what you need, I'm glad to help "
    "with that instead."
)


def _blocked(text: str) -> bool:
    return bool(_JAILBREAK.search(text) or _HARMFUL.search(text))


def respond(message, history):
    """ChatInterface callback. Version-agnostic history handling (dict or tuple)."""
    if _blocked(message):
        return _REFUSAL
    if not API_KEY:
        return (
            "Server is missing its GROQ_API_KEY secret. Add it under the Space's "
            "Settings -> Variables and secrets, then restart the Space."
        )

    msgs = [{"role": "system", "content": SYSTEM_PROMPT}]
    for turn in history or []:
        if isinstance(turn, dict) and turn.get("content"):
            msgs.append({"role": turn.get("role", "user"), "content": turn["content"]})
        elif isinstance(turn, (list, tuple)) and len(turn) == 2:
            user_msg, bot_msg = turn
            if user_msg:
                msgs.append({"role": "user", "content": user_msg})
            if bot_msg:
                msgs.append({"role": "assistant", "content": bot_msg})
    msgs.append({"role": "user", "content": message})

    try:
        r = requests.post(
            GROQ_URL,
            headers={
                "Authorization": f"Bearer {API_KEY}",
                "Content-Type": "application/json",
            },
            json={
                "model": MODEL,
                "messages": msgs,
                "temperature": 0.7,
                "max_tokens": 640,
            },
            timeout=60,
        )
        if r.status_code == 429:
            return ("This free demo's shared quota is busy right now — "
                    "please try again in a minute.")
        r.raise_for_status()
        return r.json()["choices"][0]["message"]["content"].strip()
    except Exception as exc:  # noqa: BLE001
        return f"(Sorry — error contacting the model: {exc})"


demo = gr.ChatInterface(
    fn=respond,
    title="AI Personal Assistant — Llama 3.3 70B (via Groq)",
    description=(
        "Live demo of the large open-source assistant from a dual-assistant "
        "evaluation project. Runs Llama 3.3 70B on Groq's fast inference, with "
        "the same safety guardrails used in the benchmark. The small local model "
        "(Llama 3.2 3B) and the full hallucination / bias / safety evaluation "
        "live in the linked GitHub repo."
    ),
    examples=[
        "Explain the difference between TCP and UDP in two sentences.",
        "What is 18.5% of 240?",
        "Summarize the plot of Romeo and Juliet for a 10-year-old.",
    ],
)

if __name__ == "__main__":
    demo.queue().launch()
