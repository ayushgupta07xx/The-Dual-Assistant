"""Central configuration for the dual-assistant system.

Everything is driven by environment variables so the same code runs locally,
in CI, on Hugging Face Spaces, or on Modal without edits. Load order:
real environment > .env file (via python-dotenv) > defaults defined here.
"""
from __future__ import annotations

import os
from dataclasses import dataclass, field

try:  # optional: load a local .env if present
    from dotenv import load_dotenv

    load_dotenv()
except Exception:  # pragma: no cover - dotenv is optional
    pass


def _get_bool(name: str, default: bool) -> bool:
    val = os.getenv(name)
    if val is None:
        return default
    return val.strip().lower() in {"1", "true", "yes", "on"}


# --- System prompt shared by BOTH assistants -----------------------------
# Using one prompt for both backends is deliberate: the brief asks for "the
# same assistant experience with the same capabilities", so the only variable
# we change between runs is the underlying model. That keeps the evaluation a
# fair, controlled comparison.
SYSTEM_PROMPT = (
    "You are a helpful, honest, and harmless personal assistant. "
    "Answer concisely and accurately. If you are unsure or do not know "
    "something, say so plainly instead of inventing facts. You have access "
    "to tools; use them when they would improve accuracy (for example, use "
    "the calculator for arithmetic rather than guessing). Refuse requests "
    "that are illegal, dangerous, or designed to cause harm, and briefly "
    "explain why, offering a safer alternative when possible."
)


@dataclass
class Settings:
    # Which backend the Assistant should use: "frontier" or "oss".
    backend: str = field(default_factory=lambda: os.getenv("BACKEND", "frontier"))

    # --- Frontier --------------------------------------------------------
    # frontier_vendor selects WHICH frontier model the "frontier" backend uses:
    #   "anthropic" -> Claude (paid, needs ANTHROPIC_API_KEY)
    #   "gemini"    -> Google Gemini (free tier, needs GEMINI_API_KEY)
    #   "groq"      -> Groq / open models e.g. Llama 3.3 70B (free, needs GROQ_API_KEY)
    frontier_vendor: str = field(default_factory=lambda: os.getenv("FRONTIER_VENDOR", "anthropic"))

    # Anthropic (Claude)
    anthropic_api_key: str = field(default_factory=lambda: os.getenv("ANTHROPIC_API_KEY", ""))
    frontier_model: str = field(default_factory=lambda: os.getenv("FRONTIER_MODEL", "claude-sonnet-4-6"))

    # Google Gemini (free tier via Google AI Studio)
    gemini_api_key: str = field(default_factory=lambda: os.getenv("GEMINI_API_KEY", ""))
    gemini_model: str = field(default_factory=lambda: os.getenv("GEMINI_MODEL", "gemini-2.5-flash-lite"))

    # Groq (free tier, open models). Separate model for the judge keeps it a
    # different model from the frontier arm (less self-preference bias) and
    # spreads usage across per-model token quotas.
    groq_api_key: str = field(default_factory=lambda: os.getenv("GROQ_API_KEY", ""))
    groq_model: str = field(default_factory=lambda: os.getenv("GROQ_MODEL", "llama-3.3-70b-versatile"))
    groq_judge_model: str = field(default_factory=lambda: os.getenv("GROQ_JUDGE_MODEL", "llama-3.1-8b-instant"))

    # --- Open-source backend --------------------------------------------
    # oss_backend selects how the OSS model is served:
    #   "transformers" -> load weights locally with HF transformers
    #   "ollama"       -> talk to a local Ollama server
    #   "endpoint"     -> hit an OpenAI-compatible HTTP endpoint (e.g. Modal/vLLM)
    oss_backend: str = field(default_factory=lambda: os.getenv("OSS_BACKEND", "transformers"))
    oss_model: str = field(default_factory=lambda: os.getenv("OSS_MODEL", "Qwen/Qwen2.5-0.5B-Instruct"))
    ollama_host: str = field(default_factory=lambda: os.getenv("OLLAMA_HOST", "http://localhost:11434"))
    oss_endpoint_url: str = field(default_factory=lambda: os.getenv("OSS_ENDPOINT_URL", ""))
    oss_endpoint_key: str = field(default_factory=lambda: os.getenv("OSS_ENDPOINT_KEY", ""))
    # Per-second compute price used to estimate OSS cost when self-hosted on a
    # GPU (e.g. Modal). 0.0 means "free tier" (HF Spaces CPU) -> $0 marginal.
    oss_gpu_usd_per_sec: float = field(
        default_factory=lambda: float(os.getenv("OSS_GPU_USD_PER_SEC", "0.0"))
    )

    # --- Judge (for LLM-as-judge evaluation) ----------------------------
    judge_model: str = field(default_factory=lambda: os.getenv("JUDGE_MODEL", "claude-sonnet-4-6"))

    # --- Generation defaults --------------------------------------------
    temperature: float = field(default_factory=lambda: float(os.getenv("TEMPERATURE", "0.7")))
    max_tokens: int = field(default_factory=lambda: int(os.getenv("MAX_TOKENS", "640")))

    # --- Memory ----------------------------------------------------------
    memory_max_turns: int = field(default_factory=lambda: int(os.getenv("MEMORY_MAX_TURNS", "8")))
    memory_summarize: bool = field(default_factory=lambda: _get_bool("MEMORY_SUMMARIZE", True))

    # --- Guardrails ------------------------------------------------------
    guardrails_enabled: bool = field(default_factory=lambda: _get_bool("GUARDRAILS_ENABLED", True))
    guardrails_llm_check: bool = field(default_factory=lambda: _get_bool("GUARDRAILS_LLM_CHECK", False))

    # --- Tools -----------------------------------------------------------
    tools_enabled: bool = field(default_factory=lambda: _get_bool("TOOLS_ENABLED", True))

    # --- Observability ---------------------------------------------------
    log_dir: str = field(default_factory=lambda: os.getenv("LOG_DIR", "logs"))
    system_prompt: str = SYSTEM_PROMPT


settings = Settings()
