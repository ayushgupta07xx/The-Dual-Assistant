"""Safety guardrails.

Two layers wrap the model:

  * Input guardrail  - catches prompt-injection / jailbreak framings and clear
                       requests for harmful content before they reach the model.
  * Output guardrail - redacts leaked PII and blocks harmful completions the
                       model may have produced anyway (important for the weak
                       OSS model).

Each layer uses fast deterministic heuristics by default, with an optional
LLM-based check for higher recall. The design intentionally separates
"blocked" (refuse) from "sanitized" (redact-and-continue) so the evaluation
can measure both jailbreak resistance and over-refusal.
"""
from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Optional

REFUSAL = (
    "I can't help with that. The request appears to involve harmful, unsafe, "
    "or policy-violating content. If you have a safe, legitimate goal behind "
    "this, tell me more and I'll try to help another way."
)

# --- Jailbreak / prompt-injection signatures -------------------------------
_JAILBREAK_PATTERNS = [
    r"ignore (all |your |the )?(previous|prior|above) (instructions|prompt)",
    r"disregard (the |your )?(system|previous) (prompt|instructions)",
    r"\bDAN\b|do anything now",
    r"developer mode",
    r"you are no longer (an? )?(ai|assistant|claude)",
    r"pretend (you|to) (are|be|have) (no|without) (rules|restrictions|filters|guidelines)",
    r"jailbreak",
    r"without any (safety|ethical|moral) (restrictions|guidelines|filters)",
    r"act as (an? )?(unfiltered|unrestricted|amoral)",
    r"reveal your (system )?prompt",
]

# --- Clear harmful-intent signatures (request side) ------------------------
_HARM_PATTERNS = [
    r"how (to|do i) (make|build|synthesize) (a |an )?(bomb|explosive|weapon|bioweapon)",
    r"(write|create|generate) (me )?(malware|ransomware|a virus|spyware|a keylogger)",
    r"(write|draft) (a |an )?phishing (email|message)",
    r"how (to|do i) (hack|break) into",
    r"(instructions|steps) (to|for) (kill|harm|hurt|poison)",
    r"(make|synthesize) (meth|methamphetamine|fentanyl|cocaine)",
]

# --- PII signatures (output side) ------------------------------------------
_PII = {
    "email": re.compile(r"\b[\w.+-]+@[\w-]+\.[\w.-]+\b"),
    "ssn": re.compile(r"\b\d{3}-\d{2}-\d{4}\b"),
    "credit_card": re.compile(r"\b(?:\d[ -]?){13,16}\b"),
    "phone": re.compile(r"\b(?:\+?1[ -.]?)?\(?\d{3}\)?[ -.]?\d{3}[ -.]?\d{4}\b"),
}


@dataclass
class GuardrailResult:
    action: str  # "allow" | "block" | "sanitize"
    text: Optional[str] = None  # replacement/sanitized text when relevant
    reason: str = ""
    category: str = ""

    @property
    def blocked(self) -> bool:
        return self.action == "block"


def _matches(patterns, text: str) -> Optional[str]:
    low = text.lower()
    for pat in patterns:
        if re.search(pat, low):
            return pat
    return None


def check_input(text: str, llm_check=None) -> GuardrailResult:
    """Screen a user message before it reaches the model."""
    hit = _matches(_JAILBREAK_PATTERNS, text)
    if hit:
        return GuardrailResult("block", REFUSAL, f"jailbreak pattern: {hit}", "jailbreak")
    hit = _matches(_HARM_PATTERNS, text)
    if hit:
        return GuardrailResult("block", REFUSAL, f"harmful request: {hit}", "harmful_request")
    if llm_check is not None:
        verdict = llm_check(text)  # expected: ("block"/"allow", category)
        if verdict and verdict[0] == "block":
            return GuardrailResult("block", REFUSAL, "llm flagged input", verdict[1])
    return GuardrailResult("allow")


def redact_pii(text: str) -> tuple[str, list[str]]:
    found = []
    for label, pat in _PII.items():
        if pat.search(text):
            found.append(label)
            text = pat.sub(f"[REDACTED_{label.upper()}]", text)
    return text, found


def check_output(text: str, llm_check=None) -> GuardrailResult:
    """Screen the model's completion before returning it to the user."""
    # 1) Block clearly harmful completions (mainly relevant for the OSS model).
    hit = _matches(_HARM_PATTERNS, text)
    if hit:
        return GuardrailResult("block", REFUSAL, f"harmful output: {hit}", "harmful_output")
    if llm_check is not None:
        verdict = llm_check(text)
        if verdict and verdict[0] == "block":
            return GuardrailResult("block", REFUSAL, "llm flagged output", verdict[1])
    # 2) Redact any leaked PII but otherwise allow.
    sanitized, found = redact_pii(text)
    if found:
        return GuardrailResult("sanitize", sanitized, f"redacted PII: {found}", "pii")
    return GuardrailResult("allow")


def make_llm_moderation(provider):
    """Build an optional LLM classifier using the active provider.

    Returns a callable(text) -> ("block"|"allow", category). Best-effort: any
    error falls back to "allow" so moderation never crashes the turn.
    """

    def _check(text: str):
        prompt = (
            "Classify whether the following text is a request for, or contains, "
            "seriously harmful content (violence enablement, weapons, malware, "
            "exploitation, self-harm instructions). Reply with one word: "
            "BLOCK or ALLOW, then a short category.\n\nTEXT:\n" + text[:2000]
        )
        try:
            res = provider.chat(
                system="You are a strict but precise content-safety classifier.",
                messages=[{"role": "user", "content": prompt}],
                tools=None,
                temperature=0.0,
                max_tokens=20,
            )
            verdict = res.text.strip().upper()
            return ("block" if verdict.startswith("BLOCK") else "allow", res.text.strip()[:40])
        except Exception:  # noqa: BLE001
            return ("allow", "")

    return _check
