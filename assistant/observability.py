"""Observability and cost accounting.

Every assistant turn is recorded with latency, token usage, computed cost, the
tools it called, and whether any guardrail fired. Records go to both a JSONL
file (easy to grep / stream) and a SQLite database (easy to aggregate for the
cost+latency table and the evaluation report). This is the backbone for the
"observability/evals" and "cost + latency table" bonus items.
"""
from __future__ import annotations

import json
import os
import sqlite3
import time
import uuid
from dataclasses import asdict, dataclass, field
from typing import List, Optional

# Anthropic API list prices, USD per million tokens (input, output).
# Source: Anthropic pricing, verified May 2026.
FRONTIER_PRICING = {
    "claude-opus-4-7": (5.0, 25.0),
    "claude-sonnet-4-6": (3.0, 15.0),
    "claude-haiku-4-5": (1.0, 5.0),
}


def frontier_cost(model: str, in_tok: int, out_tok: int) -> float:
    if model.startswith("gemini") or model.startswith("llama") or "groq" in model:
        return 0.0  # free tiers (Google AI Studio / Groq): no marginal cost
    base = model.split("-2")[0]  # tolerate dated suffixes like -20251001
    inp, outp = FRONTIER_PRICING.get(base, FRONTIER_PRICING["claude-sonnet-4-6"])
    return (in_tok / 1e6) * inp + (out_tok / 1e6) * outp


def oss_cost(latency_s: float, usd_per_sec: float) -> float:
    """Self-hosted OSS cost is wall-clock GPU time * the platform's rate.

    On a free CPU tier (HF Spaces) usd_per_sec is 0 -> $0 marginal cost.
    """
    return latency_s * usd_per_sec


@dataclass
class TurnRecord:
    conversation_id: str
    provider: str
    model: str
    user_message: str
    assistant_message: str
    latency_ms: float
    input_tokens: int
    output_tokens: int
    cost_usd: float
    tools_used: List[str] = field(default_factory=list)
    guardrail_input: str = "allow"
    guardrail_output: str = "allow"
    error: str = ""
    ts: float = field(default_factory=time.time)


class Observer:
    def __init__(self, log_dir: str = "logs"):
        os.makedirs(log_dir, exist_ok=True)
        self.jsonl_path = os.path.join(log_dir, "turns.jsonl")
        self.db_path = os.path.join(log_dir, "turns.db")
        self._init_db()

    def _init_db(self) -> None:
        con = sqlite3.connect(self.db_path)
        con.execute(
            """CREATE TABLE IF NOT EXISTS turns (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                ts REAL, conversation_id TEXT, provider TEXT, model TEXT,
                user_message TEXT, assistant_message TEXT, latency_ms REAL,
                input_tokens INT, output_tokens INT, cost_usd REAL,
                tools_used TEXT, guardrail_input TEXT, guardrail_output TEXT, error TEXT
            )"""
        )
        con.commit()
        con.close()

    def log(self, rec: TurnRecord) -> None:
        with open(self.jsonl_path, "a", encoding="utf-8") as f:
            f.write(json.dumps(asdict(rec)) + "\n")
        con = sqlite3.connect(self.db_path)
        con.execute(
            """INSERT INTO turns (ts, conversation_id, provider, model, user_message,
                assistant_message, latency_ms, input_tokens, output_tokens, cost_usd,
                tools_used, guardrail_input, guardrail_output, error)
               VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?)""",
            (
                rec.ts, rec.conversation_id, rec.provider, rec.model, rec.user_message,
                rec.assistant_message, rec.latency_ms, rec.input_tokens, rec.output_tokens,
                rec.cost_usd, json.dumps(rec.tools_used), rec.guardrail_input,
                rec.guardrail_output, rec.error,
            ),
        )
        con.commit()
        con.close()

    def summary_by_provider(self) -> list[dict]:
        """Aggregate latency/token/cost stats -> feeds the cost+latency table."""
        con = sqlite3.connect(self.db_path)
        con.row_factory = sqlite3.Row
        rows = con.execute(
            """SELECT provider, model, COUNT(*) n,
                      AVG(latency_ms) avg_latency_ms,
                      AVG(input_tokens) avg_in, AVG(output_tokens) avg_out,
                      SUM(cost_usd) total_cost, AVG(cost_usd) avg_cost
               FROM turns GROUP BY provider, model"""
        ).fetchall()
        con.close()
        return [dict(r) for r in rows]


def new_conversation_id() -> str:
    return uuid.uuid4().hex[:12]
