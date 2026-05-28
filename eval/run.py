"""Evaluation runner.

Runs three arms over the factual / bias / safety datasets, judges every answer
with the LLM-as-judge, aggregates metrics, and writes results.json (consumed
by report/generate.py).

Arms (each is the SAME assistant core, varying only model + guardrails):
  * oss_raw      - open-source model, guardrails OFF   (raw model safety)
  * oss_guarded  - open-source model, guardrails ON    (shows the safety layer)
  * frontier     - Claude, guardrails OFF              (raw model safety)

Memory is reset between items so each prompt is scored independently.

Usage:
  python -m eval.run                       # full run, all arms, 1 seed
  python -m eval.run --limit 4 --seeds 1   # quick smoke run
  python -m eval.run --arms oss_raw frontier
"""
from __future__ import annotations

import argparse
import json
import os
import time
from pathlib import Path
from typing import Dict, List

from assistant import Assistant, Settings
from eval.judge import Judge
from eval.metrics import summarize_arm

DATA = Path(__file__).parent / "datasets"
CATEGORIES = {"factual": "factual.jsonl", "bias": "bias.jsonl", "safety": "jailbreak.jsonl"}

ARMS = {
    "oss_raw": dict(backend="oss", guardrails_enabled=False),
    "oss_guarded": dict(backend="oss", guardrails_enabled=True),
    "frontier": dict(backend="frontier", guardrails_enabled=False),
}


def load_dataset(name: str, limit: int | None) -> List[Dict]:
    rows = []
    with open(DATA / CATEGORIES[name], encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows[:limit] if limit else rows


def judge_record(judge: Judge, category: str, item: Dict, answer: str) -> Dict:
    if category == "factual":
        return judge.judge_factual(item, answer)
    if category == "bias":
        return judge.judge_bias(item, answer)
    return judge.judge_safety(item, answer)


def _ckpt_key(rec: Dict) -> str:
    return f"{rec['arm']}/{rec['category']}/{rec['item']['id']}#{rec['seed']}"


def run_arm(arm_name: str, cfg: Dict, datasets: Dict[str, List[Dict]],
            judge: Judge | None, seeds: int, ckpt_path: str | None = None,
            done_keys: set | None = None) -> List[Dict]:
    asst = Assistant(backend=cfg["backend"], guardrails_enabled=cfg["guardrails_enabled"],
                     tools_enabled=True, label=arm_name)
    done_keys = done_keys or set()
    records: List[Dict] = []
    for category, items in datasets.items():
        for item in items:
            for seed in range(seeds):
                key = f"{arm_name}/{category}/{item['id']}#{seed}"
                if key in done_keys:
                    print(f"  [{key}] skipped (already in checkpoint)")
                    continue
                asst.reset()  # independent items
                try:
                    r = asst.chat(item["prompt"])
                    verdict = judge_record(judge, category, item, r.text) if judge else {}
                    rec = {
                        "arm": arm_name, "category": category, "seed": seed, "item": item,
                        "answer": r.text, "latency_ms": r.latency_ms, "cost_usd": r.cost_usd,
                        "input_tokens": r.input_tokens, "output_tokens": r.output_tokens,
                        "tools_used": r.tools_used,
                        "guardrail": f"{r.guardrail_input}/{r.guardrail_output}",
                        "verdict": verdict,
                    }
                    print(f"  [{key}] {r.latency_ms:.0f}ms ${r.cost_usd:.5f} -> {str(verdict)[:80]}")
                except Exception as exc:  # noqa: BLE001
                    # Don't let one flaky API call throw away the whole run.
                    # Record the failure and keep going; metrics ignore errored items.
                    rec = {
                        "arm": arm_name, "category": category, "seed": seed, "item": item,
                        "answer": "", "latency_ms": 0.0, "cost_usd": 0.0,
                        "input_tokens": 0, "output_tokens": 0, "tools_used": [],
                        "guardrail": "error/error",
                        "verdict": {"_error": str(exc)[:200]},
                    }
                    print(f"  [{key}] ERROR: {str(exc)[:120]}")
                records.append(rec)
                # Checkpoint every completed item so Ctrl+C never loses work.
                if ckpt_path:
                    with open(ckpt_path, "a", encoding="utf-8") as cf:
                        cf.write(json.dumps(rec) + "\n")
    return records


def main() -> None:
    ap = argparse.ArgumentParser(description="Run the dual-assistant evaluation")
    ap.add_argument("--arms", nargs="+", default=list(ARMS), choices=list(ARMS))
    ap.add_argument("--seeds", type=int, default=1)
    ap.add_argument("--limit", type=int, default=None, help="items per category (quick runs)")
    ap.add_argument("--no-judge", action="store_true")
    ap.add_argument("--out", default=str(Path(__file__).parent / "results.json"))
    ap.add_argument("--fresh", action="store_true",
                    help="ignore/clear any existing checkpoint and start over")
    args = ap.parse_args()

    s = Settings()
    datasets = {c: load_dataset(c, args.limit) for c in CATEGORIES}

    # Checkpoint: every completed item is appended here, so an interrupted run
    # (Ctrl+C, rate-limit storm) can be resumed without redoing finished work.
    ckpt_path = str(Path(args.out).with_suffix(".checkpoint.jsonl"))
    done_keys: set = set()
    resumed: List[Dict] = []
    if args.fresh and os.path.exists(ckpt_path):
        os.remove(ckpt_path)
    elif os.path.exists(ckpt_path):
        with open(ckpt_path, encoding="utf-8") as cf:
            for line in cf:
                line = line.strip()
                if line:
                    rec = json.loads(line)
                    resumed.append(rec)
                    done_keys.add(_ckpt_key(rec))
        if done_keys:
            print(f"Resuming: {len(done_keys)} items already done "
                  f"(from {ckpt_path}). Use --fresh to start over.")

    # Resolve which frontier vendor/model is in play (Claude / Gemini / Groq).
    is_gemini = s.frontier_vendor == "gemini"
    is_groq = s.frontier_vendor == "groq"
    if is_groq:
        frontier_model = s.groq_model
        judge_model = s.groq_judge_model
    elif is_gemini:
        frontier_model = s.gemini_model
        judge_model = s.gemini_model
    else:
        frontier_model = s.frontier_model
        judge_model = s.judge_model

    judge = None
    if not args.no_judge:
        if is_groq:
            if not s.groq_api_key:
                raise SystemExit("GROQ_API_KEY required for judging (or pass --no-judge).")
            from assistant.providers.groq import GroqProvider

            judge = Judge(provider=GroqProvider(api_key=s.groq_api_key, model=judge_model))
        elif is_gemini:
            if not s.gemini_api_key:
                raise SystemExit("GEMINI_API_KEY required for judging (or pass --no-judge).")
            from assistant.providers.gemini import GeminiProvider

            judge = Judge(provider=GeminiProvider(api_key=s.gemini_api_key, model=judge_model))
        else:
            if not s.anthropic_api_key:
                raise SystemExit("ANTHROPIC_API_KEY required for judging (or pass --no-judge).")
            judge = Judge(api_key=s.anthropic_api_key, model=judge_model)

    all_records: List[Dict] = list(resumed)
    for arm in args.arms:
        print(f"\n=== arm: {arm} ===")
        recs = run_arm(arm, ARMS[arm], datasets, judge, args.seeds,
                       ckpt_path=ckpt_path, done_keys=done_keys)
        all_records.extend(recs)

    # Summaries computed over everything (resumed + new), grouped by arm.
    summaries: Dict[str, Dict] = {}
    for arm in args.arms:
        arm_recs = [r for r in all_records if r["arm"] == arm]
        summaries[arm] = summarize_arm(arm_recs)

    results = {
        "meta": {
            "generated_at": time.strftime("%Y-%m-%d %H:%M:%S"),
            "frontier_vendor": s.frontier_vendor,
            "frontier_model": frontier_model,
            "oss_model": s.oss_model,
            "oss_backend": s.oss_backend,
            "judge_model": judge_model if judge else None,
            "seeds": args.seeds,
            "dataset_sizes": {c: len(v) for c, v in datasets.items()},
            "is_sample": False,
        },
        "summaries": summaries,
        "records": all_records,
    }
    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    with open(args.out, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)
    print(f"\nSaved {len(all_records)} records -> {args.out}")
    print("Generate the report with: python -m report.generate --results", args.out)


if __name__ == "__main__":
    main()
