"""Generate ILLUSTRATIVE sample results so the report renders before a real run.

These numbers are *not measured* -- they encode the qualitative pattern you
should expect (a 0.5B open model hallucinates and complies with jailbreaks far
more than a frontier model; guardrails sharply cut OSS jailbreak success at a
small over-refusal cost). Replace by running `python -m eval.run`, which writes
results.json with is_sample=false. The report stamps a watermark whenever
is_sample is true so nobody mistakes these for real measurements.
"""
from __future__ import annotations

import json
from pathlib import Path

from eval.metrics import wilson_ci

COUNTS = {"factual": 15, "bias": 12, "safety_harmful": 10, "safety_benign": 6}


def arm(hall, incorrect, biased, complied, over_ref, lat, avg_in, avg_out, avg_cost, total_cost):
    nf, nb, nh, ng = COUNTS["factual"], COUNTS["bias"], COUNTS["safety_harmful"], COUNTS["safety_benign"]
    return {
        "n_total": nf + nb + nh + ng,
        "hallucination_rate": hall / nf * 100, "hallucination_ci": wilson_ci(hall, nf),
        "incorrect_rate": incorrect / nf * 100,
        "bias_rate": biased / nb * 100, "bias_ci": wilson_ci(biased, nb),
        "jailbreak_success_rate": complied / nh * 100, "jailbreak_ci": wilson_ci(complied, nh),
        "over_refusal_rate": over_ref / ng * 100, "over_refusal_ci": wilson_ci(over_ref, ng),
        "avg_latency_ms": lat, "avg_cost_usd": avg_cost, "total_cost_usd": total_cost,
        "avg_input_tokens": avg_in, "avg_output_tokens": avg_out, "counts": COUNTS,
    }


def main() -> None:
    summaries = {
        # hall, incorrect, biased, complied, over_ref, latency, in, out, avg_cost, total_cost
        "oss_raw": arm(8, 9, 5, 7, 0, 3800, 265, 95, 0.0, 0.0),
        "oss_guarded": arm(8, 9, 5, 2, 1, 3870, 265, 92, 0.0, 0.0),
        "frontier": arm(1, 2, 1, 0, 0, 1600, 232, 121, 0.00249, 0.107),
    }
    results = {
        "meta": {
            "generated_at": "ILLUSTRATIVE SAMPLE",
            "frontier_model": "claude-sonnet-4-6",
            "oss_model": "Qwen/Qwen2.5-0.5B-Instruct",
            "oss_backend": "transformers (HF Spaces free CPU)",
            "judge_model": "claude-sonnet-4-6",
            "seeds": 1,
            "dataset_sizes": {"factual": 15, "bias": 12, "safety": 16},
            "is_sample": True,
        },
        "summaries": summaries,
        "records": [],
    }
    out = Path(__file__).parent / "sample_results.json"
    out.write_text(json.dumps(results, indent=2))
    print(f"Wrote {out}")


if __name__ == "__main__":
    main()
