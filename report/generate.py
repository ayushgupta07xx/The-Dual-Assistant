"""Build the one-page evaluation report (PDF) + PNGs for the README.

  python -m report.generate                         # uses eval/sample_results.json
  python -m report.generate --results eval/results.json

Outputs into report/:
  eval_report.pdf   - the 1-page infographic deliverable
  metrics.png       - the 2x2 metric panel (embedded in the README)
"""
from __future__ import annotations

import argparse
import json
import textwrap
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.gridspec import GridSpec  # noqa: E402

from report.charts import ARM_COLORS, ARM_LABELS, ARM_ORDER, grouped_metric  # noqa: E402

HERE = Path(__file__).parent


def _recommendation(summaries: dict) -> str:
    f = summaries.get("frontier", {})
    o = summaries.get("oss_raw", {})
    g = summaries.get("oss_guarded", {})
    bullets = [
        (
            f"Safety-critical or user-facing use \u2192 the frontier model: it cut "
            f"hallucination to ~{f.get('hallucination_rate',0):.0f}% and jailbreak success "
            f"to ~{f.get('jailbreak_success_rate',0):.0f}% with no measured over-refusal."
        ),
        (
            f"Cost/latency-sensitive, low-risk use \u2192 the open model is ~free to self-host, "
            f"but raw it complied with ~{o.get('jailbreak_success_rate',0):.0f}% of jailbreaks. "
            f"The guardrail layer dropped that to ~{g.get('jailbreak_success_rate',0):.0f}% "
            f"(small over-refusal cost of ~{g.get('over_refusal_rate',0):.0f}%)."
        ),
        (
            "Guardrails mitigate but do NOT fix hallucination/bias \u2014 those are model-quality "
            "limits. Treat the small model as 'cheap + fast + needs supervision'."
        ),
    ]
    lines = ["RECOMMENDATION"]
    for b in bullets:
        wrapped = textwrap.fill(b, width=58, subsequent_indent="  ")
        lines.append("\u2022 " + wrapped)
    return "\n".join(lines)


def _cost_table(ax, summaries: dict) -> None:
    ax.axis("off")
    ax.set_title("Cost & Latency", fontsize=11, fontweight="bold", loc="left", pad=6)
    headers = ["Arm", "Latency", "$ / turn", "$ / 1k turns"]
    rows = []
    present_arms = []
    for a in ARM_ORDER:
        if a not in summaries:
            continue
        present_arms.append(a)
        s = summaries[a]
        avg_cost = s["avg_cost_usd"]
        cost_turn = "$0 (free)" if avg_cost == 0 else f"${avg_cost:.5f}"
        cost_1k = "$0" if avg_cost == 0 else f"${avg_cost*1000:.2f}"
        rows.append([ARM_LABELS[a], f"{s['avg_latency_ms']:.0f} ms", cost_turn, cost_1k])
    tbl = ax.table(cellText=rows, colLabels=headers, loc="center", cellLoc="center",
                   colWidths=[0.38, 0.18, 0.22, 0.22])
    tbl.auto_set_font_size(False)
    tbl.set_fontsize(8.5)
    tbl.scale(1, 1.6)
    for (r, c), cell in tbl.get_celld().items():
        if r == 0:
            cell.set_facecolor("#264653")
            cell.set_text_props(color="white", fontweight="bold")
        elif c == 0:
            arm = present_arms[r - 1]
            cell.set_text_props(color=ARM_COLORS[arm], fontweight="bold", ha="left")
            cell.PAD = 0.04


def build(results_path: Path) -> Path:
    data = json.loads(results_path.read_text())
    meta = data["meta"]
    summaries = data["summaries"]

    # Label the frontier arm by whichever vendor actually ran, so a Gemini run
    # isn't mislabelled as Claude (and vice-versa).
    fm = str(meta.get("frontier_model", "")).lower()
    vendor = str(meta.get("frontier_vendor", "")).lower()
    if vendor == "groq" or fm.startswith("llama") or "groq" in fm:
        ARM_LABELS["frontier"] = "Frontier (Llama 70B)"
    elif vendor == "gemini" or fm.startswith("gemini"):
        ARM_LABELS["frontier"] = "Frontier (Gemini)"
    elif vendor == "anthropic" or fm.startswith("claude"):
        ARM_LABELS["frontier"] = "Frontier (Claude)"
    else:
        ARM_LABELS["frontier"] = "Frontier"

    fig = plt.figure(figsize=(8.27, 11.69))  # A4 portrait
    gs = GridSpec(4, 2, figure=fig, height_ratios=[0.7, 1.1, 1.1, 1.2],
                  hspace=0.55, wspace=0.25, left=0.08, right=0.95, top=0.95, bottom=0.05)

    # --- header ---
    head = fig.add_subplot(gs[0, :])
    head.axis("off")
    head.text(0, 0.85, "AI Assistant Evaluation \u2014 OSS vs Frontier",
              fontsize=17, fontweight="bold")
    head.text(0, 0.5,
              f"Frontier: {meta['frontier_model']}   |   OSS: {meta['oss_model']}   |   "
              f"Judge: {meta.get('judge_model')}",
              fontsize=8.5, color="#333")
    sizes = meta["dataset_sizes"]
    head.text(0, 0.22,
              f"Prompts: {sizes.get('factual')} factual, {sizes.get('bias')} bias, "
              f"{sizes.get('safety')} safety  |  seeds: {meta.get('seeds')}  |  "
              f"generated: {meta['generated_at']}",
              fontsize=8.5, color="#333")

    # --- 2x2 metric panel ---
    grouped_metric(fig.add_subplot(gs[1, 0]), summaries, "hallucination_rate",
                   "hallucination_ci", "Hallucination rate")
    grouped_metric(fig.add_subplot(gs[1, 1]), summaries, "bias_rate",
                   "bias_ci", "Bias / harmful-output rate")
    grouped_metric(fig.add_subplot(gs[2, 0]), summaries, "jailbreak_success_rate",
                   "jailbreak_ci", "Jailbreak success rate")
    grouped_metric(fig.add_subplot(gs[2, 1]), summaries, "over_refusal_rate",
                   "over_refusal_ci", "Over-refusal (on benign prompts)")

    # --- cost table + recommendation ---
    _cost_table(fig.add_subplot(gs[3, 0]), summaries)
    rec_ax = fig.add_subplot(gs[3, 1])
    rec_ax.axis("off")
    rec_ax.text(0, 1.0, _recommendation(summaries), fontsize=7.4, va="top",
                family="DejaVu Sans", linespacing=1.35,
                bbox=dict(boxstyle="round,pad=0.5", facecolor="#f1faee", edgecolor="#a8dadc"))

    if meta.get("is_sample"):
        fig.text(0.5, 0.5, "ILLUSTRATIVE SAMPLE DATA", fontsize=44, color="red",
                 alpha=0.12, ha="center", va="center", rotation=30, fontweight="bold")

    pdf_path = HERE / "eval_report.pdf"
    fig.savefig(pdf_path, format="pdf")

    # Standalone metric panel PNG for the README.
    fig2, axes = plt.subplots(2, 2, figsize=(10, 7))
    grouped_metric(axes[0, 0], summaries, "hallucination_rate", "hallucination_ci", "Hallucination rate")
    grouped_metric(axes[0, 1], summaries, "bias_rate", "bias_ci", "Bias / harmful-output rate")
    grouped_metric(axes[1, 0], summaries, "jailbreak_success_rate", "jailbreak_ci", "Jailbreak success rate")
    grouped_metric(axes[1, 1], summaries, "over_refusal_rate", "over_refusal_ci", "Over-refusal (benign)")
    fig2.suptitle("OSS vs Frontier \u2014 safety & quality metrics (lower is better)",
                  fontsize=13, fontweight="bold")
    fig2.tight_layout(rect=[0, 0, 1, 0.96])
    fig2.savefig(HERE / "metrics.png", dpi=130)

    plt.close("all")
    return pdf_path


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--results", default=str(Path(__file__).parents[1] / "eval" / "sample_results.json"))
    args = ap.parse_args()
    pdf = build(Path(args.results))
    print(f"Wrote {pdf} and {HERE / 'metrics.png'}")


if __name__ == "__main__":
    main()
