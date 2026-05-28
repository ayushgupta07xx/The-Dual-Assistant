"""Reusable chart helpers for the evaluation report."""
from __future__ import annotations

from typing import Dict, List

import matplotlib.pyplot as plt

# Consistent arm styling across every figure.
ARM_ORDER = ["oss_raw", "oss_guarded", "frontier"]
ARM_LABELS = {"oss_raw": "OSS (raw)", "oss_guarded": "OSS + guardrails", "frontier": "Frontier (Claude)"}
ARM_COLORS = {"oss_raw": "#d1495b", "oss_guarded": "#edae49", "frontier": "#2a9d8f"}


def grouped_metric(ax, summaries: Dict, rate_key: str, ci_key: str, title: str) -> None:
    """Draw one 'lower is better' metric as a labelled bar chart with 95% CI."""
    arms = [a for a in ARM_ORDER if a in summaries]
    vals = [summaries[a][rate_key] for a in arms]
    labels = [ARM_LABELS[a] for a in arms]
    colors = [ARM_COLORS[a] for a in arms]

    # Asymmetric error bars from the Wilson interval.
    yerr_low, yerr_high = [], []
    for a, v in zip(arms, vals):
        lo, hi = summaries[a].get(ci_key, (v, v))
        yerr_low.append(max(0.0, v - lo))
        yerr_high.append(max(0.0, hi - v))

    x = range(len(arms))
    bars = ax.bar(x, vals, color=colors, width=0.6,
                  yerr=[yerr_low, yerr_high], capsize=4, error_kw={"elinewidth": 1, "alpha": 0.6})
    ax.set_title(title, fontsize=11, fontweight="bold", pad=8)
    ax.set_xticks(list(x))
    ax.set_xticklabels(labels, fontsize=8)
    ax.set_ylim(0, max(100, max(vals) * 1.25 if vals else 100))
    ax.set_ylabel("%", fontsize=8)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.tick_params(axis="y", labelsize=7)
    for b, v in zip(bars, vals):
        ax.text(b.get_x() + b.get_width() / 2, v + 2, f"{v:.0f}%", ha="center",
                va="bottom", fontsize=8, fontweight="bold")
    ax.text(0.99, 0.97, "lower is better", transform=ax.transAxes, ha="right",
            va="top", fontsize=6.5, style="italic", color="#666")
