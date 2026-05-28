---
title: Open-Source Personal Assistant (Qwen2.5-0.5B)
emoji: 🤖
colorFrom: indigo
colorTo: green
sdk: gradio
sdk_version: 4.44.0
app_file: app.py
pinned: false
license: apache-2.0
short_description: OSS arm of a dual-assistant hallucination/bias/safety eval
---

# Open-Source Personal Assistant — Qwen2.5-0.5B-Instruct

This Space is the **publicly deployed open-source arm** of a dual-assistant
evaluation project. It runs `Qwen/Qwen2.5-0.5B-Instruct` on the free CPU tier
with the same system prompt and the same lightweight safety guardrails used in
the benchmarked `oss_guarded` configuration.

The companion frontier assistant (Claude) and the full evaluation harness —
measuring **hallucination rate**, **bias rate**, and **content-safety /
jailbreak resistance** with confidence intervals — live in the GitHub repo.

## Why 0.5B on free CPU?
The free Spaces tier is CPU-only with limited RAM. The 0.5B Qwen is the largest
variant that loads and streams at an acceptable latency here. For a larger model
(e.g. Qwen2.5-7B-Instruct) the project ships a **Modal** deployment that exposes
a GPU-backed, OpenAI-compatible endpoint; the main app targets it with
`OSS_BACKEND=endpoint`.

## Safety note
Inputs matching known jailbreak / prompt-injection patterns or high-harm
requests are refused before generation. This mirrors the project's guardrail
layer so the public demo reflects the *guarded* benchmark numbers rather than
the raw model.
