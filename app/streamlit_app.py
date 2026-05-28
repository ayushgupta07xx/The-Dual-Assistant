"""Streamlit demo.

A single chat UI that lets you flip between the open-source and frontier
backends (and toggle guardrails/tools) live, so a reviewer can see the
capability and safety contrast in real time. Every message shows its latency,
token usage, cost, any tools called, and whether a guardrail fired.

Run:  streamlit run app/streamlit_app.py
"""
from __future__ import annotations

import sys
from pathlib import Path

import streamlit as st

# Make the repo root importable when run via `streamlit run app/streamlit_app.py`.
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from assistant import Assistant  # noqa: E402
from assistant.config import settings  # noqa: E402

st.set_page_config(page_title="Dual Assistant", page_icon="🤖", layout="centered")
st.title("🤖 Dual Assistant — OSS vs Frontier")
st.caption("Same assistant core, swappable brain. Memory + tools + guardrails + live cost/latency.")

# --- sidebar controls -------------------------------------------------------
with st.sidebar:
    st.header("Configuration")
    backend = st.radio(
        "Backend",
        ["frontier", "oss"],
        format_func=lambda b: "Frontier (Claude)" if b == "frontier" else "Open-source (Qwen2.5)",
    )
    guardrails = st.toggle("Guardrails", value=True, help="Jailbreak/PII/harmful-content layer")
    tools = st.toggle("Tools", value=True, help="calculator, datetime, wikipedia")
    if st.button("Reset conversation"):
        st.session_state.pop("asst", None)
        st.session_state.pop("history", None)
        st.rerun()

    model = settings.frontier_model if backend == "frontier" else settings.oss_model
    st.markdown(f"**Model:** `{model}`")
    if backend == "frontier" and not settings.anthropic_api_key:
        st.warning("ANTHROPIC_API_KEY not set — frontier backend will error.")

# --- (re)build the assistant when config changes ----------------------------
sig = (backend, guardrails, tools)
if st.session_state.get("sig") != sig or "asst" not in st.session_state:
    st.session_state["asst"] = Assistant(
        backend=backend, guardrails_enabled=guardrails, tools_enabled=tools
    )
    st.session_state["sig"] = sig
    st.session_state.setdefault("history", [])

asst: Assistant = st.session_state["asst"]

# --- render history ---------------------------------------------------------
for turn in st.session_state["history"]:
    with st.chat_message(turn["role"]):
        st.markdown(turn["content"])
        if turn.get("meta"):
            st.caption(turn["meta"])

# --- input ------------------------------------------------------------------
if prompt := st.chat_input("Ask me anything…"):
    st.session_state["history"].append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)
    with st.chat_message("assistant"):
        with st.spinner("thinking…"):
            r = asst.chat(prompt)
        st.markdown(r.text)
        bits = [f"⏱ {r.latency_ms:.0f} ms", f"🔢 {r.input_tokens}+{r.output_tokens} tok",
                f"💲 ${r.cost_usd:.5f}"]
        if r.tools_used:
            bits.append("🛠 " + ", ".join(r.tools_used))
        if r.guardrail_input != "allow" or r.guardrail_output != "allow":
            bits.append(f"🛡 in:{r.guardrail_input} out:{r.guardrail_output}")
        meta = " · ".join(bits)
        st.caption(meta)
    st.session_state["history"].append({"role": "assistant", "content": r.text, "meta": meta})
