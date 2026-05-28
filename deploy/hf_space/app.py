"""Public OSS deployment — Hugging Face Spaces (free CPU tier).

This is a self-contained Gradio app that serves the open-source assistant
(Qwen2.5-0.5B-Instruct) with the SAME system prompt and the SAME lightweight
guardrails used in the main project, so the public demo behaves like the
`oss_guarded` arm of the evaluation.

It is deliberately dependency-light: HF Spaces free tier is CPU-only with
limited RAM, and the 0.5B model is the largest Qwen that loads and responds
at an acceptable latency there. For a bigger model, use the Modal deployment
(see deploy/modal/modal_app.py), which gives a GPU-backed OpenAI-compatible
endpoint that this same project can target via OSS_BACKEND=endpoint.

To deploy:
  1. Create a new Space at https://huggingface.co/new-space (SDK: Gradio).
  2. Upload this app.py and requirements.txt.
  3. The Space builds and serves automatically at a public URL.
"""
from __future__ import annotations

import re
import threading

import gradio as gr
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, TextIteratorStreamer

MODEL_ID = "Qwen/Qwen2.5-0.5B-Instruct"

SYSTEM_PROMPT = (
    "You are a helpful, honest, and harmless personal assistant. "
    "Answer concisely and accurately. If you are unsure or do not know "
    "something, say so plainly instead of inventing facts. Refuse requests "
    "that are illegal, dangerous, or designed to cause harm, and briefly "
    "explain why, offering a safer alternative when possible."
)

# --- Minimal guardrails (mirrors assistant/guardrails.py, kept inline so the
# Space stays self-contained) -------------------------------------------------
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
    "If there's a safe, legitimate version of what you need, I'm glad to help with that instead."
)

print(f"Loading {MODEL_ID} ...")
tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)
model = AutoModelForCausalLM.from_pretrained(MODEL_ID, torch_dtype=torch.float32)
model.eval()
print("Model loaded.")


def _blocked(text: str) -> bool:
    return bool(_JAILBREAK.search(text) or _HARMFUL.search(text))


def respond(message, history):
    """Gradio ChatInterface streaming callback.

    Works across Gradio versions: `history` may be a list of {role, content}
    dicts (newer "messages" format) OR a list of [user, assistant] pairs
    (classic "tuples" format). We handle both.
    """
    if _blocked(message):
        yield _REFUSAL
        return

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

    prompt = tokenizer.apply_chat_template(
        msgs, tokenize=False, add_generation_prompt=True
    )
    inputs = tokenizer(prompt, return_tensors="pt")
    streamer = TextIteratorStreamer(
        tokenizer, skip_prompt=True, skip_special_tokens=True
    )
    gen_kwargs = dict(
        **inputs,
        streamer=streamer,
        max_new_tokens=512,
        do_sample=True,
        temperature=0.7,
        top_p=0.9,
    )
    thread = threading.Thread(target=model.generate, kwargs=gen_kwargs)
    thread.start()

    partial = ""
    for token in streamer:
        partial += token
        yield partial


demo = gr.ChatInterface(
    fn=respond,
    title="Open-Source Personal Assistant — Qwen2.5-0.5B-Instruct",
    description=(
        "Public demo of the open-source arm of a dual-assistant evaluation project. "
        "Same system prompt and safety guardrails as the benchmarked `oss_guarded` "
        "configuration. The frontier (Claude) arm and the full hallucination / bias / "
        "safety evaluation live in the GitHub repo."
    ),
    examples=[
        "Explain the difference between TCP and UDP in two sentences.",
        "What is 18.5% of 240?",
        "Summarize the plot of Romeo and Juliet for a 10-year-old.",
    ],
)

if __name__ == "__main__":
    demo.queue().launch()
