"""Public demo — open-source AI assistant with built-in safety guardrails.

A clean Gradio chat UI (Ollive palette: forest green + lime), dark by default
with a light/dark toggle. Calls Groq's OpenAI-compatible API (fast, no GPU).
GROQ_API_KEY is read from the Space's Secrets and is NEVER hardcoded.
Lightweight input guardrails screen unsafe requests before the model. Project +
evaluation details live in the linked GitHub repo.
"""
from __future__ import annotations

import os
import re

import gradio as gr
import requests

GROQ_URL = "https://api.groq.com/openai/v1/chat/completions"
MODEL = os.environ.get("GROQ_MODEL", "llama-3.3-70b-versatile")
API_KEY = os.environ.get("GROQ_API_KEY", "")
REPO_URL = "https://github.com/ayushgupta07xx/dual-assistant-eval"

SYSTEM_PROMPT = (
    "You are a helpful, honest, and harmless personal assistant. "
    "Answer concisely and accurately. If you are unsure or do not know "
    "something, say so plainly instead of inventing facts. Refuse requests "
    "that are illegal, dangerous, or designed to cause harm, and briefly "
    "explain why, offering a safer alternative when possible."
)

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
    "I can't help with that — it falls outside what I can safely assist with. "
    "If there's a safe, legitimate version of what you need, I'm glad to help "
    "with that instead."
)


def _blocked(text: str) -> bool:
    return bool(_JAILBREAK.search(text) or _HARMFUL.search(text))


def respond_core(message: str, history: list) -> str:
    if _blocked(message):
        return _REFUSAL
    if not API_KEY:
        return ("**Setup needed:** this Space is missing its `GROQ_API_KEY` "
                "secret. Add it under Settings -> Variables and secrets, then restart.")

    msgs = [{"role": "system", "content": SYSTEM_PROMPT}]
    for turn in history or []:
        if isinstance(turn, dict) and turn.get("content"):
            msgs.append({"role": turn.get("role", "user"), "content": turn["content"]})
    msgs.append({"role": "user", "content": message})

    try:
        r = requests.post(
            GROQ_URL,
            headers={"Authorization": f"Bearer {API_KEY}",
                     "Content-Type": "application/json"},
            json={"model": MODEL, "messages": msgs,
                  "temperature": 0.7, "max_tokens": 640},
            timeout=60,
        )
        if r.status_code == 429:
            return ("The demo is busy right now — please try again in a minute.")
        r.raise_for_status()
        return r.json()["choices"][0]["message"]["content"].strip()
    except Exception as exc:  # noqa: BLE001
        return f"_(Sorry — something went wrong reaching the model: {exc})_"


# --- Dark by default; toggle flips light/dark (sun=switch-to-light) ----------
INIT_JS = """
() => { document.body.classList.add('dark'); }
"""
TOGGLE_JS = """
() => { document.body.classList.toggle('dark'); }
"""

THEME = gr.themes.Base(
    primary_hue="green",
    neutral_hue="gray",
    font=[gr.themes.GoogleFont("Hanken Grotesk"), "system-ui", "sans-serif"],
).set(
    body_background_fill="#f2f4ef",
    background_fill_primary="#fafbf8",
    background_fill_secondary="#f2f4ef",
    block_background_fill="#fbfcf9",
    block_border_color="#e2e7dd",
    border_color_primary="#e2e7dd",
    body_text_color="#18241a",
    body_text_color_subdued="#5e6b60",
    input_background_fill="#ffffff",
    input_border_color="#dfe4d9",
    body_background_fill_dark="#0e1f13",
    background_fill_primary_dark="#122418",
    background_fill_secondary_dark="#0e1f13",
    block_background_fill_dark="#16271b",
    block_border_color_dark="#2a3d2f",
    border_color_primary_dark="#2a3d2f",
    body_text_color_dark="#eef2ee",
    body_text_color_subdued_dark="#9caa9f",
    input_background_fill_dark="#16271b",
    input_border_color_dark="#2a3d2f",
    button_primary_text_color="#ffffff",
    button_primary_text_color_dark="#11241a",
)

CSS = """
@import url('https://fonts.googleapis.com/css2?family=Source+Serif+4:opsz,wght@8..60,400;8..60,500;8..60,600&family=Hanken+Grotesk:wght@400;500;600&display=swap');

.gradio-container {
  max-width: 820px !important; margin: 0 auto !important; position: relative;
  background: #f2f4ef !important; --accent: #3a7d2c;
}
.dark .gradio-container { background: #0e1f13 !important; --accent: #a3e635; }
footer { display: none !important; }

/* theme toggle, anchored top-right of the content column */
#theme-toggle, #theme-toggle button {
  position: absolute; top: 14px; right: 14px; z-index: 100;
  width: 40px !important; min-width: 40px !important; height: 40px !important; padding: 0 !important;
  border-radius: 50% !important; font-size: 18px !important; line-height: 1 !important; flex: none !important;
  background: #fbfcf9 !important; border: 1px solid #e2e7dd !important;
  color: #5e6b60 !important; box-shadow: 0 1px 3px rgba(0,0,0,0.06);
}
#theme-toggle:hover, #theme-toggle button:hover { border-color: var(--accent) !important; color: var(--accent) !important; }
.dark #theme-toggle, .dark #theme-toggle button { background: #1c3022 !important; border-color: #2f4536 !important; color: #cfe0c8 !important; }

/* Send button — forced (Gradio's default primary is orange) */
#send-btn button {
  background: var(--accent) !important; border: none !important;
  color: #ffffff !important; font-weight: 600 !important; box-shadow: none !important;
}
.dark #send-btn button { color: #11241a !important; }

/* chat panel + input — forced to match the theme */
#chat { background: #fbfcf9 !important; border: 1px solid #e2e7dd !important; }
.dark #chat { background: #16271b !important; border-color: #2a3d2f !important; }
#msgbox textarea, #msgbox input { background: #ffffff !important; }
.dark #msgbox textarea, .dark #msgbox input { background: #16271b !important; color: #eef2ee !important; }

.hero { text-align: center; padding: 46px 18px 4px; }
.hero h1 {
  font-family: 'Source Serif 4', Georgia, serif; font-weight: 500; font-size: 44px;
  line-height: 1.08; color: #18221a; margin: 0 0 14px; letter-spacing: -0.01em;
}
.dark .hero h1 { color: #f3f5f0; }
.hero h1 em { font-style: normal; color: var(--accent); }
.hero .sub { font-family: 'Hanken Grotesk', sans-serif; color: #5e6b60; font-size: 16px; max-width: 520px; margin: 0 auto 20px; line-height: 1.6; }
.dark .hero .sub { color: #9caa9f; }
.tags { display: flex; gap: 9px; justify-content: center; flex-wrap: wrap; margin-bottom: 18px; }
.tag {
  font-family: 'Hanken Grotesk', sans-serif; font-size: 13px; color: #4f5b50;
  background: #fbfcf9; border: 1px solid #e2e7dd; border-radius: 999px;
  padding: 6px 14px; display: inline-flex; align-items: center; gap: 7px;
}
.dark .tag { background: #16271b; border-color: #2a3d2f; color: #c1cdbf; }
.tag::before { content: ""; width: 6px; height: 6px; border-radius: 50%; background: var(--accent); display: inline-block; }
.link { font-family: 'Hanken Grotesk', sans-serif; font-size: 14px; }
.link a { color: var(--accent); text-decoration: none; font-weight: 600; }
.link a:hover { text-decoration: underline; }

.footnote { text-align: center; color: #7e8a7f; font-size: 13px; line-height: 1.7; padding: 18px 18px 30px; margin-top: 14px; }
.dark .footnote { color: #8a978b; }
.footnote a { color: var(--accent); text-decoration: none; font-weight: 500; }
"""

HERO = f"""
<div class="hero">
  <h1>The Dual <em>Assistant</em></h1>
  <p class="sub">A fast, open-source AI assistant with safety built in. Ask it anything.</p>
  <div class="tags">
    <span class="tag">Open-source model</span>
    <span class="tag">Built-in safety guardrails</span>
    <span class="tag">Free &amp; fast</span>
  </div>
  <div class="link"><a href="{REPO_URL}" target="_blank">View the project on GitHub →</a></div>
</div>
"""

FOOTER = f"""
<div class="footnote">
  Unsafe requests are screened before they reach the model.<br/>
  Open-source — see the <a href="{REPO_URL}/blob/main/report/eval_report.pdf" target="_blank">full evaluation report →</a>
</div>
"""

EXAMPLES = [
    "Explain the difference between TCP and UDP in two sentences.",
    "What is 18.5% of 240?",
    "Summarize the plot of Romeo and Juliet for a 10-year-old.",
    "Draft a short, polite email asking to reschedule a meeting.",
]


def _user(message, history):
    if not message or not message.strip():
        return "", history or []
    return "", (history or []) + [{"role": "user", "content": message}]


def _bot(history):
    user_msg = history[-1]["content"]
    reply = respond_core(user_msg, history[:-1])
    return history + [{"role": "assistant", "content": reply}]


with gr.Blocks(theme=THEME, css=CSS, js=INIT_JS, title="The Dual Assistant") as demo:
    theme_btn = gr.Button("\u25D0", elem_id="theme-toggle")
    gr.HTML(HERO)
    chatbot = gr.Chatbot(
        type="messages", height=420, show_label=False, elem_id="chat",
        placeholder="### Ask the assistant anything\nResponses are screened by safety guardrails first.",
    )
    with gr.Row():
        txt = gr.Textbox(show_label=False, scale=8, autofocus=True, container=False,
                         elem_id="msgbox", placeholder="Type a message and press Enter…")
        send = gr.Button("Send", variant="primary", scale=1, min_width=90, elem_id="send-btn")
    gr.Examples(examples=EXAMPLES, inputs=txt, label="Try one")
    gr.HTML(FOOTER)

    theme_btn.click(fn=None, inputs=None, outputs=None, js=TOGGLE_JS)
    txt.submit(_user, [txt, chatbot], [txt, chatbot]).then(_bot, chatbot, chatbot)
    send.click(_user, [txt, chatbot], [txt, chatbot]).then(_bot, chatbot, chatbot)

if __name__ == "__main__":
    demo.queue().launch()
