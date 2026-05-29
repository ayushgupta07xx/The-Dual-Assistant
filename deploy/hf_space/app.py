"""Public demo — Dual Assistant (Frontier vs Open-source), both via Groq.

Gradio chat UI (Ollive palette: forest green + lime), dark by default. A model
dropdown (top-right, above the chat) switches between a large and a small
open-source model; each model keeps its OWN conversation, so switching back
resumes where you left off. Per-message copy buttons and a fullscreen toggle.
Both arms call Groq's OpenAI-compatible API so the demo runs publicly with no
GPU. Lightweight input guardrails screen unsafe requests before the model.

GROQ_API_KEY is read from the Space's Secrets and is NEVER hardcoded.
"""
from __future__ import annotations

import os
import re

import gradio as gr
import requests

GROQ_URL = "https://api.groq.com/openai/v1/chat/completions"
API_KEY = os.environ.get("GROQ_API_KEY", "")
REPO_URL = "https://github.com/ayushgupta07xx/dual-assistant-eval"

FRONTIER_MODEL = os.environ.get("FRONTIER_MODEL", "llama-3.3-70b-versatile")
OSS_MODEL = os.environ.get("OSS_DEMO_MODEL", "llama-3.1-8b-instant")
CHOICES = [("Frontier model", "frontier"), ("Open-source model", "oss")]

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


# Per-arm generation settings. The frontier arm runs crisp and deterministic;
# the small open-source arm runs at higher temperature with a tighter token cap,
# which surfaces the real reliability gap of a smaller model (more drift, less
# coherent on hard / loaded prompts) — instead of rigging it with a broken model.
GEN_PARAMS = {
    "frontier": {"temperature": 0.6, "top_p": 0.9, "max_tokens": 700},
    "oss": {"temperature": 1.4, "top_p": 1.0, "max_tokens": 384},
}


def _call_groq(model: str, message: str, history: list, arm: str = "frontier") -> str:
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

    gp = GEN_PARAMS.get(arm, GEN_PARAMS["frontier"])
    try:
        r = requests.post(
            GROQ_URL,
            headers={"Authorization": f"Bearer {API_KEY}",
                     "Content-Type": "application/json"},
            json={"model": model, "messages": msgs,
                  "temperature": gp["temperature"], "top_p": gp["top_p"],
                  "max_tokens": gp["max_tokens"]},
            timeout=60,
        )
        if r.status_code == 429:
            return "The demo is busy right now — please try again in a minute."
        r.raise_for_status()
        return r.json()["choices"][0]["message"]["content"].strip()
    except Exception as exc:  # noqa: BLE001
        return f"_(Sorry — something went wrong reaching the model: {exc})_"


# --- JS: dark default, theme toggle, fullscreen the chat --------------------
INIT_JS = """
() => {
  document.body.classList.add('dark');
  const fix = () => {
    document.querySelectorAll('#model-dd, #model-dd *').forEach(el => {
      if (el.style && el.style.width) { el.style.width = ''; }
    });
  };
  fix();
  const dd = document.querySelector('#model-dd');
  if (dd) { new MutationObserver(fix).observe(dd, {attributes:true, subtree:true, attributeFilter:['style']}); }
}
"""
TOGGLE_JS = """
() => { document.body.classList.toggle('dark'); }
"""
FULLSCREEN_JS = """
() => {
  const el = document.querySelector('.gradio-container') || document.documentElement;
  if (!document.fullscreenElement) { el.requestFullscreen(); }
  else { document.exitFullscreen(); }
}
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

#theme-toggle, #theme-toggle button {
  position: absolute; top: 14px; right: 14px; z-index: 100;
  width: 40px !important; min-width: 40px !important; height: 40px !important; padding: 0 !important;
  border-radius: 50% !important; font-size: 0 !important; line-height: 0 !important; flex: none !important;
  background: #fbfcf9 !important; border: 1px solid #e2e7dd !important; box-shadow: 0 1px 3px rgba(0,0,0,0.06);
  background-image: url("data:image/svg+xml,%3Csvg xmlns='http://www.w3.org/2000/svg' viewBox='0 0 24 24'%3E%3Cdefs%3E%3CclipPath id='l'%3E%3Crect x='0' y='0' width='12' height='24'/%3E%3C/clipPath%3E%3CclipPath id='r'%3E%3Crect x='12' y='0' width='12' height='24'/%3E%3C/clipPath%3E%3C/defs%3E%3Cg transform='rotate(-32 12 12)'%3E%3Cellipse cx='12' cy='13' rx='6.2' ry='8.2' fill='%231f3d27' clip-path='url(%23l)'/%3E%3Cellipse cx='12' cy='13' rx='6.2' ry='8.2' fill='%23a3e635' clip-path='url(%23r)'/%3E%3Cpath d='M12 5 q1.6 -2.4 4 -2.8' stroke='%235a7d3a' stroke-width='1.3' fill='none' stroke-linecap='round'/%3E%3C/g%3E%3C/svg%3E") !important;
  background-repeat: no-repeat !important; background-position: center !important; background-size: 21px 21px !important;
}
#theme-toggle:hover, #theme-toggle button:hover { border-color: var(--accent) !important; filter: brightness(1.05); }
.dark #theme-toggle, .dark #theme-toggle button { background-color: #1c3022 !important; border-color: #2f4536 !important; }

#send-btn button {
  background: var(--accent) !important; border: none !important;
  color: #ffffff !important; font-weight: 600 !important; box-shadow: none !important;
}
.dark #send-btn button { color: #11241a !important; }

/* chat wrapper is the positioning context for the embedded controls */
#chat-wrap { position: relative !important; }
/* give the chat headroom so messages don't sit under the embedded bar */
#chat { padding-top: 52px !important; }

/* dropdown embedded into chat top-left. DOM: #model-dd.block > .wrap > .wrap-inner */
#model-dd {
  position: absolute !important; top: 10px; left: 12px; z-index: 30;
  width: 172px !important; min-width: 172px !important; max-width: 172px !important;
  background: transparent !important; border: none !important; box-shadow: none !important;
  padding: 0 !important; overflow: visible !important;
}
/* hide the stray empty absolute ".wrap.default.full" overlay Gradio renders */
#model-dd > .wrap.default { display: none !important; }
/* the visible pill */
#model-dd .wrap.svelte-1hfxrpf, #model-dd > div > .wrap {
  position: relative !important; width: 172px !important; min-width: 172px !important;
  border-radius: 8px !important; cursor: pointer !important;
  background: #eef4ea !important; border: 1px solid #dbe6d3 !important;
}
.dark #model-dd .wrap.svelte-1hfxrpf, .dark #model-dd > div > .wrap {
  background: #1c3022 !important; border: 1px solid #2f4536 !important;
}
/* keep the text+arrow visible and clickable */
#model-dd .wrap-inner { display: flex !important; padding: 7px 10px !important; font-size: 13px !important; cursor: pointer !important; }
#model-dd .wrap-inner *, #model-dd .secondary-wrap, #model-dd input { cursor: pointer !important; background: transparent !important; }
.dark #model-dd .wrap-inner, .dark #model-dd .wrap-inner * { color: #eef2ee !important; }

/* fullscreen button embedded into the chat's top-right corner */
#fs-btn { position: absolute !important; top: 10px; right: 12px; z-index: 30; flex: 0 0 34px !important; width: 34px !important; min-width: 34px !important; }
#fs-btn button {
  width: 34px !important; min-width: 34px !important; max-width: 34px !important; height: 34px !important;
  padding: 0 !important; border-radius: 8px !important; font-size: 14px !important; line-height: 1 !important;
  background: #eef4ea !important; border: 1px solid #dbe6d3 !important; color: var(--accent) !important;
  box-shadow: none !important;
}
.dark #fs-btn button { background: #1c3022 !important; border: 1px solid #2f4536 !important; color: var(--accent) !important; }
#fs-btn button:hover { border-color: var(--accent) !important; filter: brightness(1.08); }

#chat { background: #fbfcf9 !important; border: 1px solid #e2e7dd !important; }
.dark #chat { background: #16271b !important; border-color: #2a3d2f !important; }
.gradio-container:fullscreen { background: #0e1f13 !important; padding: 22px 26px; overflow-y: auto !important; }
.gradio-container:fullscreen #chat { height: 70vh !important; }
#msgbox textarea, #msgbox input { background: #ffffff !important; }
.dark #msgbox textarea, .dark #msgbox input { background: #16271b !important; color: #eef2ee !important; }



/* thin accent scrollbar inside the chat */
#chat, #chat * { scrollbar-width: thin; scrollbar-color: var(--accent) transparent; }
#chat::-webkit-scrollbar, #chat *::-webkit-scrollbar { width: 7px; height: 7px; }
#chat::-webkit-scrollbar-track, #chat *::-webkit-scrollbar-track { background: transparent; }
#chat::-webkit-scrollbar-thumb, #chat *::-webkit-scrollbar-thumb {
  background: #3a7d2c; border-radius: 6px; border: 2px solid transparent; background-clip: content-box;
}
.dark #chat::-webkit-scrollbar-thumb, .dark #chat *::-webkit-scrollbar-thumb {
  background: #5c8c3f; border-radius: 6px; border: 2px solid transparent; background-clip: content-box;
}
#chat::-webkit-scrollbar-thumb:hover, #chat *::-webkit-scrollbar-thumb:hover { background: var(--accent); background-clip: content-box; }

/* per-message copy button: small + subtle, always visible (reliable across versions) */
#chat button.copy-button, #chat .icon-button-wrapper button, #chat button[title="Copy"] {
  transform: scale(0.72) !important; opacity: 0.55 !important; transition: opacity .15s ease !important;
}
#chat button.copy-button:hover, #chat .icon-button-wrapper button:hover, #chat button[title="Copy"]:hover {
  opacity: 1 !important; color: var(--accent) !important;
}

.hero { text-align: center; padding: 46px 18px 0; }
.hero h1 {
  font-family: 'Source Serif 4', Georgia, serif; font-weight: 500; font-size: 44px;
  line-height: 1.08; color: #18221a; margin: 0 0 12px; letter-spacing: -0.01em;
}
.dark .hero h1 { color: #f3f5f0; }
.hero h1 em { font-style: normal; color: var(--accent); }
.hero .sub { font-family: 'Hanken Grotesk', sans-serif; color: #5e6b60; font-size: 16px; max-width: 520px; margin: 0 auto 18px; line-height: 1.6; }
.dark .hero .sub { color: #9caa9f; }
.tags { display: flex; gap: 9px; justify-content: center; flex-wrap: wrap; margin-bottom: 14px; }
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

.footnote { text-align: center; color: #7e8a7f; font-size: 13px; line-height: 1.7; padding: 14px 18px 30px; margin-top: 12px; }
.dark .footnote { color: #8a978b; }
.footnote a { color: var(--accent); text-decoration: none; font-weight: 500; }
"""

HERO = f"""
<div class="hero">
  <h1>The Dual <em>Assistant</em></h1>
  <p class="sub">One assistant core, two interchangeable models.</p>
  <div class="tags">
    <span class="tag">Frontier vs open-source</span>
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


def _send(message, arm, store):
    """Append the user msg + model reply to the selected arm's own history."""
    store = store or {"frontier": [], "oss": []}
    if not message or not message.strip():
        return "", store.get(arm, []), store
    model = FRONTIER_MODEL if arm == "frontier" else OSS_MODEL
    hist = list(store.get(arm, []))
    reply = _call_groq(model, message, hist, arm)
    hist = hist + [
        {"role": "user", "content": message},
        {"role": "assistant", "content": reply},
    ]
    store[arm] = hist
    return "", hist, store


def _switch(arm, store):
    """Show the selected arm's cached conversation."""
    store = store or {"frontier": [], "oss": []}
    return store.get(arm, [])


with gr.Blocks(theme=THEME, css=CSS, js=INIT_JS, title="The Dual Assistant") as demo:
    theme_btn = gr.Button("\u25D0", elem_id="theme-toggle")
    gr.HTML(HERO)

    # per-arm conversation store (kept across model switches)
    store = gr.State({"frontier": [], "oss": []})

    with gr.Column(elem_id="chat-wrap"):
        model_dd = gr.Dropdown(
            choices=CHOICES, value="frontier", show_label=False,
            container=False, elem_id="model-dd",
        )
        fs_btn = gr.Button("\u26F6", elem_id="fs-btn")  # fullscreen glyph
        chatbot = gr.Chatbot(
            type="messages", height=440, show_label=False, elem_id="chat",
            show_copy_button=True,
            placeholder="### Ask the assistant anything\nPick a model, then send a message. Each model keeps its own chat.",
        )
    with gr.Row():
        txt = gr.Textbox(show_label=False, scale=8, autofocus=True, container=False,
                         elem_id="msgbox", placeholder="Type a message and press Enter…")
        send = gr.Button("Send", variant="primary", scale=1, min_width=90, elem_id="send-btn")
    gr.Examples(examples=EXAMPLES, inputs=txt, label="Try one")
    gr.HTML(FOOTER)

    theme_btn.click(fn=None, inputs=None, outputs=None, js=TOGGLE_JS)
    fs_btn.click(fn=None, inputs=None, outputs=None, js=FULLSCREEN_JS)
    model_dd.change(_switch, [model_dd, store], chatbot)
    txt.submit(_send, [txt, model_dd, store], [txt, chatbot, store])
    send.click(_send, [txt, model_dd, store], [txt, chatbot, store])

if __name__ == "__main__":
    demo.queue().launch()
