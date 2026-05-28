"""Minimal CLI chat loop. Useful for smoke-testing without the UI.

    python cli.py                  # uses BACKEND from env/.env (default frontier)
    python cli.py --backend oss    # force the open-source backend
    python cli.py --no-guardrails  # disable the safety layer

Commands inside the chat: /reset to clear memory, /quit to exit.
"""
from __future__ import annotations

import argparse

from assistant import Assistant


def main() -> None:
    ap = argparse.ArgumentParser(description="Dual-assistant CLI")
    ap.add_argument("--backend", choices=["frontier", "oss"], default=None)
    ap.add_argument("--no-guardrails", action="store_true")
    ap.add_argument("--no-tools", action="store_true")
    args = ap.parse_args()

    asst = Assistant(
        backend=args.backend,
        guardrails_enabled=False if args.no_guardrails else None,
        tools_enabled=False if args.no_tools else None,
    )
    print(f"[{asst.label}] model={asst.provider.model}  (type /quit to exit)\n")

    while True:
        try:
            msg = input("you> ").strip()
        except (EOFError, KeyboardInterrupt):
            print()
            break
        if not msg:
            continue
        if msg == "/quit":
            break
        if msg == "/reset":
            asst.reset()
            print("(memory cleared)\n")
            continue
        r = asst.chat(msg)
        meta = (
            f"  [{r.latency_ms:.0f} ms | {r.input_tokens}+{r.output_tokens} tok "
            f"| ${r.cost_usd:.5f}"
            + (f" | tools: {', '.join(r.tools_used)}" if r.tools_used else "")
            + (f" | guard: {r.guardrail_input}/{r.guardrail_output}]" )
        )
        print(f"bot> {r.text}\n{meta}\n")


if __name__ == "__main__":
    main()
