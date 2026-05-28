"""Open-source provider.

Serves a Hugging Face model through one of three backends selected by config:

  * transformers - load weights locally (default: Qwen2.5-0.5B-Instruct)
  * ollama       - call a local Ollama daemon
  * endpoint     - call an OpenAI-compatible HTTP endpoint (e.g. a Modal/vLLM
                   deployment)

Small open models don't have reliable native tool calling, so tools are
offered through a ReAct text protocol: the model emits an ACTION line, we run
the tool, feed back an OBSERVATION, and let it continue. The protocol is
parsed defensively because tiny models produce malformed output often --
which is itself a finding the evaluation surfaces.
"""
from __future__ import annotations

import json
import re
from typing import List, Optional, Tuple

from .base import ChatResult, Provider

_ACTION_RE = re.compile(r"ACTION:\s*(\{.*?\})", re.DOTALL)

_REACT_INSTRUCTIONS = (
    "\n\nYou can use tools. Available tools:\n{manual}\n\n"
    "To call a tool, output exactly one line:\n"
    'ACTION: {{"tool": "<name>", "args": {{...}}}}\n'
    "Then stop and wait. You will receive an OBSERVATION with the result. "
    "When you have the final answer, reply normally without an ACTION line."
)


class OSSProvider(Provider):
    name = "oss"

    def __init__(
        self,
        backend: str = "transformers",
        model: str = "Qwen/Qwen2.5-0.5B-Instruct",
        ollama_host: str = "http://localhost:11434",
        endpoint_url: str = "",
        endpoint_key: str = "",
    ):
        self.backend = backend
        self.model = model
        self.ollama_host = ollama_host.rstrip("/")
        self.endpoint_url = endpoint_url.rstrip("/")
        self.endpoint_key = endpoint_key
        self._hf_model = None
        self._hf_tok = None

    # --- backend-specific raw generation --------------------------------
    def _load_hf(self):
        if self._hf_model is None:
            import torch
            from transformers import AutoModelForCausalLM, AutoTokenizer

            self._hf_tok = AutoTokenizer.from_pretrained(self.model)
            self._hf_model = AutoModelForCausalLM.from_pretrained(
                self.model,
                torch_dtype="auto",
                device_map="auto" if torch.cuda.is_available() else None,
            )
        return self._hf_model, self._hf_tok

    def _gen_transformers(self, system, messages, temperature, max_tokens) -> Tuple[str, int, int]:
        import torch

        model, tok = self._load_hf()
        chat = [{"role": "system", "content": system}] + messages
        prompt = tok.apply_chat_template(chat, tokenize=False, add_generation_prompt=True)
        inputs = tok(prompt, return_tensors="pt").to(model.device)
        in_tokens = int(inputs.input_ids.shape[1])
        with torch.no_grad():
            out = model.generate(
                **inputs,
                max_new_tokens=max_tokens,
                do_sample=temperature > 0,
                temperature=max(temperature, 0.01),
                top_p=0.9,
                pad_token_id=tok.eos_token_id,
            )
        gen = out[0][in_tokens:]
        out_tokens = int(gen.shape[0])
        text = tok.decode(gen, skip_special_tokens=True)
        return text.strip(), in_tokens, out_tokens

    def _gen_ollama(self, system, messages, temperature, max_tokens) -> Tuple[str, int, int]:
        import requests

        payload = {
            "model": self.model,
            "messages": [{"role": "system", "content": system}] + messages,
            "stream": False,
            "options": {"temperature": temperature, "num_predict": max_tokens},
        }
        r = requests.post(f"{self.ollama_host}/api/chat", json=payload, timeout=120)
        r.raise_for_status()
        data = r.json()
        text = data.get("message", {}).get("content", "")
        return text.strip(), int(data.get("prompt_eval_count", 0)), int(data.get("eval_count", 0))

    def _gen_endpoint(self, system, messages, temperature, max_tokens) -> Tuple[str, int, int]:
        import requests

        headers = {"Content-Type": "application/json"}
        if self.endpoint_key:
            headers["Authorization"] = f"Bearer {self.endpoint_key}"
        payload = {
            "model": self.model,
            "messages": [{"role": "system", "content": system}] + messages,
            "temperature": temperature,
            "max_tokens": max_tokens,
        }
        r = requests.post(
            f"{self.endpoint_url}/v1/chat/completions", json=payload, headers=headers, timeout=120
        )
        r.raise_for_status()
        data = r.json()
        text = data["choices"][0]["message"]["content"]
        usage = data.get("usage", {})
        return text.strip(), int(usage.get("prompt_tokens", 0)), int(usage.get("completion_tokens", 0))

    def _generate(self, system, messages, temperature, max_tokens) -> Tuple[str, int, int]:
        if self.backend == "transformers":
            return self._gen_transformers(system, messages, temperature, max_tokens)
        if self.backend == "ollama":
            return self._gen_ollama(system, messages, temperature, max_tokens)
        if self.backend == "endpoint":
            return self._gen_endpoint(system, messages, temperature, max_tokens)
        raise ValueError(f"unknown OSS backend: {self.backend}")

    # --- public API ------------------------------------------------------
    def chat(
        self,
        system: str,
        messages: List[dict],
        tools=None,
        temperature: float = 0.7,
        max_tokens: int = 640,
    ) -> ChatResult:
        full_system = system
        if tools is not None:
            full_system = system + _REACT_INSTRUCTIONS.format(manual=tools.react_manual())

        working = [dict(m) for m in messages]
        tool_trace: List[dict] = []
        in_total = out_total = 0
        last_text = ""

        for _ in range(4):  # ReAct iterations
            text, in_tok, out_tok = self._generate(full_system, working, temperature, max_tokens)
            in_total += in_tok
            out_total += out_tok
            last_text = text

            match = _ACTION_RE.search(text) if tools is not None else None
            if not match:
                break
            try:
                call = json.loads(match.group(1))
                name, args = call.get("tool", ""), call.get("args", {}) or {}
            except json.JSONDecodeError:
                break  # malformed tool call -> treat current text as the answer
            result = tools.run(name, args)
            tool_trace.append({"tool": name, "args": args, "result": result})
            working.append({"role": "assistant", "content": text})
            working.append({"role": "user", "content": f"OBSERVATION: {result}"})

        # Strip any leftover ACTION lines from the final answer.
        clean = _ACTION_RE.sub("", last_text).strip() or last_text.strip()
        return ChatResult(
            text=clean,
            input_tokens=in_total,
            output_tokens=out_total,
            model=self.model,
            provider=self.name,
            tool_trace=tool_trace,
        )
