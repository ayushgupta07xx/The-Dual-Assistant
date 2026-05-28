"""Tools the assistant can call.

Each tool is dependency-light and self-describing. The calculator and clock
are deterministic (great for demos and reproducible eval runs); the Wikipedia
lookup performs a real network call when run on a machine with internet
access. Tools expose a JSON-schema so the frontier model can use native tool
calling, and a plain-text signature so the small OSS model can use a
ReAct-style prompt.
"""
from __future__ import annotations

import ast
import datetime as _dt
import json
import operator
from dataclasses import dataclass
from typing import Any, Callable, Dict, List

# ---------------------------------------------------------------------------
# Safe arithmetic evaluator (no eval(); only a whitelist of AST node types).
# ---------------------------------------------------------------------------
_ALLOWED_BINOPS = {
    ast.Add: operator.add,
    ast.Sub: operator.sub,
    ast.Mult: operator.mul,
    ast.Div: operator.truediv,
    ast.Pow: operator.pow,
    ast.Mod: operator.mod,
    ast.FloorDiv: operator.floordiv,
}
_ALLOWED_UNARY = {ast.UAdd: operator.pos, ast.USub: operator.neg}


def _safe_eval(node: ast.AST) -> float:
    if isinstance(node, ast.Constant):  # numbers
        if isinstance(node.value, (int, float)):
            return node.value
        raise ValueError("only numeric constants are allowed")
    if isinstance(node, ast.BinOp) and type(node.op) in _ALLOWED_BINOPS:
        return _ALLOWED_BINOPS[type(node.op)](_safe_eval(node.left), _safe_eval(node.right))
    if isinstance(node, ast.UnaryOp) and type(node.op) in _ALLOWED_UNARY:
        return _ALLOWED_UNARY[type(node.op)](_safe_eval(node.operand))
    raise ValueError("unsupported expression")


def calculator(expression: str) -> str:
    """Evaluate a basic arithmetic expression, e.g. '12.5 * (3 + 4) ** 2'."""
    try:
        tree = ast.parse(expression, mode="eval")
        result = _safe_eval(tree.body)
        return f"{expression} = {result}"
    except Exception as exc:  # noqa: BLE001
        return f"calculator error: {exc}"


def current_datetime(timezone: str = "UTC") -> str:
    """Return the current date and time. Only 'UTC' is guaranteed offline."""
    now = _dt.datetime.now(_dt.timezone.utc)
    return now.strftime("%Y-%m-%d %H:%M:%S UTC (%A)")


def search_wikipedia(query: str) -> str:
    """Look up a short summary from Wikipedia (requires internet)."""
    try:
        import urllib.parse
        import urllib.request

        title = urllib.parse.quote(query.strip().replace(" ", "_"))
        url = f"https://en.wikipedia.org/api/rest_v1/page/summary/{title}"
        req = urllib.request.Request(url, headers={"User-Agent": "dual-assistant-eval/1.0"})
        with urllib.request.urlopen(req, timeout=8) as resp:  # noqa: S310
            data = json.loads(resp.read().decode("utf-8"))
        extract = data.get("extract")
        if extract:
            return extract[:600]
        return f"No Wikipedia summary found for '{query}'."
    except Exception as exc:  # noqa: BLE001
        return f"wikipedia lookup unavailable ({exc})."


@dataclass
class Tool:
    name: str
    description: str
    parameters: Dict[str, Any]  # JSON-schema "properties" + "required"
    fn: Callable[..., str]

    def to_anthropic(self) -> Dict[str, Any]:
        """Schema for the Anthropic native tool-use API."""
        return {
            "name": self.name,
            "description": self.description,
            "input_schema": {"type": "object", **self.parameters},
        }

    def signature(self) -> str:
        """Compact text signature used in the OSS ReAct prompt."""
        props = self.parameters.get("properties", {})
        args = ", ".join(f"{k}: {v.get('type', 'string')}" for k, v in props.items())
        return f"{self.name}({args}) - {self.description}"


def default_tools() -> List[Tool]:
    return [
        Tool(
            name="calculator",
            description="Evaluate an arithmetic expression and return the result.",
            parameters={
                "properties": {"expression": {"type": "string", "description": "e.g. '2 + 2 * 5'"}},
                "required": ["expression"],
            },
            fn=lambda expression: calculator(expression),
        ),
        Tool(
            name="current_datetime",
            description="Get the current UTC date and time.",
            parameters={
                "properties": {"timezone": {"type": "string", "description": "IANA tz, default UTC"}},
                "required": [],
            },
            fn=lambda timezone="UTC": current_datetime(timezone),
        ),
        Tool(
            name="search_wikipedia",
            description="Fetch a short factual summary about a topic from Wikipedia.",
            parameters={
                "properties": {"query": {"type": "string", "description": "topic to look up"}},
                "required": ["query"],
            },
            fn=lambda query: search_wikipedia(query),
        ),
    ]


class ToolRegistry:
    """Holds tools and runs them by name with basic error isolation."""

    def __init__(self, tools: List[Tool] | None = None):
        self.tools = {t.name: t for t in (tools if tools is not None else default_tools())}

    def anthropic_specs(self) -> List[Dict[str, Any]]:
        return [t.to_anthropic() for t in self.tools.values()]

    def react_manual(self) -> str:
        return "\n".join(f"- {t.signature()}" for t in self.tools.values())

    def run(self, name: str, args: Dict[str, Any]) -> str:
        tool = self.tools.get(name)
        if tool is None:
            return f"error: unknown tool '{name}'"
        try:
            return str(tool.fn(**args))
        except Exception as exc:  # noqa: BLE001
            return f"error running {name}: {exc}"
