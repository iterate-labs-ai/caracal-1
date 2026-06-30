"""Caracal s07 inference - in-model agent loop.

Not a framework wrapper. Modelo gera special tokens nativamente
(<tool_call>, <observation>, \\boxed{}). Python parsea + chama tool + injeta result.
"""

import re
from collections.abc import Callable

from eval.s07.hier_reward import normalize_cwe

TOOL_CALL_RE = re.compile(r"<tool_call>(.*?)</tool_call>", re.DOTALL)
ANSWER_RE = re.compile(r"\\boxed\{(CWE-?\d{1,4})\}")
HALT_RE = re.compile(r'<halt confidence="([\d.]+)"\s*/>')

MAX_TURNS = 8


def caracal_agent_generate(
    cve_desc: str,
    model_generate_fn: Callable,
    tools: dict[str, Callable],
    max_turns: int = MAX_TURNS,
) -> dict:
    """Run Caracal-Agent inference loop.

    model_generate_fn(text, stop_tokens) -> new_text
    tools: dict of name -> callable
    """
    text = f"<cve>{cve_desc}</cve>\n"
    trajectory = []

    for turn in range(max_turns):
        new_text = model_generate_fn(text, stop_tokens=["</tool_call>", "</answer>", "<halt/>"])
        text += new_text
        trajectory.append({"turn": turn, "generated": new_text})

        if ANSWER_RE.search(text):
            cwe = normalize_cwe(text)
            return {"cwe": cwe, "trajectory": trajectory, "final_text": text}

        if HALT_RE.search(text):
            cwe = normalize_cwe(text)
            return {"cwe": cwe, "trajectory": trajectory, "final_text": text, "halted": True}

        tool_match = TOOL_CALL_RE.search(text)
        if tool_match:
            tool_call_raw = tool_match.group(1).strip()
            try:
                import json

                tool_data = json.loads(tool_call_raw)
                tool_name = tool_data["name"]
                tool_args = tool_data.get("arguments", {})
                if tool_name in tools:
                    obs = tools[tool_name](**tool_args)
                else:
                    obs = {"error": f"unknown tool {tool_name}"}
                text += f"<observation>{json.dumps(obs)}</observation>\n"
            except (json.JSONDecodeError, KeyError) as e:
                text += f"<observation>{{'error': '{e}'}}</observation>\n"

    return {
        "cwe": normalize_cwe(text),
        "trajectory": trajectory,
        "final_text": text,
        "max_turns_reached": True,
    }
