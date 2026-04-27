from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Callable

from openrlhf_reward import _db_path_from_label, score_response
from sandbox import execute_sql
from sft_data import parse_tagged_response

ModelFn = Callable[[str], str]


def _feedback_message(error: str, rows: list[Any]) -> str:
    if error:
        return (
            "The previous SQL failed in SQLite.\n"
            f"Error: {error}\n"
            "Revise the SQL and answer again with exactly <thought> and <action>."
        )
    if not rows:
        return (
            "The previous SQL executed but returned an empty result.\n"
            "Check whether the selected tables, joins, and filters are too restrictive. "
            "Answer again with exactly <thought> and <action>."
        )
    return (
        "The previous SQL executed but did not match the expected result.\n"
        "Revise the SQL and answer again with exactly <thought> and <action>."
    )


def _append_feedback(prompt: str, response: str, feedback: str) -> str:
    return (
        f"{prompt}\n\n"
        f"Previous response:\n{response}\n\n"
        f"Execution feedback:\n{feedback}"
    )


def rollout_self_correction(
    prompt: str,
    label: str | dict[str, Any],
    model_fn: ModelFn,
    max_turns: int = 2,
    timeout_s: float = 3.0,
) -> dict[str, Any]:
    if max_turns < 1:
        raise ValueError("max_turns must be >= 1")

    parsed_label = json.loads(label) if isinstance(label, str) else label
    current_prompt = prompt
    turns: list[dict[str, Any]] = []
    best = {"reward": -1.0, "turn_index": -1, "response": ""}

    for turn_index in range(max_turns):
        response = model_fn(current_prompt)
        score = score_response(response, parsed_label, timeout_s=timeout_s)
        turn: dict[str, Any] = {
            "turn_index": turn_index,
            "prompt": current_prompt,
            "response": response,
            "reward": score["reward"],
            "format_match": score["format_match"],
            "no_error": score["no_error"],
            "exec_match": score["exec_match"],
            "error": score["error"],
        }
        turns.append(turn)

        if score["reward"] > best["reward"]:
            best = {
                "reward": score["reward"],
                "turn_index": turn_index,
                "response": response,
            }
        if score["exec_match"]:
            break
        if turn_index == max_turns - 1:
            break

        try:
            action_sql = parse_tagged_response(response)["action"]
            execution = execute_sql(
                _db_path_from_label(parsed_label),
                action_sql,
                timeout_s=timeout_s,
            )
            feedback = _feedback_message(
                execution.get("error", ""),
                execution.get("rows", []),
            )
        except Exception as exc:
            feedback = _feedback_message(f"{type(exc).__name__}: {exc}", [])
        current_prompt = _append_feedback(current_prompt, response, feedback)

    return {
        "best_reward": best["reward"],
        "best_turn_index": best["turn_index"],
        "best_response": best["response"],
        "turns": turns,
    }


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Run a deterministic self-correction rollout from prerecorded responses."
    )
    parser.add_argument("--prompt", required=True)
    parser.add_argument("--label", required=True)
    parser.add_argument(
        "--responses-json",
        type=Path,
        required=True,
        help="JSON list of responses returned one per rollout turn.",
    )
    parser.add_argument("--max-turns", type=int, default=2)
    args = parser.parse_args()

    responses = json.loads(args.responses_json.read_text(encoding="utf-8"))
    if not isinstance(responses, list):
        raise TypeError("--responses-json must contain a JSON list")

    def model_fn(_: str) -> str:
        if not responses:
            raise RuntimeError("No prerecorded responses left")
        return responses.pop(0)

    result = rollout_self_correction(
        prompt=args.prompt,
        label=args.label,
        model_fn=model_fn,
        max_turns=args.max_turns,
    )
    print(json.dumps(result, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
