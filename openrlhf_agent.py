from __future__ import annotations

import json
import os
import re
from typing import Any

from openrlhf_reward import _db_path_from_label, _extra_logs_from_score, score_response
from sandbox import execute_sql
from sft_data import parse_tagged_response

try:
    from openrlhf.utils.agent import AgentInstanceBase, MultiTurnAgentExecutor
except Exception:  # pragma: no cover - only used when OpenRLHF is absent locally.
    class AgentInstanceBase:  # type: ignore[no-redef]
        pass

    class MultiTurnAgentExecutor:  # type: ignore[no-redef]
        def __init__(self, agent_cls: type[AgentInstanceBase]) -> None:
            self.agent_cls = agent_cls


def _tensor(value: float) -> Any:
    try:
        import torch

        return torch.tensor(float(value), dtype=torch.float32)
    except Exception:  # pragma: no cover - OpenRLHF server will have torch.
        return float(value)


def _as_float(value: Any) -> float:
    if hasattr(value, "item"):
        return float(value.item())
    return float(value)


TAGGED_RESPONSE_SEARCH = re.compile(
    r"<thought>.*?</thought>\s*<action>.*?</action>",
    re.DOTALL,
)


def _response_from_action_text(action_text: str, states: dict[str, Any]) -> str:
    response = action_text.strip()
    for key in ("observation_text", "observation", "prompt"):
        prefix = states.get(key)
        if isinstance(prefix, str) and prefix and response.startswith(prefix):
            response = response[len(prefix) :].strip()
            break

    try:
        parse_tagged_response(response)
        return response
    except ValueError:
        pass

    matches = list(TAGGED_RESPONSE_SEARCH.finditer(response))
    for match in reversed(matches):
        candidate = match.group(0).strip()
        try:
            parse_tagged_response(candidate)
        except ValueError:
            continue
        return candidate
    return response


def _label_from_states(states: dict[str, Any]) -> dict[str, Any]:
    label = states.get("label", {})
    if isinstance(label, str):
        return json.loads(label)
    return dict(label)


def _feedback_from_score(
    response: str,
    label: dict[str, Any],
    score: dict[str, Any],
    timeout_s: float,
) -> str:
    if not score["format_match"]:
        return (
            "Execution feedback:\n"
            f"The previous response had an invalid format: {score['error']}\n"
            "Return exactly <thought>...</thought> and <action>one SQLite query</action>."
        )

    if not score["no_error"]:
        return (
            "Execution feedback:\n"
            "The previous SQL failed in SQLite.\n"
            f"Error: {score['error']}\n"
            "Revise the SQL and answer again with exactly <thought> and <action>."
        )

    try:
        action_sql = parse_tagged_response(response)["action"]
        execution = execute_sql(_db_path_from_label(label), action_sql, timeout_s=timeout_s)
        if not execution.get("rows", []):
            return (
                "Execution feedback:\n"
                "The previous SQL executed but returned an empty result.\n"
                "Check whether joins or filters are too restrictive, then answer again "
                "with exactly <thought> and <action>."
            )
    except Exception:
        pass

    return (
        "Execution feedback:\n"
        "The previous SQL executed but did not match the expected result.\n"
        "Revise the selected tables, joins, filters, aggregation, ordering, or DISTINCT "
        "decision, then answer again with exactly <thought> and <action>."
    )


class AgentInstance(AgentInstanceBase):
    def __init__(self, *args: Any, **kwargs: Any) -> None:
        del args, kwargs
        self.step_idx = 0
        self.max_turns = int(os.environ.get("LOGTIR_AGENT_MAX_TURNS", "2"))
        self.timeout_s = float(os.environ.get("LOGTIR_AGENT_TIMEOUT", "3.0"))
        self.turn_penalty = float(os.environ.get("LOGTIR_AGENT_TURN_PENALTY", "0.05"))
        self.turn1_exec_match = False

    async def reset(self, states: dict[str, Any], **kwargs: Any) -> dict[str, Any]:
        del kwargs
        self.step_idx = 0
        self.turn1_exec_match = False
        observation = states.get("observation") or states.get("prompt") or ""
        return {"observation": observation}

    async def step(self, states: dict[str, Any], **kwargs: Any) -> dict[str, Any]:
        del kwargs
        self.step_idx += 1
        label = _label_from_states(states)
        action_text = states.get("action_text") or states.get("response") or ""
        response = _response_from_action_text(str(action_text), states)
        score = score_response(response, label, timeout_s=self.timeout_s)
        if self.step_idx == 1:
            self.turn1_exec_match = bool(score["exec_match"])
        done = bool(score["exec_match"]) or self.step_idx >= self.max_turns

        raw_reward = float(score["reward"])
        reward_value = raw_reward if done else 0.0
        if done:
            reward_value = max(0.0, reward_value - self.turn_penalty * (self.step_idx - 1))
        reward = _tensor(reward_value)
        feedback = ""
        if not done:
            feedback = "\n\n" + _feedback_from_score(
                response=response,
                label=label,
                score=score,
                timeout_s=self.timeout_s,
            ) + "\n\n"

        extra_logs = _extra_logs_from_score(response, score)
        extra_logs.update(
            {
                "turn": float(self.step_idx),
                "terminal_rate": float(done),
                "avg_turns_used": float(self.step_idx),
                "turn1_exec_match_rate": float(self.step_idx == 1 and score["exec_match"]),
                "turn2_exec_match_rate": float(self.step_idx == 2 and score["exec_match"]),
                "self_correction_rate": float(
                    self.step_idx > 1 and (not self.turn1_exec_match) and score["exec_match"]
                ),
                "raw_reward": raw_reward,
                "turn_penalty": raw_reward - reward_value if done else 0.0,
                "reward": reward_value,
            }
        )

        return {
            "rewards": reward,
            "scores": reward,
            "environment_feedback": feedback,
            "done": done,
            "sampling_params": states.get("sampling_params", None),
            "extra_logs": extra_logs,
        }


class AgentExecutor(MultiTurnAgentExecutor):
    def __init__(self) -> None:
        super().__init__(AgentInstance)
