from __future__ import annotations

import json
import os
import re
from functools import lru_cache
from pathlib import Path
from typing import Any

from eval import _normalize_rows, _should_preserve_order
from sft_data import parse_tagged_response
from sandbox import execute_sql

FORMAT_REWARD = 0.1
NO_ERROR_REWARD = 0.2
EXEC_MATCH_REWARD = 1.0
TAGGED_RESPONSE_SEARCH = re.compile(
    r"<thought>.*?</thought>\s*<action>.*?</action>",
    re.DOTALL,
)
REPO_ROOT = Path(__file__).resolve().parent
_REWARD_DEBUG_COUNT = 0


def _last_parseable_tagged_response(text: str) -> str | None:
    for match in reversed(list(TAGGED_RESPONSE_SEARCH.finditer(text))):
        candidate = match.group(0).strip()
        try:
            parse_tagged_response(candidate)
        except ValueError:
            continue
        return candidate
    return None


def _parseable_or_recovered_response(text: str) -> str | None:
    candidate = text.strip()
    try:
        parse_tagged_response(candidate)
    except ValueError:
        return _last_parseable_tagged_response(candidate)
    return candidate


def _response_from_query(query: str, prompt: str) -> str:
    candidates: list[str] = []
    if query.startswith(prompt):
        candidates.append(query[len(prompt) :])
    candidates.append(query)

    for candidate in candidates:
        recovered = _parseable_or_recovered_response(candidate)
        if recovered is not None:
            return recovered

    return candidates[0].strip()


def _spider_root() -> Path:
    spider_root = Path(os.environ.get("SPIDER_ROOT", "data/spider/spider_data"))
    if spider_root.is_absolute():
        return spider_root
    return REPO_ROOT / spider_root


def _db_path_from_label(label: dict[str, Any]) -> Path:
    if label.get("db_path"):
        db_path = Path(label["db_path"])
        if db_path.is_absolute():
            return db_path
        if db_path.exists():
            return db_path
        repo_relative = REPO_ROOT / db_path
        if repo_relative.exists():
            return repo_relative
        return _spider_root() / db_path
    db_id = label["db_id"]
    return _spider_root() / "database" / db_id / f"{db_id}.sqlite"


def _debug_path() -> Path:
    path = Path(os.environ.get("LOGTIR_REWARD_DEBUG_PATH", "logs/reward_debug.jsonl"))
    if path.is_absolute():
        return path
    return REPO_ROOT / path


def _reward_timeout_s() -> float:
    timeout = os.environ.get("LOGTIR_REWARD_TIMEOUT")
    if timeout is None:
        timeout = os.environ.get("LOGTIR_AGENT_TIMEOUT", "3.0")
    return float(timeout)


def _maybe_debug_reward(
    query: str,
    prompt: str,
    label: str | dict[str, Any],
    response: str,
    score: dict[str, Any],
) -> None:
    global _REWARD_DEBUG_COUNT
    limit = int(os.environ.get("LOGTIR_REWARD_DEBUG_LIMIT", "0") or 0)
    if limit <= 0 or _REWARD_DEBUG_COUNT >= limit:
        return
    _REWARD_DEBUG_COUNT += 1

    parsed_label = json.loads(label) if isinstance(label, str) else label
    db_path = _db_path_from_label(parsed_label)
    payload = {
        "index": _REWARD_DEBUG_COUNT,
        "cwd": os.getcwd(),
        "repo_root": str(REPO_ROOT),
        "reward_file": __file__,
        "query_startswith_prompt": query.startswith(prompt),
        "query_len": len(query),
        "prompt_len": len(prompt),
        "query_tail": query[-1000:],
        "prompt_tail": prompt[-500:],
        "response": response,
        "label": parsed_label,
        "db_path": str(db_path),
        "db_exists": db_path.exists(),
        "score": score,
    }
    path = _debug_path()
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(payload, ensure_ascii=False, sort_keys=True) + "\n")


@lru_cache(maxsize=8192)
def _execute_gold_sql_cached(db_path: str, gold_sql: str, timeout_s: float) -> str:
    result = execute_sql(db_path, gold_sql, timeout_s=timeout_s)
    return json.dumps(result, ensure_ascii=False, sort_keys=True)


def _execution_match_from_results(
    pred_sql: str,
    pred_result: dict[str, Any],
    gold_sql: str,
    gold_result: dict[str, Any],
) -> bool:
    if not pred_result.get("ok") or not gold_result.get("ok"):
        return False
    preserve_order = _should_preserve_order(pred_sql) or _should_preserve_order(gold_sql)
    pred_rows = _normalize_rows(pred_result.get("rows", []), preserve_order)
    gold_rows = _normalize_rows(gold_result.get("rows", []), preserve_order)
    return pred_rows == gold_rows


def _classify_error(error: str) -> str:
    lowered = error.lower()
    if not error:
        return ""
    if "no such column" in lowered or "no such table" in lowered:
        return "schema_hallucination"
    if "syntax error" in lowered:
        return "syntax"
    if "permissionerror" in lowered or "read-only" in lowered or "not read-only" in lowered:
        return "unsafe_sql"
    if "timeout" in lowered or "timed out" in lowered:
        return "timeout"
    return "execution"


def _action_sql_from_response(response: str) -> str:
    try:
        return parse_tagged_response(response)["action"]
    except ValueError:
        return ""


def _extra_logs_from_score(response: str, score: dict[str, Any]) -> dict[str, float]:
    action_sql = str(score.get("action_sql") or _action_sql_from_response(response))
    error_category = str(score.get("error_category") or "")
    reward = float(score["reward"])
    return {
        "format_match_rate": float(score["format_match"]),
        "no_error_rate": float(score["no_error"]),
        "exec_match_rate": float(score["exec_match"]),
        "format_only_reward_share": float(
            score["format_match"] and not score["no_error"] and reward == FORMAT_REWARD
        ),
        "err_schema_hallucination_rate": float(error_category == "schema_hallucination"),
        "err_syntax_rate": float(error_category == "syntax"),
        "err_empty_result_rate": float(score.get("empty_result", False)),
        "err_wrong_result_rate": float(score.get("wrong_result", False)),
        "response_length_chars": float(len(response)),
        "avg_action_sql_length": float(len(action_sql)),
    }


def score_response(response: str, label: str | dict[str, Any], timeout_s: float = 3.0) -> dict[str, Any]:
    parsed_label = json.loads(label) if isinstance(label, str) else label
    reward = 0.0
    details: dict[str, Any] = {
        "format_match": False,
        "no_error": False,
        "exec_match": False,
        "error": "",
        "error_category": "",
        "action_sql": "",
        "empty_result": False,
        "wrong_result": False,
    }

    try:
        parsed = parse_tagged_response(response)
    except ValueError as exc:
        details["error"] = f"format: {exc}"
        details["error_category"] = "format"
        return {"reward": reward, **details}

    reward += FORMAT_REWARD
    details["format_match"] = True
    action_sql = parsed["action"]
    details["action_sql"] = action_sql
    db_path = _db_path_from_label(parsed_label)

    execution = execute_sql(db_path, action_sql, timeout_s=timeout_s)
    if not execution.get("ok"):
        details["error"] = execution.get("error", "")
        details["error_category"] = _classify_error(details["error"])
        return {"reward": reward, **details}

    reward += NO_ERROR_REWARD
    details["no_error"] = True
    details["empty_result"] = not bool(execution.get("rows", []))

    gold_result = json.loads(
        _execute_gold_sql_cached(str(db_path), parsed_label["gold_sql"], timeout_s)
    )
    if _execution_match_from_results(
        pred_sql=action_sql,
        pred_result=execution,
        gold_sql=parsed_label["gold_sql"],
        gold_result=gold_result,
    ):
        reward += EXEC_MATCH_REWARD
        details["exec_match"] = True
    else:
        details["error"] = execution.get("error") or gold_result.get("error", "")
        details["wrong_result"] = True
        details["error_category"] = "empty_result" if details["empty_result"] else "wrong_result"

    return {"reward": reward, **details}


def reward_func(queries, prompts, labels):  # type: ignore[no-untyped-def]
    payloads = []
    for query, prompt, label in zip(queries, prompts, labels):
        response = _response_from_query(query, prompt)
        score = score_response(response, label, timeout_s=_reward_timeout_s())
        _maybe_debug_reward(query, prompt, label, response, score)
        reward = float(score["reward"])
        payloads.append(
            {
                "rewards": reward,
                "scores": reward,
                "extra_logs": _extra_logs_from_score(response, score),
            }
        )

    if len(payloads) == 1:
        return payloads[0]
    return payloads
