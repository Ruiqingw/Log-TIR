from __future__ import annotations

import json
import os
from functools import lru_cache
from pathlib import Path
from typing import Any

from eval import _normalize_rows, _should_preserve_order
from sft_data import parse_tagged_response
from sandbox import execute_sql

FORMAT_REWARD = 0.1
NO_ERROR_REWARD = 0.2
EXEC_MATCH_REWARD = 1.0


def _response_from_query(query: str, prompt: str) -> str:
    if query.startswith(prompt):
        return query[len(prompt) :].strip()
    return query.strip()


def _db_path_from_label(label: dict[str, Any]) -> Path:
    if label.get("db_path"):
        db_path = Path(label["db_path"])
        if db_path.is_absolute():
            return db_path
        if db_path.exists():
            return db_path
        spider_root = Path(os.environ.get("SPIDER_ROOT", "data/spider/spider_data"))
        return spider_root / db_path
    spider_root = Path(os.environ.get("SPIDER_ROOT", "data/spider/spider_data"))
    db_id = label["db_id"]
    return spider_root / "database" / db_id / f"{db_id}.sqlite"


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


def score_response(response: str, label: str | dict[str, Any], timeout_s: float = 3.0) -> dict[str, Any]:
    parsed_label = json.loads(label) if isinstance(label, str) else label
    reward = 0.0
    details: dict[str, Any] = {
        "format_match": False,
        "no_error": False,
        "exec_match": False,
        "error": "",
    }

    try:
        parsed = parse_tagged_response(response)
    except ValueError as exc:
        details["error"] = f"format: {exc}"
        return {"reward": reward, **details}

    reward += FORMAT_REWARD
    details["format_match"] = True
    action_sql = parsed["action"]
    db_path = _db_path_from_label(parsed_label)

    execution = execute_sql(db_path, action_sql, timeout_s=timeout_s)
    if not execution.get("ok"):
        details["error"] = execution.get("error", "")
        return {"reward": reward, **details}

    reward += NO_ERROR_REWARD
    details["no_error"] = True

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

    return {"reward": reward, **details}


def reward_func(queries, prompts, labels):  # type: ignore[no-untyped-def]
    import torch

    scores = []
    for query, prompt, label in zip(queries, prompts, labels):
        response = _response_from_query(query, prompt)
        scores.append(score_response(response, label)["reward"])
    return torch.tensor(scores, dtype=torch.float32)
