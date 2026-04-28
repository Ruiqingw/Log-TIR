from __future__ import annotations

import argparse
import json
import sys
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from dataset_adapter import (
    TextToSQLExample,
    load_text_to_sql_examples,
    resolve_dataset_root,
    resolve_db_path,
    resolve_split_file,
)
from eval import _normalize_rows, _should_preserve_order
from infer_eval import extract_sql
from sandbox import execute_sql


def _load_known_gold_failures(path: Path | None) -> dict[int, dict[str, Any]]:
    if path is None:
        return {}
    failures = json.loads(path.read_text(encoding="utf-8"))
    return {int(item["index"]): item for item in failures}


def _gold_result_from_failure(failure: dict[str, Any]) -> dict[str, Any]:
    return {
        "ok": False,
        "error": failure.get("gold_error") or failure.get("pred_error") or "",
    }


def _evaluate_gold_results(
    *,
    dataset: str,
    root: Path,
    examples: list[TextToSQLExample],
    timeout_s: float,
    workers: int,
    known_gold_failures: dict[int, dict[str, Any]],
) -> list[dict[str, Any]]:
    def evaluate_one(example: TextToSQLExample) -> dict[str, Any]:
        if example.index in known_gold_failures:
            return _gold_result_from_failure(known_gold_failures[example.index])
        db_path = resolve_db_path(dataset, root, example.db_id)
        return execute_sql(db_path, example.gold_sql, timeout_s=timeout_s)

    if workers <= 1:
        return [evaluate_one(example) for example in examples]
    with ThreadPoolExecutor(max_workers=workers) as executor:
        return list(executor.map(evaluate_one, examples))


def _row_match(
    *,
    pred_sql: str,
    gold_sql: str,
    pred_result: dict[str, Any],
    gold_result: dict[str, Any],
) -> bool:
    if not pred_result.get("ok") or not gold_result.get("ok"):
        return False
    preserve_order = _should_preserve_order(pred_sql) or _should_preserve_order(gold_sql)
    pred_rows = _normalize_rows(pred_result.get("rows", []), preserve_order)
    gold_rows = _normalize_rows(gold_result.get("rows", []), preserve_order)
    return pred_rows == gold_rows


def _evaluate_saved_report(
    *,
    dataset: str,
    root: Path,
    examples: list[TextToSQLExample],
    source_path: Path,
    gold_results: list[dict[str, Any]],
    timeout_s: float,
    workers: int,
) -> dict[str, Any]:
    source = json.loads(source_path.read_text(encoding="utf-8"))
    source_results = source["results"]
    if len(source_results) != len(examples):
        raise ValueError(
            f"{source_path} has {len(source_results)} results, expected {len(examples)}"
        )

    def evaluate_one(
        payload: tuple[TextToSQLExample, dict[str, Any], dict[str, Any]],
    ) -> dict[str, Any]:
        example, source_row, gold_result = payload
        response = str(source_row.get("response", ""))
        pred_sql = str(source_row.get("pred_sql") or extract_sql(response))
        if gold_result.get("ok"):
            db_path = resolve_db_path(dataset, root, example.db_id)
            pred_result = execute_sql(db_path, pred_sql, timeout_s=timeout_s)
        else:
            pred_result = {
                "ok": False,
                "error": "Skipped because gold SQL failed under this timeout.",
            }
        match = _row_match(
            pred_sql=pred_sql,
            gold_sql=example.gold_sql,
            pred_result=pred_result,
            gold_result=gold_result,
        )
        return {
            "index": example.index,
            "db_id": example.db_id,
            "question": example.question,
            "gold_sql": example.gold_sql,
            "response": response,
            "pred_sql": pred_sql,
            "match": match,
            "pred_error": pred_result.get("error", ""),
            "gold_error": gold_result.get("error", ""),
        }

    payloads = list(zip(examples, source_results, gold_results))
    if workers <= 1:
        results = [evaluate_one(payload) for payload in payloads]
    else:
        with ThreadPoolExecutor(max_workers=workers) as executor:
            results = list(executor.map(evaluate_one, payloads))

    total = len(results)
    matched = sum(int(row["match"]) for row in results)
    gold_timeout_count = sum(
        int(str(result.get("error", "")).startswith("TimeoutExpired"))
        for result in gold_results
    )
    return {
        "dataset": dataset,
        "source_report": str(source_path),
        "timeout": timeout_s,
        "workers": workers,
        "total": total,
        "matched": matched,
        "accuracy": matched / total if total else 0.0,
        "gold_failures": sum(int(not result.get("ok")) for result in gold_results),
        "gold_timeout_count": gold_timeout_count,
        "results": results,
    }


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Re-evaluate saved infer_eval responses with a new SQL timeout."
    )
    parser.add_argument("--dataset", choices=["spider", "bird"], required=True)
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--split", default="dev")
    parser.add_argument("--timeout", type=float, required=True)
    parser.add_argument("--workers", type=int, default=1)
    parser.add_argument("--gold-failures-file", type=Path, default=None)
    parser.add_argument(
        "--input-output",
        action="append",
        nargs=2,
        metavar=("INPUT_JSON", "OUTPUT_JSON"),
        required=True,
        help="Saved infer_eval JSON and destination JSON.",
    )
    args = parser.parse_args()

    root = resolve_dataset_root(args.dataset, args.data_root)
    split_file = resolve_split_file(args.dataset, root, args.split)
    examples = load_text_to_sql_examples(args.dataset, split_file)
    known_gold_failures = _load_known_gold_failures(args.gold_failures_file)
    gold_results = _evaluate_gold_results(
        dataset=args.dataset,
        root=root,
        examples=examples,
        timeout_s=args.timeout,
        workers=args.workers,
        known_gold_failures=known_gold_failures,
    )

    summaries = []
    for input_json, output_json in args.input_output:
        report = _evaluate_saved_report(
            dataset=args.dataset,
            root=root,
            examples=examples,
            source_path=Path(input_json),
            gold_results=gold_results,
            timeout_s=args.timeout,
            workers=args.workers,
        )
        output_path = Path(output_json)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(
            json.dumps(report, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
        summaries.append({key: value for key, value in report.items() if key != "results"})

    print(json.dumps(summaries, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
