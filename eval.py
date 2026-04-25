from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path
from typing import Any

from dataset_adapter import (
    load_text_to_sql_examples,
    resolve_dataset_root,
    resolve_db_path,
    resolve_split_file,
)
from sandbox import execute_sql


def _resolve_spider_root(spider_root: Path) -> Path:
    spider_root = spider_root.resolve()
    candidates = (
        spider_root,
        spider_root / "spider",
        spider_root / "spider_data",
    )
    for candidate in candidates:
        if (candidate / "database").exists():
            return candidate
    raise FileNotFoundError(
        f"Could not find Spider database directory under {spider_root}"
    )


def _resolve_dev_file(dev_path: Path | None, spider_root: Path) -> Path:
    if dev_path is not None:
        if dev_path.exists():
            return dev_path.resolve()
        raise FileNotFoundError(f"Spider dev file not found: {dev_path}")

    candidates = (
        spider_root / "dev.json",
        spider_root / "spider" / "dev.json",
        spider_root / "spider_data" / "dev.json",
    )
    for candidate in candidates:
        if candidate.exists():
            return candidate.resolve()

    tried = ", ".join(str(path) for path in candidates)
    raise FileNotFoundError(f"Could not find Spider dev file. Tried: {tried}")


def _normalize_cell(value: Any) -> Any:
    if value is None:
        return ("null", None)
    if isinstance(value, bytes):
        return ("str", value.decode("utf-8", errors="replace").strip())
    if isinstance(value, bool):
        return ("bool", value)
    if isinstance(value, int | float):
        return ("number", round(float(value), 8))
    if isinstance(value, str):
        return ("str", value.strip())
    return (type(value).__name__, str(value))


def _normalize_rows(
    rows: list[list[Any]] | list[tuple[Any, ...]],
    preserve_order: bool,
) -> Any:
    normalized = [tuple(_normalize_cell(cell) for cell in row) for row in rows]
    if preserve_order:
        return normalized
    return Counter(normalized)


def _should_preserve_order(sql: str) -> bool:
    lowered = " ".join(sql.lower().split())
    return " order by " in f" {lowered} "


def execution_match(
    pred_sql: str,
    gold_sql: str,
    db_path: Path,
    timeout_s: float = 3.0,
) -> dict[str, Any]:
    pred_result = execute_sql(db_path, pred_sql, timeout_s=timeout_s)
    gold_result = execute_sql(db_path, gold_sql, timeout_s=timeout_s)

    if not pred_result.get("ok") or not gold_result.get("ok"):
        return {
            "match": False,
            "pred_result": pred_result,
            "gold_result": gold_result,
        }

    preserve_order = _should_preserve_order(pred_sql) or _should_preserve_order(gold_sql)
    pred_rows = _normalize_rows(pred_result.get("rows", []), preserve_order)
    gold_rows = _normalize_rows(gold_result.get("rows", []), preserve_order)
    return {
        "match": pred_rows == gold_rows,
        "pred_result": pred_result,
        "gold_result": gold_result,
    }


def _load_predictions(
    pred_path: Path | None,
    gold_sql: list[str],
    use_gold_predictions: bool,
) -> list[str]:
    if use_gold_predictions:
        return gold_sql
    if pred_path is None:
        raise ValueError("Provide --pred-file or pass --use-gold-predictions")

    if pred_path.suffix == ".json":
        with pred_path.open("r", encoding="utf-8") as handle:
            payload = json.load(handle)
        if not isinstance(payload, list):
            raise TypeError("JSON prediction file must be a list")
        if payload and isinstance(payload[0], dict):
            return [item["query"] for item in payload]
        return [str(item) for item in payload]

    with pred_path.open("r", encoding="utf-8") as handle:
        return [line.rstrip("\n") for line in handle]

def evaluate_dataset(
    dev_path: Path,
    spider_root: Path,
    pred_path: Path | None,
    use_gold_predictions: bool,
    timeout_s: float,
    limit: int | None,
    dataset: str = "spider",
) -> dict[str, Any]:
    examples = load_text_to_sql_examples(dataset, dev_path)
    if limit is not None:
        examples = examples[:limit]

    predictions = _load_predictions(
        pred_path,
        [example.gold_sql for example in examples],
        use_gold_predictions,
    )
    if len(predictions) != len(examples):
        raise ValueError(
            f"Prediction count {len(predictions)} does not match example count {len(examples)}"
        )

    matched = 0
    failures: list[dict[str, Any]] = []

    for index, (example, pred_sql) in enumerate(zip(examples, predictions)):
        db_path = resolve_db_path(dataset, spider_root, example.db_id)
        outcome = execution_match(pred_sql, example.gold_sql, db_path, timeout_s)
        if outcome["match"]:
            matched += 1
            continue

        failures.append(
            {
                "index": index,
                "db_id": example.db_id,
                "question": example.question,
                "gold_sql": example.gold_sql,
                "pred_sql": pred_sql,
                "pred_error": outcome["pred_result"].get("error", ""),
                "gold_error": outcome["gold_result"].get("error", ""),
            }
        )

    total = len(examples)
    return {
        "total": total,
        "matched": matched,
        "accuracy": (matched / total) if total else 0.0,
        "failures": failures,
    }


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Execution-match evaluator for Spider/BIRD-style dev sets."
    )
    parser.add_argument(
        "--dataset",
        choices=["spider", "bird"],
        default="spider",
        help="Dataset format to evaluate.",
    )
    parser.add_argument(
        "--spider-root",
        type=Path,
        default=Path("data/spider"),
        help="Root directory of the unpacked Spider dataset.",
    )
    parser.add_argument(
        "--data-root",
        type=Path,
        default=None,
        help="Root directory of the dataset. Overrides --spider-root.",
    )
    parser.add_argument(
        "--dev-file",
        type=Path,
        default=None,
        help="Optional explicit path to dev.json.",
    )
    parser.add_argument(
        "--pred-file",
        type=Path,
        default=None,
        help="Prediction file: .txt one SQL per line or .json list.",
    )
    parser.add_argument(
        "--use-gold-predictions",
        action="store_true",
        help="Use gold SQL as predictions to validate the evaluator.",
    )
    parser.add_argument(
        "--timeout",
        type=float,
        default=3.0,
        help="Per-query timeout in seconds.",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Optional limit for smoke tests.",
    )
    parser.add_argument(
        "--failures-out",
        type=Path,
        default=None,
        help="Optional path to write mismatches as JSON.",
    )
    args = parser.parse_args()

    data_root_arg = args.data_root if args.data_root is not None else args.spider_root
    if args.dataset == "spider":
        spider_root = _resolve_spider_root(data_root_arg)
        dev_file = _resolve_dev_file(args.dev_file, spider_root)
    else:
        spider_root = resolve_dataset_root(args.dataset, data_root_arg)
        dev_file = resolve_split_file(args.dataset, spider_root, "dev", args.dev_file)
    report = evaluate_dataset(
        dev_path=dev_file,
        spider_root=spider_root,
        pred_path=args.pred_file,
        use_gold_predictions=args.use_gold_predictions,
        timeout_s=args.timeout,
        limit=args.limit,
        dataset=args.dataset,
    )

    if args.failures_out is not None:
        args.failures_out.write_text(
            json.dumps(report["failures"], ensure_ascii=False, indent=2),
            encoding="utf-8",
        )

    summary = {key: value for key, value in report.items() if key != "failures"}
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    if report["failures"]:
        print(
            "First failure:",
            json.dumps(report["failures"][0], ensure_ascii=False),
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
