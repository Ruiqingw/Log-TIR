from __future__ import annotations

import argparse
import json
import random
from pathlib import Path
from typing import Any

from sft_data import _normalize_sql, _render_schema, _resolve_spider_root


def _load_tables(tables_path: Path) -> dict[str, dict[str, Any]]:
    payload = json.loads(tables_path.read_text(encoding="utf-8"))
    return {entry["db_id"]: entry for entry in payload}


def _load_examples(path: Path) -> list[dict[str, Any]]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, list):
        raise TypeError(f"{path} must contain a JSON list")
    return payload


def _build_rl_prompt(question: str, schema_text: str) -> str:
    return (
        "You are a SQLite Text-to-SQL assistant.\n"
        "Given a database schema and a natural-language question, think briefly "
        "and then output one SQL query.\n"
        "You must answer using exactly two tags:\n"
        "<thought>...</thought>\n"
        "<action>...</action>\n"
        "The <action> section must contain only one executable SQLite query.\n\n"
        f"{schema_text}\n\n"
        f"Question: {question}"
    )


def build_grpo_prompts(
    spider_root: Path,
    output_path: Path,
    split: str = "train_spider",
    limit: int | None = 2000,
    seed: int = 42,
) -> dict[str, Any]:
    spider_root = _resolve_spider_root(spider_root)
    tables = _load_tables(spider_root / "tables.json")
    examples = _load_examples(spider_root / f"{split}.json")

    candidates = list(enumerate(examples))
    rng = random.Random(seed)
    rng.shuffle(candidates)
    if limit is not None:
        candidates = candidates[:limit]

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as handle:
        for source_index, example in candidates:
            schema_text = _render_schema(tables[example["db_id"]])
            label = {
                "db_id": example["db_id"],
                "db_path": str(
                    Path("database")
                    / example["db_id"]
                    / f"{example['db_id']}.sqlite"
                ),
                "gold_sql": _normalize_sql(example["query"]),
                "question": example["question"],
            }
            record = {
                "task": "text-to-sql-rl",
                "record_id": f"{split}:{source_index}",
                "datasource": "spider",
                "prompt": _build_rl_prompt(example["question"], schema_text),
                "label": json.dumps(label, ensure_ascii=False),
            }
            handle.write(json.dumps(record, ensure_ascii=False) + "\n")

    return {
        "output_path": str(output_path),
        "count": len(candidates),
        "split": split,
        "seed": seed,
    }


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Build Spider prompt/label JSONL for OpenRLHF GRPO."
    )
    parser.add_argument("--spider-root", type=Path, default=Path("data/spider"))
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("data/rl/spider_grpo_train_2000.jsonl"),
    )
    parser.add_argument("--split", default="train_spider")
    parser.add_argument("--limit", type=int, default=2000)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    report = build_grpo_prompts(
        spider_root=args.spider_root,
        output_path=args.output,
        split=args.split,
        limit=args.limit,
        seed=args.seed,
    )
    print(json.dumps(report, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
