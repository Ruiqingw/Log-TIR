from __future__ import annotations

import argparse
import json
import random
import re
from pathlib import Path
from typing import Any

from sandbox import execute_sql

TEACHER_SYSTEM_PROMPT = """You are an expert SQLite Text-to-SQL teacher creating training data for a smaller student model.

Given a SQLite database schema and a natural-language question, output exactly two tags:

<thought>your reasoning</thought>
<action>one SQLite query</action>

Rules for <thought>:
- Write authentic first-person reasoning in 2-4 sentences.
- Identify relevant tables/columns and WHY they are relevant.
- Mention required joins, filters, aggregations, ordering, or DISTINCT decisions.
- Vary your phrasing across examples; never begin with a fixed template like "I should use the relevant schema".
- Do NOT restate the question verbatim.
- Do NOT mention that you were "given" or "shown" the answer; you are reasoning from scratch.

Rules for <action>:
- Exactly one valid SQLite query, no markdown fences, no comments, no semicolons inside.
- Use only tables/columns present in the schema.
- Use double quotes for string literals (Spider convention).
- Prefer table aliases T1, T2, ... when joining.
- Output only the SQL, single line preferred.

Output ONLY the two tags. No preamble, no trailing text.

Example:

Schema:
Database: concert_singer
Tables:
- stadium: Stadium_ID, Location, Name, Capacity, Highest, Lowest, Average
- singer: Singer_ID, Name, Country, Song_Name, Song_release_year, Age, Is_male
- concert: concert_ID, concert_Name, Theme, Stadium_ID, Year
- singer_in_concert: concert_ID, Singer_ID
Primary keys: stadium.Stadium_ID, singer.Singer_ID, concert.concert_ID
Foreign keys: concert.Stadium_ID -> stadium.Stadium_ID, singer_in_concert.Singer_ID -> singer.Singer_ID, singer_in_concert.concert_ID -> concert.concert_ID

Question: How many singers older than 30 are from each country?

<thought>The question asks for a count grouped by country, so I need GROUP BY on singer.Country with COUNT. The age filter "older than 30" maps to a WHERE clause on singer.Age > 30. Only the singer table is needed; no join required.</thought>
<action>SELECT Country, COUNT(*) FROM singer WHERE Age > 30 GROUP BY Country</action>"""

FORMAT_ONLY_THOUGHT = (
    "Read the schema, identify relevant tables and columns, then write one SQLite "
    "query that answers the question."
)
TAGGED_RESPONSE_PATTERN = re.compile(
    r"^\s*<thought>(.*?)</thought>\s*<action>(.*?)</action>\s*$",
    re.DOTALL,
)


def _resolve_spider_root(spider_root: Path) -> Path:
    spider_root = spider_root.resolve()
    candidates = (
        spider_root,
        spider_root / "spider",
        spider_root / "spider_data",
    )
    for candidate in candidates:
        if (candidate / "tables.json").exists():
            return candidate
    raise FileNotFoundError(
        f"Could not find Spider tables.json under {spider_root}"
    )


def _render_schema(schema: dict[str, Any]) -> str:
    lines = [f"Database: {schema['db_id']}", "Tables:"]

    table_to_columns: dict[int, list[str]] = {
        idx: [] for idx in range(len(schema["table_names_original"]))
    }
    for table_idx, column_name in schema["column_names_original"]:
        if table_idx >= 0:
            table_to_columns[table_idx].append(column_name)

    for idx, table_name in enumerate(schema["table_names_original"]):
        columns = ", ".join(table_to_columns[idx]) or "(no columns)"
        lines.append(f"- {table_name}: {columns}")

    primary_keys = [
        _column_label(schema, column_idx) for column_idx in schema["primary_keys"]
    ]
    foreign_keys = [
        f"{_column_label(schema, src)} -> {_column_label(schema, dst)}"
        for src, dst in schema["foreign_keys"]
    ]
    lines.append(
        "Primary keys: " + (", ".join(primary_keys) if primary_keys else "(none)")
    )
    lines.append(
        "Foreign keys: " + (", ".join(foreign_keys) if foreign_keys else "(none)")
    )
    return "\n".join(lines)


def _column_label(schema: dict[str, Any], column_idx: int) -> str:
    table_idx, column_name = schema["column_names_original"][column_idx]
    if table_idx < 0:
        return column_name
    table_name = schema["table_names_original"][table_idx]
    return f"{table_name}.{column_name}"


def _normalize_sql(sql: str) -> str:
    return " ".join(sql.strip().split())


def _build_prompt(question: str, schema_text: str) -> str:
    return (
        "You are a SQLite Text-to-SQL assistant.\n"
        "Given a database schema and a natural-language question, think briefly and then output one SQL query.\n"
        "You must answer using exactly two tags:\n"
        "<thought>...</thought>\n"
        "<action>...</action>\n"
        "The <action> section must contain only one executable SQLite query.\n\n"
        f"{schema_text}\n\n"
        f"Question: {question}"
    )


def _build_teacher_user_message(question: str, schema_text: str) -> str:
    return f"Schema:\n{schema_text}\n\nQuestion: {question}"


def _build_response(sql: str) -> str:
    normalized_sql = _normalize_sql(sql)
    return f"<thought>{FORMAT_ONLY_THOUGHT}</thought>\n<action>{normalized_sql}</action>"


def parse_tagged_response(response: str) -> dict[str, str]:
    match = TAGGED_RESPONSE_PATTERN.match(response)
    if match is None:
        raise ValueError("Response must contain only <thought> and <action> tags")

    thought = match.group(1).strip()
    action = match.group(2).strip()
    if not thought:
        raise ValueError("<thought> must not be empty")
    if not action:
        raise ValueError("<action> must not be empty")
    if ";" in action:
        raise ValueError("<action> must not contain semicolons")
    return {"thought": thought, "action": action}


def validate_tagged_response(response: str) -> bool:
    try:
        parse_tagged_response(response)
    except ValueError:
        return False
    return True


def _load_tables(tables_path: Path) -> dict[str, dict[str, Any]]:
    payload = json.loads(tables_path.read_text(encoding="utf-8"))
    return {entry["db_id"]: entry for entry in payload}


def _load_examples(split_path: Path) -> list[dict[str, Any]]:
    payload = json.loads(split_path.read_text(encoding="utf-8"))
    if not isinstance(payload, list):
        raise TypeError(f"{split_path} must contain a JSON list")
    return payload


def build_sft_examples(
    spider_root: Path,
    output_path: Path,
    limit: int = 2000,
    seed: int = 42,
    include_train_others: bool = False,
    require_executable_gold: bool = False,
    teacher_requests_output: Path | None = None,
    timeout_s: float = 3.0,
) -> dict[str, Any]:
    spider_root = _resolve_spider_root(spider_root)
    tables = _load_tables(spider_root / "tables.json")

    split_specs = [("train_spider", spider_root / "train_spider.json")]
    if include_train_others:
        split_specs.append(("train_others", spider_root / "train_others.json"))

    candidates: list[tuple[str, int, dict[str, Any]]] = []
    for split_name, split_path in split_specs:
        for source_index, example in enumerate(_load_examples(split_path)):
            candidates.append((split_name, source_index, example))

    rng = random.Random(seed)
    rng.shuffle(candidates)

    records: list[dict[str, Any]] = []
    skipped_non_executable = 0
    for split_name, source_index, example in candidates:
        if len(records) >= limit:
            break

        schema = tables[example["db_id"]]
        schema_text = _render_schema(schema)
        gold_sql = _normalize_sql(example["query"])
        if require_executable_gold:
            db_path = (
                spider_root / "database" / example["db_id"] / f"{example['db_id']}.sqlite"
            )
            outcome = execute_sql(db_path, gold_sql, timeout_s=timeout_s)
            if not outcome.get("ok"):
                skipped_non_executable += 1
                continue

        prompt = _build_prompt(example["question"], schema_text)
        response = _build_response(gold_sql)
        records.append(
            {
                "task": "text-to-sql",
                "record_id": f"{split_name}:{source_index}",
                "split": split_name,
                "db_id": example["db_id"],
                "question": example["question"],
                "prompt": prompt,
                "response": response,
                "gold_sql": gold_sql,
                "thought_mode": "format_only",
                "teacher_user_message": _build_teacher_user_message(
                    example["question"], schema_text
                ),
            }
        )

    selected = records

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as handle:
        for record in selected:
            sft_record = {
                key: value
                for key, value in record.items()
                if key != "teacher_user_message"
            }
            handle.write(json.dumps(sft_record, ensure_ascii=False) + "\n")

    if teacher_requests_output is not None:
        teacher_requests_output.parent.mkdir(parents=True, exist_ok=True)
        with teacher_requests_output.open("w", encoding="utf-8") as handle:
            for record in selected:
                request = {
                    "task": "teacher_reasoning_generation",
                    "record_id": record["record_id"],
                    "db_id": record["db_id"],
                    "question": record["question"],
                    "messages": [
                        {"role": "system", "content": TEACHER_SYSTEM_PROMPT},
                        {
                            "role": "user",
                            "content": record["teacher_user_message"],
                        },
                    ],
                }
                handle.write(json.dumps(request, ensure_ascii=False) + "\n")

    return {
        "output_path": str(output_path),
        "count": len(selected),
        "seed": seed,
        "include_train_others": include_train_others,
        "require_executable_gold": require_executable_gold,
        "skipped_non_executable": skipped_non_executable,
        "teacher_requests_output": (
            str(teacher_requests_output) if teacher_requests_output is not None else None
        ),
    }


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Build Spider SFT cold-start data with <thought>/<action> formatting."
    )
    parser.add_argument(
        "--spider-root",
        type=Path,
        default=Path("data/spider"),
        help="Root directory that contains Spider data.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("data/sft/spider_sft_2000.jsonl"),
        help="Path to the generated JSONL file.",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=2000,
        help="Maximum number of output examples.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed used before sampling examples.",
    )
    parser.add_argument(
        "--include-train-others",
        action="store_true",
        help="Also include Spider train_others.json examples before sampling.",
    )
    parser.add_argument(
        "--require-executable-gold",
        action="store_true",
        help="Filter out examples whose gold SQL cannot execute in the local Spider DB.",
    )
    parser.add_argument(
        "--teacher-requests-output",
        type=Path,
        default=None,
        help="Optional JSONL path for teacher-model prompt requests.",
    )
    parser.add_argument(
        "--timeout",
        type=float,
        default=3.0,
        help="Per-query timeout when --require-executable-gold is enabled.",
    )
    args = parser.parse_args()

    report = build_sft_examples(
        spider_root=args.spider_root,
        output_path=args.output,
        limit=args.limit,
        seed=args.seed,
        include_train_others=args.include_train_others,
        require_executable_gold=args.require_executable_gold,
        teacher_requests_output=args.teacher_requests_output,
        timeout_s=args.timeout,
    )
    print(json.dumps(report, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
