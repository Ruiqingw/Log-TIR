from __future__ import annotations

import json
from pathlib import Path

from sft_data import (
    FORMAT_ONLY_THOUGHT,
    _build_response,
    _render_schema,
    build_sft_examples,
    parse_tagged_response,
    validate_tagged_response,
)


def _write_spider_fixture(root: Path) -> Path:
    spider_root = root / "spider_data"
    spider_root.mkdir(parents=True)

    tables = [
        {
            "db_id": "toy_db",
            "table_names_original": ["items", "orders"],
            "table_names": ["items", "orders"],
            "column_names_original": [
                [-1, "*"],
                [0, "item_id"],
                [0, "name"],
                [1, "order_id"],
                [1, "item_id"],
            ],
            "column_names": [
                [-1, "*"],
                [0, "item_id"],
                [0, "name"],
                [1, "order_id"],
                [1, "item_id"],
            ],
            "column_types": ["text", "number", "text", "number", "number"],
            "primary_keys": [1, 3],
            "foreign_keys": [[4, 1]],
        }
    ]
    train_spider = [
        {
            "db_id": "toy_db",
            "question": "List all item names.",
            "query": "SELECT name FROM items",
            "sql": {},
            "query_toks": [],
            "query_toks_no_value": [],
            "question_toks": [],
        },
        {
            "db_id": "toy_db",
            "question": "Count all orders.",
            "query": "SELECT count(*) FROM orders",
            "sql": {},
            "query_toks": [],
            "query_toks_no_value": [],
            "question_toks": [],
        },
    ]
    (spider_root / "tables.json").write_text(json.dumps(tables), encoding="utf-8")
    (spider_root / "train_spider.json").write_text(
        json.dumps(train_spider), encoding="utf-8"
    )
    (spider_root / "train_others.json").write_text("[]", encoding="utf-8")
    return spider_root


def test_render_schema_contains_tables_and_keys() -> None:
    schema = {
        "db_id": "toy_db",
        "table_names_original": ["items"],
        "column_names_original": [[-1, "*"], [0, "item_id"], [0, "name"]],
        "primary_keys": [1],
        "foreign_keys": [],
    }
    rendered = _render_schema(schema)
    assert "Database: toy_db" in rendered
    assert "- items: item_id, name" in rendered
    assert "Primary keys: items.item_id" in rendered


def test_build_response_produces_valid_tagged_format() -> None:
    response = _build_response("SELECT name FROM items")
    parsed = parse_tagged_response(response)
    assert parsed["thought"] == FORMAT_ONLY_THOUGHT
    assert parsed["action"] == "SELECT name FROM items"
    assert validate_tagged_response(response) is True


def test_build_response_does_not_leak_gold_sql_tables_into_thought() -> None:
    response = _build_response("SELECT secret_table.answer FROM secret_table")
    parsed = parse_tagged_response(response)
    assert "secret_table" not in parsed["thought"]


def test_validate_tagged_response_rejects_missing_action() -> None:
    assert validate_tagged_response("<thought>only thought</thought>") is False


def test_validate_tagged_response_rejects_action_semicolon() -> None:
    response = "<thought>think</thought>\n<action>SELECT 1;</action>"
    assert validate_tagged_response(response) is False


def test_validate_tagged_response_rejects_trailing_text() -> None:
    response = "<thought>think</thought>\n<action>SELECT 1</action>\nextra"
    assert validate_tagged_response(response) is False


def test_build_sft_examples_writes_jsonl(tmp_path: Path) -> None:
    spider_root = _write_spider_fixture(tmp_path)
    output_path = tmp_path / "out" / "sft.jsonl"
    report = build_sft_examples(
        spider_root=spider_root,
        output_path=output_path,
        limit=1,
        seed=7,
    )
    assert report["count"] == 1
    lines = output_path.read_text(encoding="utf-8").strip().splitlines()
    assert len(lines) == 1
    record = json.loads(lines[0])
    assert record["task"] == "text-to-sql"
    assert record["record_id"].startswith("train_spider:")
    assert record["db_id"] == "toy_db"
    assert record["thought_mode"] == "format_only"
    assert "<thought>" in record["response"]
    assert "<action>" in record["response"]


def test_build_sft_examples_can_write_teacher_requests(tmp_path: Path) -> None:
    spider_root = _write_spider_fixture(tmp_path)
    output_path = tmp_path / "out" / "sft.jsonl"
    teacher_path = tmp_path / "out" / "teacher.jsonl"
    report = build_sft_examples(
        spider_root=spider_root,
        output_path=output_path,
        teacher_requests_output=teacher_path,
        limit=1,
        seed=7,
    )
    assert report["teacher_requests_output"] == str(teacher_path)
    request = json.loads(teacher_path.read_text(encoding="utf-8").splitlines()[0])
    assert request["task"] == "teacher_reasoning_generation"
    assert "gold_sql" not in request
    assert request["messages"][0]["role"] == "system"
    assert "Schema:" in request["messages"][1]["content"]
