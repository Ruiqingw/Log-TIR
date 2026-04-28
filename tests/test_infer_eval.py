from __future__ import annotations

import sqlite3
from pathlib import Path

from dataset_adapter import TextToSQLExample
from infer_eval import build_prompt, evaluate_responses, extract_sql


def _build_db(path: Path) -> None:
    conn = sqlite3.connect(path)
    conn.executescript(
        """
        create table items(name text);
        insert into items(name) values ('alpha'), ('beta');
        """
    )
    conn.commit()
    conn.close()


def test_extract_sql_prefers_action_tag() -> None:
    response = "<thought>Use count.</thought>\n<action>SELECT count(*) FROM items</action>"
    assert extract_sql(response) == "SELECT count(*) FROM items"


def test_build_prompt_includes_bird_evidence() -> None:
    example = TextToSQLExample(
        index=0,
        db_id="toy",
        question="How many rows?",
        gold_sql="SELECT count(*) FROM items",
        evidence="rows means records in items",
    )
    prompt = build_prompt(example, "Database: toy\nTables:\n- items: name")
    assert "Evidence: rows means records in items" in prompt


def test_evaluate_responses_reports_exec_match(tmp_path: Path) -> None:
    root = tmp_path / "spider_data"
    db_dir = root / "database" / "toy"
    db_dir.mkdir(parents=True)
    _build_db(db_dir / "toy.sqlite")
    examples = [
        TextToSQLExample(
            index=0,
            db_id="toy",
            question="How many rows?",
            gold_sql="SELECT count(*) FROM items",
        )
    ]
    responses = ["<thought>Use count.</thought>\n<action>SELECT count(*) FROM items</action>"]
    report = evaluate_responses(
        "spider", root, examples, responses, timeout_s=3.0, workers=2
    )
    assert report["total"] == 1
    assert report["matched"] == 1
    assert report["accuracy"] == 1.0
