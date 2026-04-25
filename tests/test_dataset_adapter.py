from __future__ import annotations

import json
import sqlite3
from pathlib import Path

from dataset_adapter import (
    load_text_to_sql_examples,
    resolve_dataset_root,
    resolve_db_path,
    resolve_split_file,
)


def test_load_bird_examples_accepts_uppercase_sql_and_evidence(tmp_path: Path) -> None:
    dev_path = tmp_path / "dev.json"
    dev_path.write_text(
        json.dumps(
            [
                {
                    "db_id": "toy",
                    "question": "How many rows?",
                    "SQL": " select count(*) from items ",
                    "evidence": "items means rows",
                }
            ]
        ),
        encoding="utf-8",
    )
    examples = load_text_to_sql_examples("bird", dev_path)
    assert examples[0].db_id == "toy"
    assert examples[0].gold_sql == "select count(*) from items"
    assert examples[0].evidence == "items means rows"


def test_resolve_bird_paths(tmp_path: Path) -> None:
    root = tmp_path / "bird"
    db_dir = root / "dev_databases" / "toy"
    db_dir.mkdir(parents=True)
    db_path = db_dir / "toy.sqlite"
    sqlite3.connect(db_path).close()
    dev_path = root / "dev.json"
    dev_path.write_text("[]", encoding="utf-8")

    assert resolve_dataset_root("bird", root) == root.resolve()
    assert resolve_split_file("bird", root, "dev") == dev_path.resolve()
    assert resolve_db_path("bird", root, "toy") == db_path


def test_resolve_bird_dated_dev_package_paths(tmp_path: Path) -> None:
    root = tmp_path / "bird"
    dated_root = root / "dev_20240627"
    db_dir = dated_root / "dev_databases" / "toy"
    db_dir.mkdir(parents=True)
    db_path = db_dir / "toy.sqlite"
    sqlite3.connect(db_path).close()
    dev_path = dated_root / "dev.json"
    dev_path.write_text("[]", encoding="utf-8")

    assert resolve_dataset_root("bird", root) == dated_root.resolve()
    assert resolve_split_file("bird", root, "dev") == dev_path.resolve()
    assert resolve_db_path("bird", root, "toy") == db_path
