from __future__ import annotations

import json
import sqlite3
from pathlib import Path

import eval as eval_module
from eval import _normalize_cell, evaluate_dataset, execution_match


def _build_db(db_path: Path) -> None:
    conn = sqlite3.connect(db_path)
    conn.executescript(
        """
        create table numbers (value real);
        insert into numbers(value) values (1.0), (2.0);

        create table letters (ch text);
        insert into letters(ch) values ('b'), ('a');
        """
    )
    conn.commit()
    conn.close()


def _build_spider_fixture(root: Path) -> tuple[Path, Path]:
    spider_root = root / "spider_data"
    db_root = spider_root / "database" / "toy_db"
    db_root.mkdir(parents=True)
    db_path = db_root / "toy_db.sqlite"
    _build_db(db_path)

    dev_path = spider_root / "dev.json"
    dev_examples = [
        {
            "db_id": "toy_db",
            "question": "Return letters in ascending order.",
            "query": "select ch from letters order by ch asc",
        }
    ]
    dev_path.write_text(json.dumps(dev_examples), encoding="utf-8")
    return spider_root, dev_path


def test_normalize_cell_decodes_bytes() -> None:
    assert _normalize_cell(b"abc") == ("str", "abc")


def test_execution_match_preserves_order(tmp_path: Path) -> None:
    db_path = tmp_path / "toy.sqlite"
    _build_db(db_path)
    outcome = execution_match(
        pred_sql="select ch from letters order by ch desc",
        gold_sql="select ch from letters order by ch asc",
        db_path=db_path,
    )
    assert outcome["match"] is False


def test_execution_match_normalizes_numeric_types(tmp_path: Path) -> None:
    db_path = tmp_path / "toy.sqlite"
    _build_db(db_path)
    outcome = execution_match(
        pred_sql="select 1.0",
        gold_sql="select 1",
        db_path=db_path,
    )
    assert outcome["match"] is True


def test_evaluate_dataset_with_gold_predictions(tmp_path: Path) -> None:
    spider_root, dev_path = _build_spider_fixture(tmp_path)
    report = evaluate_dataset(
        dev_path=dev_path,
        spider_root=spider_root,
        pred_path=None,
        use_gold_predictions=True,
        timeout_s=3.0,
        limit=None,
    )
    assert report["total"] == 1
    assert report["matched"] == 1
    assert report["accuracy"] == 1.0


def test_evaluate_dataset_gold_predictions_execute_gold_once(
    tmp_path: Path, monkeypatch
) -> None:
    spider_root, dev_path = _build_spider_fixture(tmp_path)
    calls: list[str] = []

    def fake_execute_sql(db_path: Path, sql: str, timeout_s: float):
        calls.append(sql)
        return {"ok": True, "rows": []}

    monkeypatch.setattr(eval_module, "execute_sql", fake_execute_sql)
    report = evaluate_dataset(
        dev_path=dev_path,
        spider_root=spider_root,
        pred_path=None,
        use_gold_predictions=True,
        timeout_s=3.0,
        limit=None,
    )
    assert report["matched"] == 1
    assert calls == ["select ch from letters order by ch asc"]


def test_evaluate_dataset_with_bird_fixture(tmp_path: Path) -> None:
    bird_root = tmp_path / "bird"
    db_root = bird_root / "dev_databases" / "toy_db"
    db_root.mkdir(parents=True)
    db_path = db_root / "toy_db.sqlite"
    _build_db(db_path)
    dev_path = bird_root / "dev.json"
    dev_path.write_text(
        json.dumps(
            [
                {
                    "db_id": "toy_db",
                    "question": "How many numbers are present?",
                    "SQL": "select count(*) from numbers",
                    "evidence": "numbers means rows in the numbers table",
                }
            ]
        ),
        encoding="utf-8",
    )

    report = evaluate_dataset(
        dev_path=dev_path,
        spider_root=bird_root,
        pred_path=None,
        use_gold_predictions=True,
        timeout_s=3.0,
        limit=None,
        dataset="bird",
        workers=2,
    )
    assert report["total"] == 1
    assert report["matched"] == 1
    assert report["accuracy"] == 1.0
