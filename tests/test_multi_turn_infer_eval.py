from __future__ import annotations

import sqlite3
from pathlib import Path

import pytest

from dataset_adapter import TextToSQLExample
from multi_turn_infer_eval import run_multi_turn_eval


class FakeGenerator:
    def __init__(self, batches: list[list[str]]) -> None:
        self.batches = batches

    def generate(self, prompts: list[str]) -> list[str]:
        del prompts
        if not self.batches:
            raise RuntimeError("No fake response batch left")
        return self.batches.pop(0)


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


def _tagged(sql: str) -> str:
    return f"<thought>Use the table.</thought>\n<action>{sql}</action>"


def test_multi_turn_eval_counts_wrong_result_rescue(tmp_path: Path) -> None:
    root = tmp_path / "spider"
    db_dir = root / "database" / "toy"
    db_dir.mkdir(parents=True)
    _build_db(db_dir / "toy.sqlite")
    examples = [
        TextToSQLExample(
            index=0,
            db_id="toy",
            question="How many rows?",
            gold_sql="SELECT count(*) FROM items",
        ),
        TextToSQLExample(
            index=1,
            db_id="toy",
            question="List names.",
            gold_sql="SELECT name FROM items",
        ),
    ]
    generator = FakeGenerator(
        [
            [_tagged("SELECT 1"), _tagged("SELECT name FROM items")],
            [_tagged("SELECT count(*) FROM items")],
        ]
    )

    summary, trajectories = run_multi_turn_eval(
        dataset="spider",
        root=root,
        split="dev",
        examples=examples,
        prompts=["prompt 0", "prompt 1"],
        generator=generator,  # type: ignore[arg-type]
        max_turns=2,
        timeout_s=3.0,
    )

    assert summary["total"] == 2
    assert summary["turn1_matched"] == 1
    assert summary["final_matched"] == 2
    assert summary["rescued_by_turn2"] == 1
    assert summary["wrong_result_rescue"] == 1
    assert summary["accuracy_excluding_first_turn_timeout"] == 1.0
    assert summary["turn1_accuracy_excluding_timeout"] == 0.5
    assert summary["final_accuracy_excluding_first_turn_timeout"] == 1.0
    assert summary["final_accuracy_excluding_first_turn_timeouts"] == 1.0
    assert summary["turn2_rescue_rate_excluding_first_turn_timeout"] == 0.5
    assert summary["turn2_rescue_by_first_turn_error"] == {"wrong_result": 1}
    assert trajectories[0]["rescued_by_turn2"] is True
    assert trajectories[0]["turns"][0]["error_category"] == "wrong_result"
    assert trajectories[0]["turns"][1]["exec_match"] is True
