from __future__ import annotations

import sqlite3
from pathlib import Path

from agent_rollout import rollout_self_correction


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


def test_rollout_self_correction_uses_second_turn_after_sql_error(tmp_path: Path) -> None:
    db_path = tmp_path / "toy.sqlite"
    _build_db(db_path)
    label = {"db_path": str(db_path), "gold_sql": "SELECT count(*) FROM items"}
    responses = iter(
        [
            "<thought>Try the query.</thought>\n<action>SELECT count(*) FROM missing</action>",
            "<thought>Use the existing table.</thought>\n<action>SELECT count(*) FROM items</action>",
        ]
    )
    prompts_seen: list[str] = []

    def model_fn(prompt: str) -> str:
        prompts_seen.append(prompt)
        return next(responses)

    result = rollout_self_correction("Question prompt", label, model_fn, max_turns=2)
    assert result["best_reward"] == 1.3
    assert result["best_turn_index"] == 1
    assert len(result["turns"]) == 2
    assert "Execution feedback" in prompts_seen[1]
    assert "no such table" in prompts_seen[1]


def test_rollout_self_correction_stops_after_first_correct_turn(tmp_path: Path) -> None:
    db_path = tmp_path / "toy.sqlite"
    _build_db(db_path)
    label = {"db_path": str(db_path), "gold_sql": "SELECT count(*) FROM items"}

    def model_fn(_: str) -> str:
        return "<thought>Use count.</thought>\n<action>SELECT count(*) FROM items</action>"

    result = rollout_self_correction("Question prompt", label, model_fn, max_turns=2)
    assert result["best_turn_index"] == 0
    assert len(result["turns"]) == 1


def test_rollout_self_correction_resolves_relative_db_path(
    tmp_path: Path, monkeypatch
) -> None:
    spider_root = tmp_path / "spider_data"
    db_root = spider_root / "database" / "toy_db"
    db_root.mkdir(parents=True)
    db_path = db_root / "toy_db.sqlite"
    _build_db(db_path)
    monkeypatch.setenv("SPIDER_ROOT", str(spider_root))
    label = {
        "db_path": "database/toy_db/toy_db.sqlite",
        "gold_sql": "SELECT count(*) FROM items",
    }

    responses = iter(
        [
            "<thought>Try the query.</thought>\n<action>SELECT count(*) FROM missing</action>",
            "<thought>Use the existing table.</thought>\n<action>SELECT count(*) FROM items</action>",
        ]
    )

    def model_fn(_: str) -> str:
        return next(responses)

    result = rollout_self_correction("Question prompt", label, model_fn, max_turns=2)
    assert result["best_turn_index"] == 1
    assert result["turns"][1]["exec_match"] is True
