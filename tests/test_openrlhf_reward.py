from __future__ import annotations

import json
import sqlite3
from pathlib import Path

import pytest

from openrlhf_reward import score_response


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


def test_score_response_rewards_exec_match(tmp_path: Path) -> None:
    db_path = tmp_path / "toy.sqlite"
    _build_db(db_path)
    label = {"db_path": str(db_path), "gold_sql": "SELECT count(*) FROM items"}
    response = (
        "<thought>Read schema and answer.</thought>\n"
        "<action>SELECT count(*) FROM items</action>"
    )
    result = score_response(response, label)
    assert result["format_match"] is True
    assert result["no_error"] is True
    assert result["exec_match"] is True
    assert result["reward"] == 1.3


def test_score_response_rewards_no_error_without_exec_match(tmp_path: Path) -> None:
    db_path = tmp_path / "toy.sqlite"
    _build_db(db_path)
    label = {"db_path": str(db_path), "gold_sql": "SELECT count(*) FROM items"}
    response = (
        "<thought>Read schema and answer.</thought>\n"
        "<action>SELECT name FROM items</action>"
    )
    result = score_response(response, label)
    assert result["format_match"] is True
    assert result["no_error"] is True
    assert result["exec_match"] is False
    assert result["reward"] == pytest.approx(0.3)


def test_score_response_format_failure_gets_zero() -> None:
    result = score_response("SELECT 1", {"db_path": "missing.sqlite", "gold_sql": "SELECT 1"})
    assert result["reward"] == 0.0
    assert result["format_match"] is False


def test_score_response_accepts_json_label(tmp_path: Path) -> None:
    db_path = tmp_path / "toy.sqlite"
    _build_db(db_path)
    label = json.dumps({"db_path": str(db_path), "gold_sql": "SELECT name FROM items"})
    response = (
        "<thought>Read schema and answer.</thought>\n"
        "<action>SELECT name FROM items</action>"
    )
    assert score_response(response, label)["exec_match"] is True


def test_score_response_resolves_relative_db_path(
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
    response = (
        "<thought>Read schema and answer.</thought>\n"
        "<action>SELECT count(*) FROM items</action>"
    )
    assert score_response(response, label)["exec_match"] is True


def test_score_response_caches_gold_sql_execution(tmp_path: Path, monkeypatch) -> None:
    import openrlhf_reward

    db_path = tmp_path / "toy.sqlite"
    _build_db(db_path)
    label = {"db_path": str(db_path), "gold_sql": "SELECT count(*) FROM items"}
    response = (
        "<thought>Read schema and answer.</thought>\n"
        "<action>SELECT count(*) FROM items</action>"
    )
    openrlhf_reward._execute_gold_sql_cached.cache_clear()
    calls = 0
    original_execute_sql = openrlhf_reward.execute_sql

    def counting_execute_sql(*args, **kwargs):  # type: ignore[no-untyped-def]
        nonlocal calls
        calls += 1
        return original_execute_sql(*args, **kwargs)

    monkeypatch.setattr(openrlhf_reward, "execute_sql", counting_execute_sql)
    score_response(response, label)
    score_response(response, label)
    assert calls == 3
