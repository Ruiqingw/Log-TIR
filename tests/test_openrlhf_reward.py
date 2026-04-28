from __future__ import annotations

import json
import sqlite3
from pathlib import Path

import pytest

from openrlhf_reward import reward_func, score_response


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
    assert result["error_category"] == ""


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
    assert result["wrong_result"] is True
    assert result["error_category"] == "wrong_result"


def test_score_response_format_failure_gets_zero() -> None:
    result = score_response("SELECT 1", {"db_path": "missing.sqlite", "gold_sql": "SELECT 1"})
    assert result["reward"] == 0.0
    assert result["format_match"] is False
    assert result["error_category"] == "format"


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


def test_score_response_resolves_relative_db_path_from_repo_root(
    tmp_path: Path, monkeypatch
) -> None:
    import openrlhf_reward

    repo_root = tmp_path / "repo"
    spider_root = repo_root / "data" / "spider" / "spider_data"
    db_root = spider_root / "database" / "toy_db"
    db_root.mkdir(parents=True)
    db_path = db_root / "toy_db.sqlite"
    _build_db(db_path)
    cwd = tmp_path / "ray_worker"
    cwd.mkdir()
    monkeypatch.chdir(cwd)
    monkeypatch.delenv("SPIDER_ROOT", raising=False)
    monkeypatch.setattr(openrlhf_reward, "REPO_ROOT", repo_root)
    label = {
        "db_id": "toy_db",
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


def test_reward_func_returns_openrlhf_reward_payload(tmp_path: Path) -> None:
    db_path = tmp_path / "toy.sqlite"
    _build_db(db_path)
    prompt = "Question prompt"
    response = (
        "<thought>Read schema and answer.</thought>\n"
        "<action>SELECT count(*) FROM items</action>"
    )
    label = json.dumps({"db_path": str(db_path), "gold_sql": "SELECT count(*) FROM items"})

    result = reward_func([prompt + response], [prompt], [label])

    assert result["rewards"] == pytest.approx(1.3)
    assert result["scores"] == pytest.approx(1.3)
    assert result["extra_logs"]["format_match_rate"] == 1.0
    assert result["extra_logs"]["exec_match_rate"] == 1.0
    assert result["extra_logs"]["avg_action_sql_length"] > 0


def test_reward_func_recovers_response_when_decoded_prompt_differs(tmp_path: Path) -> None:
    db_path = tmp_path / "toy.sqlite"
    _build_db(db_path)
    prompt = (
        "Return exactly two tags:\n"
        "<thought>brief reasoning</thought>\n"
        "<action>one SQLite query</action>\n"
        "Question prompt"
    )
    decoded_query_prefix = " " + prompt
    response = (
        "<thought>Read schema and answer.</thought>\n"
        "<action>SELECT count(*) FROM items</action>"
    )
    label = json.dumps({"db_path": str(db_path), "gold_sql": "SELECT count(*) FROM items"})

    result = reward_func([decoded_query_prefix + response], [prompt], [label])

    assert result["rewards"] == pytest.approx(1.3)
    assert result["extra_logs"]["format_match_rate"] == 1.0
    assert result["extra_logs"]["exec_match_rate"] == 1.0


def test_reward_func_recovers_response_after_prompt_with_leading_noise(
    tmp_path: Path,
) -> None:
    db_path = tmp_path / "toy.sqlite"
    _build_db(db_path)
    prompt = "Question prompt\n"
    response = (
        "1\n"
        "<thought>Read schema and answer.</thought>\n"
        "<action>SELECT count(*) FROM items</action>"
    )
    label = json.dumps({"db_path": str(db_path), "gold_sql": "SELECT count(*) FROM items"})

    result = reward_func([prompt + response], [prompt], [label])

    assert result["rewards"] == pytest.approx(1.3)
    assert result["extra_logs"]["format_match_rate"] == 1.0
    assert result["extra_logs"]["exec_match_rate"] == 1.0
