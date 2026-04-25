from __future__ import annotations

import sqlite3
import subprocess
from pathlib import Path

import pytest

from sandbox import execute_sql


def _build_db(db_path: Path) -> None:
    conn = sqlite3.connect(db_path)
    conn.executescript(
        """
        create table items (
            id integer primary key,
            name text,
            created text,
            score real
        );
        insert into items(name, created, score) values
            ('alpha', '2024-01-01 00:00:00', 1.0),
            ('beta', '2024-01-02 00:00:00', 2.5);
        """
    )
    conn.commit()
    conn.close()


@pytest.fixture
def sample_db(tmp_path: Path) -> Path:
    db_path = tmp_path / "sample.sqlite"
    _build_db(db_path)
    return db_path


def test_execute_sql_allows_read_only_query(sample_db: Path) -> None:
    result = execute_sql(sample_db, "select name from items order by id")
    assert result["ok"] is True
    assert result["rows"] == [["alpha"], ["beta"]]


def test_execute_sql_rejects_drop(sample_db: Path) -> None:
    result = execute_sql(sample_db, "drop table items")
    assert result["ok"] is False
    assert "Only read-only" in result["error"]


def test_execute_sql_returns_syntax_error(sample_db: Path) -> None:
    result = execute_sql(sample_db, "select from items")
    assert result["ok"] is False
    assert "OperationalError" in result["error"]


def test_execute_sql_does_not_false_positive_on_created_column(sample_db: Path) -> None:
    result = execute_sql(sample_db, "select max(created) from items")
    assert result["ok"] is True
    assert result["rows"] == [["2024-01-02 00:00:00"]]


def test_execute_sql_timeout_returns_structured_error(
    sample_db: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    def fake_run(*args, **kwargs):  # type: ignore[no-untyped-def]
        raise subprocess.TimeoutExpired(cmd=kwargs.get("args", args[0]), timeout=0.01)

    monkeypatch.setattr(subprocess, "run", fake_run)
    result = execute_sql(sample_db, "select 1", timeout_s=0.01)
    assert result["ok"] is False
    assert "TimeoutExpired" in result["error"]
