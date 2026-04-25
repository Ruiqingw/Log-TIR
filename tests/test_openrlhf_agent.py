from __future__ import annotations

import asyncio
import sqlite3
from pathlib import Path

import pytest

from openrlhf_agent import AgentInstance


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


def _value(value) -> float:  # type: ignore[no-untyped-def]
    return float(value.item()) if hasattr(value, "item") else float(value)


def test_agent_returns_feedback_after_first_sql_error(tmp_path: Path, monkeypatch) -> None:
    monkeypatch.setenv("LOGTIR_AGENT_MAX_TURNS", "2")
    db_path = tmp_path / "toy.sqlite"
    _build_db(db_path)
    label = {"db_path": str(db_path), "gold_sql": "SELECT count(*) FROM items"}
    agent = AgentInstance()

    async def run() -> dict:
        await agent.reset({"observation": "Question prompt"})
        return await agent.step(
            {
                "label": label,
                "action_text": (
                    "<thought>Try missing table.</thought>\n"
                    "<action>SELECT count(*) FROM missing</action>"
                ),
            }
        )

    result = asyncio.run(run())
    assert result["done"] is False
    assert _value(result["rewards"]) == 0.0
    assert "Execution feedback" in result["environment_feedback"]
    assert "no such table" in result["environment_feedback"]


def test_agent_terminates_with_reward_on_second_correct_turn(
    tmp_path: Path, monkeypatch
) -> None:
    monkeypatch.setenv("LOGTIR_AGENT_MAX_TURNS", "2")
    db_path = tmp_path / "toy.sqlite"
    _build_db(db_path)
    label = {"db_path": str(db_path), "gold_sql": "SELECT count(*) FROM items"}
    agent = AgentInstance()

    async def run() -> tuple[dict, dict]:
        await agent.reset({"observation": "Question prompt"})
        first = await agent.step(
            {
                "label": label,
                "action_text": (
                    "<thought>Try missing table.</thought>\n"
                    "<action>SELECT count(*) FROM missing</action>"
                ),
            }
        )
        second = await agent.step(
            {
                "label": label,
                "action_text": (
                    "<thought>Use existing table.</thought>\n"
                    "<action>SELECT count(*) FROM items</action>"
                ),
            }
        )
        return first, second

    first, second = asyncio.run(run())
    assert first["done"] is False
    assert second["done"] is True
    assert _value(second["rewards"]) == pytest.approx(1.3)
    assert second["environment_feedback"] == ""


def test_agent_terminal_wrong_second_turn_gets_shaped_reward(
    tmp_path: Path, monkeypatch
) -> None:
    monkeypatch.setenv("LOGTIR_AGENT_MAX_TURNS", "2")
    db_path = tmp_path / "toy.sqlite"
    _build_db(db_path)
    label = {"db_path": str(db_path), "gold_sql": "SELECT count(*) FROM items"}
    agent = AgentInstance()

    async def run() -> dict:
        await agent.reset({"observation": "Question prompt"})
        await agent.step(
            {
                "label": label,
                "action_text": (
                    "<thought>Use names.</thought>\n"
                    "<action>SELECT name FROM items</action>"
                ),
            }
        )
        return await agent.step(
            {
                "label": label,
                "action_text": (
                    "<thought>Still use names.</thought>\n"
                    "<action>SELECT name FROM items</action>"
                ),
            }
        )

    result = asyncio.run(run())
    assert result["done"] is True
    assert _value(result["rewards"]) == pytest.approx(0.3)
