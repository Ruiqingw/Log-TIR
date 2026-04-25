from __future__ import annotations

import json
from pathlib import Path

from rl_data import build_grpo_prompts
from tests.test_sft_data import _write_spider_fixture


def test_build_grpo_prompts_writes_prompt_and_label_jsonl(tmp_path: Path) -> None:
    spider_root = _write_spider_fixture(tmp_path)
    output_path = tmp_path / "rl" / "prompts.jsonl"
    report = build_grpo_prompts(
        spider_root=spider_root,
        output_path=output_path,
        limit=1,
        seed=7,
    )
    assert report["count"] == 1
    record = json.loads(output_path.read_text(encoding="utf-8").splitlines()[0])
    assert record["task"] == "text-to-sql-rl"
    assert record["datasource"] == "spider"
    assert record["prompt"].startswith("You are a SQLite Text-to-SQL agent.")
    label = json.loads(record["label"])
    assert label["db_id"] == "toy_db"
    assert label["gold_sql"]
    assert label["db_path"].endswith("toy_db.sqlite")
