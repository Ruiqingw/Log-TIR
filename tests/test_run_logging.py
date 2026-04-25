from __future__ import annotations

import json
from pathlib import Path

from run_logging import JsonlWriter, create_run_dir


def test_create_run_dir_writes_config_and_latest(tmp_path: Path) -> None:
    run_dir = create_run_dir(
        repo_root=tmp_path,
        kind="test",
        logs_dir=Path("logs"),
        extra={"lr": "1e-5"},
        command=["train"],
    )
    assert run_dir.exists()
    assert (run_dir / "config.json").exists()
    assert (run_dir / "metrics.jsonl").exists()
    assert (tmp_path / "logs" / "latest").is_symlink()
    config = json.loads((run_dir / "config.json").read_text(encoding="utf-8"))
    assert config["kind"] == "test"
    assert config["extra"]["lr"] == "1e-5"


def test_jsonl_writer_appends_records(tmp_path: Path) -> None:
    path = tmp_path / "metrics.jsonl"
    writer = JsonlWriter(path)
    writer.write({"step": 1, "reward": 0.3})
    writer.write({"step": 2, "reward": 1.3})
    lines = path.read_text(encoding="utf-8").splitlines()
    assert len(lines) == 2
    assert json.loads(lines[1])["reward"] == 1.3

