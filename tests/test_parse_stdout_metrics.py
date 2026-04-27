from __future__ import annotations

import importlib.util
from pathlib import Path


MODULE_PATH = Path(__file__).resolve().parents[1] / "scripts" / "parse_stdout_metrics.py"
SPEC = importlib.util.spec_from_file_location("parse_stdout_metrics", MODULE_PATH)
assert SPEC is not None
parse_stdout_metrics = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
SPEC.loader.exec_module(parse_stdout_metrics)

iter_metrics = parse_stdout_metrics.iter_metrics
parse_metrics_line = parse_stdout_metrics.parse_metrics_line


def test_parse_global_step_metrics_line() -> None:
    metrics = parse_metrics_line(
        "Global step 100: {'exec_match_rate': 1.0, 'group_reward_std': 0.25, 'policy_loss': -0.01}"
    )
    assert metrics == {
        "step": 100,
        "exec_match_rate": 1.0,
        "group_reward_std": 0.25,
        "policy_loss": -0.01,
    }


def test_iter_metrics_skips_non_metric_lines() -> None:
    rows = list(
        iter_metrics(
            [
                "noise\n",
                "Global step 2: {'avg_turns_used': 1.5}\n",
            ]
        )
    )
    assert rows == [{"step": 2, "avg_turns_used": 1.5}]
