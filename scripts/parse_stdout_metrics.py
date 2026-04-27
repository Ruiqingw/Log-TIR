from __future__ import annotations

import ast
import json
import re
import sys
from collections.abc import Iterable, Iterator
from typing import Any


GLOBAL_STEP_RE = re.compile(r"Global step (\d+): (\{.*\})$")


def parse_metrics_line(line: str) -> dict[str, Any] | None:
    match = GLOBAL_STEP_RE.search(line.rstrip())
    if not match:
        return None

    step, payload = match.groups()
    metrics = ast.literal_eval(payload)
    if not isinstance(metrics, dict):
        return None
    metrics["step"] = int(step)
    return metrics


def iter_metrics(lines: Iterable[str]) -> Iterator[dict[str, Any]]:
    for line in lines:
        metrics = parse_metrics_line(line)
        if metrics is not None:
            yield metrics


def main() -> int:
    for metrics in iter_metrics(sys.stdin):
        print(json.dumps(metrics, sort_keys=True), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
