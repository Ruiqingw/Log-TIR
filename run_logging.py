from __future__ import annotations

import argparse
import json
import os
import platform
import socket
import subprocess
import sys
from datetime import datetime
from pathlib import Path
from typing import Any


class JsonlWriter:
    def __init__(self, path: Path) -> None:
        self.path = path
        self.path.parent.mkdir(parents=True, exist_ok=True)

    def write(self, record: dict[str, Any]) -> None:
        with self.path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(record, ensure_ascii=False) + "\n")


def _run_text(cmd: list[str], cwd: Path) -> str:
    try:
        return subprocess.check_output(
            cmd,
            cwd=cwd,
            stderr=subprocess.DEVNULL,
            text=True,
        ).strip()
    except Exception:
        return ""


def _torch_info() -> dict[str, Any]:
    try:
        import torch
    except Exception as exc:
        return {"available": False, "error": f"{type(exc).__name__}: {exc}"}

    info: dict[str, Any] = {
        "available": True,
        "version": getattr(torch, "__version__", ""),
        "cuda_available": torch.cuda.is_available(),
        "cuda_version": getattr(torch.version, "cuda", None),
        "gpu_count": torch.cuda.device_count() if torch.cuda.is_available() else 0,
        "gpus": [],
    }
    if torch.cuda.is_available():
        for idx in range(torch.cuda.device_count()):
            info["gpus"].append(torch.cuda.get_device_name(idx))
    return info


def collect_config(
    repo_root: Path,
    kind: str,
    command: list[str],
    extra: dict[str, Any] | None = None,
) -> dict[str, Any]:
    status = _run_text(["git", "status", "--short"], repo_root)
    return {
        "kind": kind,
        "created_at": datetime.now().isoformat(timespec="seconds"),
        "repo_root": str(repo_root),
        "command": command,
        "hostname": socket.gethostname(),
        "platform": platform.platform(),
        "python": {
            "executable": sys.executable,
            "version": sys.version,
        },
        "torch": _torch_info(),
        "git": {
            "commit": _run_text(["git", "rev-parse", "HEAD"], repo_root),
            "branch": _run_text(["git", "branch", "--show-current"], repo_root),
            "dirty": bool(status),
            "status_short": status,
            "diff": _run_text(["git", "diff"], repo_root),
            "staged_diff": _run_text(["git", "diff", "--cached"], repo_root),
        },
        "env": {
            key: value
            for key, value in os.environ.items()
            if key.startswith(("LOGTIR_", "SFT_", "GRPO_", "MODEL_", "WANDB_", "SPIDER_"))
        },
        "extra": extra or {},
    }


def create_run_dir(
    repo_root: Path,
    kind: str,
    logs_dir: Path = Path("logs"),
    extra: dict[str, Any] | None = None,
    command: list[str] | None = None,
) -> Path:
    logs_dir = logs_dir if logs_dir.is_absolute() else repo_root / logs_dir
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_dir = logs_dir / f"run_{timestamp}_{kind}"
    suffix = 1
    while run_dir.exists():
        suffix += 1
        run_dir = logs_dir / f"run_{timestamp}_{kind}_{suffix}"
    run_dir.mkdir(parents=True)

    config = collect_config(
        repo_root=repo_root,
        kind=kind,
        command=command or sys.argv,
        extra=extra,
    )
    (run_dir / "config.json").write_text(
        json.dumps(config, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    (run_dir / "metrics.jsonl").touch()
    (run_dir / "trajectories.jsonl").touch()
    (run_dir / "train.stdout.log").touch()
    (run_dir / "train.stderr.log").touch()

    latest = logs_dir / "latest"
    if latest.exists() or latest.is_symlink():
        latest.unlink()
    latest.symlink_to(run_dir.name)
    return run_dir


def init_wandb_if_enabled(
    run_dir: Path,
    config: dict[str, Any],
    project: str = "log-tir",
) -> Any:
    if os.environ.get("WANDB_DISABLED", "").lower() in {"1", "true", "yes"}:
        return None
    if os.environ.get("LOGTIR_WANDB", "0") != "1":
        return None
    try:
        import wandb
    except Exception as exc:
        print(f"Warning: wandb import failed: {type(exc).__name__}: {exc}", file=sys.stderr)
        return None
    return wandb.init(project=project, name=run_dir.name, config=config)


def main() -> int:
    parser = argparse.ArgumentParser(description="Create Log-TIR reproducible run logs.")
    parser.add_argument("command", choices=["prepare"])
    parser.add_argument("--kind", required=True)
    parser.add_argument("--repo-root", type=Path, default=Path("."))
    parser.add_argument("--logs-dir", type=Path, default=Path("logs"))
    parser.add_argument("--extra-json", default="{}")
    args, remainder = parser.parse_known_args()

    extra = json.loads(args.extra_json)
    run_dir = create_run_dir(
        repo_root=args.repo_root.resolve(),
        kind=args.kind,
        logs_dir=args.logs_dir,
        extra=extra,
        command=remainder or sys.argv,
    )
    print(run_dir)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
