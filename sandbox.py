from __future__ import annotations

import argparse
import base64
import json
import re
import sqlite3
import subprocess
import sys
import time
from pathlib import Path
from typing import Any

READ_ONLY_PREFIXES = (
    "select",
    "with",
    "explain select",
    "explain query plan",
)

DENY_TOKENS = {
    "alter",
    "attach",
    "create",
    "delete",
    "detach",
    "drop",
    "insert",
    "load_extension",
    "pragma",
    "reindex",
    "replace",
    "update",
    "vacuum",
}


def _decode_sql(encoded: str) -> str:
    return base64.b64decode(encoded.encode("utf-8")).decode("utf-8")


def _normalize_sql(sql: str) -> str:
    return " ".join(sql.strip().split())


def _sql_tokens(sql: str) -> list[str]:
    return re.findall(r"[a-z_]+", sql.lower())


def _is_read_only_sql(sql: str) -> bool:
    lowered = _normalize_sql(sql).lower().rstrip(";")
    if not lowered:
        return False
    if any(token in DENY_TOKENS for token in _sql_tokens(lowered)):
        return False
    return lowered.startswith(READ_ONLY_PREFIXES)


def _authorizer(
    action_code: int,
    param1: str | None,
    param2: str | None,
    db_name: str | None,
    trigger_name: str | None,
) -> int:
    del param1, param2, db_name, trigger_name
    forbidden = {
        sqlite3.SQLITE_ATTACH,
        sqlite3.SQLITE_DETACH,
        sqlite3.SQLITE_ALTER_TABLE,
        sqlite3.SQLITE_CREATE_INDEX,
        sqlite3.SQLITE_CREATE_TABLE,
        sqlite3.SQLITE_CREATE_TEMP_INDEX,
        sqlite3.SQLITE_CREATE_TEMP_TABLE,
        sqlite3.SQLITE_CREATE_TEMP_TRIGGER,
        sqlite3.SQLITE_CREATE_TEMP_VIEW,
        sqlite3.SQLITE_CREATE_TRIGGER,
        sqlite3.SQLITE_CREATE_VIEW,
        sqlite3.SQLITE_CREATE_VTABLE,
        sqlite3.SQLITE_DELETE,
        sqlite3.SQLITE_DROP_INDEX,
        sqlite3.SQLITE_DROP_TABLE,
        sqlite3.SQLITE_DROP_TEMP_INDEX,
        sqlite3.SQLITE_DROP_TEMP_TABLE,
        sqlite3.SQLITE_DROP_TEMP_TRIGGER,
        sqlite3.SQLITE_DROP_TEMP_VIEW,
        sqlite3.SQLITE_DROP_TRIGGER,
        sqlite3.SQLITE_DROP_VIEW,
        sqlite3.SQLITE_DROP_VTABLE,
        sqlite3.SQLITE_INSERT,
        sqlite3.SQLITE_PRAGMA,
        sqlite3.SQLITE_TRANSACTION,
        sqlite3.SQLITE_UPDATE,
    }
    return sqlite3.SQLITE_DENY if action_code in forbidden else sqlite3.SQLITE_OK


def _json_default(value: Any) -> Any:
    if isinstance(value, bytes):
        return value.decode("utf-8", errors="replace")
    raise TypeError(f"Unsupported value: {type(value)!r}")


def _worker_main(db_path: Path, sql: str) -> int:
    started = time.perf_counter()
    result: dict[str, Any]

    try:
        if not db_path.exists():
            raise FileNotFoundError(f"Database not found: {db_path}")
        if not _is_read_only_sql(sql):
            raise PermissionError("Only read-only SELECT/WITH queries are allowed")

        conn = sqlite3.connect(f"file:{db_path}?mode=ro", uri=True)
        conn.set_authorizer(_authorizer)
        conn.row_factory = None
        conn.text_factory = bytes
        try:
            cursor = conn.execute(sql)
            rows = cursor.fetchall()
            result = {
                "ok": True,
                "rows": rows,
                "columns": [desc[0] for desc in (cursor.description or [])],
                "row_count": len(rows),
                "error": "",
                "duration_sec": time.perf_counter() - started,
            }
        finally:
            conn.close()
    except Exception as exc:
        result = {
            "ok": False,
            "rows": [],
            "columns": [],
            "row_count": 0,
            "error": f"{type(exc).__name__}: {exc}",
            "duration_sec": time.perf_counter() - started,
        }

    print(json.dumps(result, ensure_ascii=False, default=_json_default))
    return 0


def execute_sql(
    db_path: str | Path,
    sql: str,
    timeout_s: float = 3.0,
    python_executable: str = sys.executable,
) -> dict[str, Any]:
    db_path = Path(db_path).resolve()
    encoded_sql = base64.b64encode(sql.encode("utf-8")).decode("utf-8")
    cmd = [
        python_executable,
        __file__,
        "--worker",
        "--db",
        str(db_path),
        "--sql-b64",
        encoded_sql,
    ]
    started = time.perf_counter()

    try:
        completed = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            timeout=timeout_s,
            check=False,
        )
    except subprocess.TimeoutExpired:
        return {
            "ok": False,
            "rows": [],
            "columns": [],
            "row_count": 0,
            "error": f"TimeoutExpired: query exceeded {timeout_s:.1f}s",
            "duration_sec": time.perf_counter() - started,
        }

    stdout = completed.stdout.strip()
    if completed.returncode != 0:
        stderr = completed.stderr.strip()
        return {
            "ok": False,
            "rows": [],
            "columns": [],
            "row_count": 0,
            "error": stderr or f"Worker exited with code {completed.returncode}",
            "duration_sec": time.perf_counter() - started,
        }

    payload = json.loads(stdout) if stdout else {}
    payload.setdefault("duration_sec", time.perf_counter() - started)
    return payload


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Read-only SQLite sandbox for Text-to-SQL execution."
    )
    parser.add_argument("--db", required=True, help="Path to SQLite database file.")
    parser.add_argument("--sql", help="SQL query to execute.")
    parser.add_argument(
        "--sql-b64",
        help="Base64-encoded SQL query. Used by parent subprocess wrapper.",
    )
    parser.add_argument(
        "--timeout",
        type=float,
        default=3.0,
        help="Parent-side timeout in seconds.",
    )
    parser.add_argument("--worker", action="store_true", help="Run in worker mode.")
    args = parser.parse_args()

    sql = args.sql if args.sql is not None else _decode_sql(args.sql_b64 or "")
    if args.worker:
        return _worker_main(Path(args.db), sql)

    result = execute_sql(args.db, sql, timeout_s=args.timeout)
    print(json.dumps(result, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
