from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any


@dataclass(frozen=True)
class TextToSQLExample:
    index: int
    db_id: str
    question: str
    gold_sql: str
    evidence: str = ""


def _read_json_list(path: Path) -> list[dict[str, Any]]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, list):
        raise TypeError(f"{path} must contain a JSON list")
    return payload


def resolve_dataset_root(dataset: str, root: Path) -> Path:
    root = root.resolve()
    if dataset == "spider":
        candidates = (root, root / "spider", root / "spider_data")
        markers = ("database", "tables.json")
    elif dataset == "bird":
        candidates = (
            root,
            root / "bird",
            root / "dev",
            root / "train",
            *sorted(root.glob("dev_*")),
            *sorted(root.glob("train_*")),
        )
        markers = ("dev_databases", "train_databases", "database", "databases")
    else:
        raise ValueError(f"Unsupported dataset: {dataset}")

    for candidate in candidates:
        if any((candidate / marker).exists() for marker in markers):
            return candidate
    return root


def resolve_split_file(
    dataset: str,
    root: Path,
    split: str = "dev",
    explicit_path: Path | None = None,
) -> Path:
    if explicit_path is not None:
        if explicit_path.exists():
            return explicit_path.resolve()
        raise FileNotFoundError(f"{dataset} split file not found: {explicit_path}")

    filenames = {
        "spider": (f"{split}.json",),
        "bird": (
            f"{split}.json",
            f"{split}_with_evidence.json",
            "dev.json" if split == "dev" else f"{split}.json",
        ),
    }[dataset]
    candidates = []
    for filename in filenames:
        candidates.extend(
            [
                root / filename,
                root / dataset / filename,
                root / "spider_data" / filename,
                root / "dev" / filename,
                root / "train" / filename,
                *[path / filename for path in sorted(root.glob("dev_*"))],
                *[path / filename for path in sorted(root.glob("train_*"))],
            ]
        )

    seen: set[Path] = set()
    for candidate in candidates:
        candidate = candidate.resolve()
        if candidate in seen:
            continue
        seen.add(candidate)
        if candidate.exists():
            return candidate
    tried = ", ".join(str(path) for path in candidates)
    raise FileNotFoundError(f"Could not find {dataset} {split} file. Tried: {tried}")


def load_text_to_sql_examples(dataset: str, split_file: Path) -> list[TextToSQLExample]:
    records = _read_json_list(split_file)
    examples: list[TextToSQLExample] = []
    for index, record in enumerate(records):
        try:
            db_id = str(record["db_id"])
            question = str(record["question"])
        except KeyError as exc:
            raise KeyError(f"{dataset} example {index} missing required field {exc}") from exc

        if dataset == "spider":
            sql_key_candidates = ("query", "SQL", "sql")
        elif dataset == "bird":
            sql_key_candidates = ("SQL", "query", "sql")
        else:
            raise ValueError(f"Unsupported dataset: {dataset}")

        gold_sql = None
        for key in sql_key_candidates:
            if key in record:
                gold_sql = str(record[key])
                break
        if gold_sql is None:
            raise KeyError(
                f"{dataset} example {index} missing SQL field; tried {sql_key_candidates}"
            )

        examples.append(
            TextToSQLExample(
                index=index,
                db_id=db_id,
                question=question,
                gold_sql=" ".join(gold_sql.strip().split()),
                evidence=str(record.get("evidence", "") or ""),
            )
        )
    return examples


def resolve_db_path(dataset: str, root: Path, db_id: str) -> Path:
    if dataset == "spider":
        candidates = [
            root / "database" / db_id / f"{db_id}.sqlite",
            root / "spider_data" / "database" / db_id / f"{db_id}.sqlite",
        ]
    elif dataset == "bird":
        candidates = [
            root / "dev_databases" / db_id / f"{db_id}.sqlite",
            root / "train_databases" / db_id / f"{db_id}.sqlite",
            root / "database" / db_id / f"{db_id}.sqlite",
            root / "databases" / db_id / f"{db_id}.sqlite",
            root / db_id / f"{db_id}.sqlite",
        ]
        for dated_root in sorted(root.glob("dev_*")) + sorted(root.glob("train_*")):
            candidates.extend(
                [
                    dated_root / "dev_databases" / db_id / f"{db_id}.sqlite",
                    dated_root / "train_databases" / db_id / f"{db_id}.sqlite",
                ]
            )
    else:
        raise ValueError(f"Unsupported dataset: {dataset}")

    for candidate in candidates:
        if candidate.exists():
            return candidate
    return candidates[0]
