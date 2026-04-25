# Day 1 Report

## Scope

- Read `CLAUDE.md` and generated `AGENTS.md`
- Downloaded and unpacked the official Spider dataset
- Implemented `sandbox.py`
- Implemented `eval.py`
- Added pytest coverage for sandbox and evaluator behavior

## Files Added Or Updated

- `AGENTS.md`
- `sandbox.py`
- `eval.py`
- `tests/test_sandbox.py`
- `tests/test_eval.py`
- `.codex-reports/day1.md`

## Dataset

- Archive: `data/spider_data.zip`
- Unpacked root: `data/spider/spider_data/`
- Dev file: `data/spider/spider_data/dev.json`

## Validation Runs

Smoke test command:

```bash
python3 eval.py --spider-root data/spider --use-gold-predictions --limit 100
```

Observed result:

```json
{
  "total": 100,
  "matched": 100,
  "accuracy": 1.0
}
```

Full validation command:

```bash
python3 eval.py --spider-root data/spider --use-gold-predictions --failures-out spider_gold_failures.json
```

Observed result:

```json
{
  "total": 1034,
  "matched": 1034,
  "accuracy": 1.0
}
```

Pytest command:

```bash
python3 -m pytest tests -q
```

Observed result:

```text
.........                                                                [100%]
9 passed in 0.43s
```

## Notes

- `sandbox.py` uses subprocess isolation, a 3 second timeout, read-only SQLite URI, and SQLite authorizer hooks.
- `eval.py` compares ordered outputs as lists when `ORDER BY` is present and compares unordered outputs as multisets otherwise.
- Numeric normalization treats `1` and `1.0` as equal.
- Byte strings from SQLite are explicitly decoded during normalization.
- `spider_gold_failures.json` is an empty list after the full gold-SQL validation run.
