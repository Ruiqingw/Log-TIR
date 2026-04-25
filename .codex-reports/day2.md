# Day 2 Report

## Scope

- Implemented SFT cold-start data generation for Spider
- Defined a consistent `<thought>/<action>` response format without gold-SQL-derived fake reasoning
- Added response-format parsing and validation helpers
- Added optional teacher-request export using schema and question only
- Added executable-gold filtering for generated SFT examples
- Added tests for schema rendering, tagged-format validation, and JSONL generation

## Files Added Or Updated

- `sft_data.py`
- `tests/test_sft_data.py`
- `.codex-reports/day2.md`
- `AGENTS.md`
- `.gitignore`

## Deliverable

Primary command:

```bash
python3 sft_data.py --spider-root data/spider --output data/sft/spider_sft_2000.jsonl --limit 2000 --require-executable-gold --teacher-requests-output data/sft/spider_teacher_requests_2000.jsonl
```

Observed output:

```json
{
  "output_path": "data/sft/spider_sft_2000.jsonl",
  "count": 2000,
  "seed": 42,
  "include_train_others": false,
  "require_executable_gold": true,
  "skipped_non_executable": 1,
  "teacher_requests_output": "data/sft/spider_teacher_requests_2000.jsonl"
}
```

Observed file check:

```text
exists True
lines 2000
teacher_requests_exists True
teacher_requests_lines 2000
```

Output schema:

- A JSONL file with up to `2000` Spider training examples
- Each record contains:
  - `task`
  - `split`
  - `db_id`
  - `question`
  - `prompt`
  - `response`
  - `gold_sql`
  - `thought_mode`

## Notes

- The prompt includes rendered schema text, the user question, and explicit output-tag rules.
- The deterministic response always uses:
  - `<thought>...</thought>`
  - `<action>...</action>`
- The deterministic `<thought>` is explicitly `format_only` and does not mention tables extracted from gold SQL.
- The `<action>` section contains exactly one normalized SQL query.
- The optional teacher-request JSONL contains only `record_id`, metadata, and chat `messages`; it does not include `gold_sql`.
- The teacher system prompt asks a stronger model to reason from schema and question only, matching the review recommendation for real reasoning data.

## Validation Commands

```bash
python3 -m py_compile sft_data.py tests/test_sft_data.py
python3 -m pytest tests -q
python3 sft_data.py --spider-root data/spider --output data/sft/spider_sft_2000.jsonl --limit 2000 --require-executable-gold --teacher-requests-output data/sft/spider_teacher_requests_2000.jsonl
```

Observed test result:

```text
.................                                                        [100%]
17 passed in 0.96s
```

## Review Fix

The earlier version generated `<thought>` by extracting table names from gold SQL. That leaked training-only information and created homogeneous pseudo-reasoning. The current version removes that path. Deterministic SFT data now teaches output format only; real reasoning should come from the exported teacher requests if used.
