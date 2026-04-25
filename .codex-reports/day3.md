# Day 3 Report

## Scope

- Added OpenRLHF SFT launch script
- Added OpenRLHF GRPO launch script
- Added GRPO prompt/label JSONL generator
- Added local reward function for format, no-error, and execution-match scoring
- Added cached gold-SQL execution inside reward scoring
- Added self-correction rollout helper with up to 2 turns and execution-feedback prompt updates
- Added tests for GRPO data and reward scoring

## Files Added Or Updated

- `rl_data.py`
- `agent_rollout.py`
- `openrlhf_reward.py`
- `scripts/train_sft_openrlhf.sh`
- `scripts/train_grpo_openrlhf.sh`
- `configs/openrlhf_day3.env.example`
- `tests/test_rl_data.py`
- `tests/test_agent_rollout.py`
- `tests/test_openrlhf_reward.py`
- `.codex-reports/day3.md`

## Commands

Build SFT data:

```bash
python3 sft_data.py --spider-root data/spider --output data/sft/spider_sft_2000.jsonl --limit 2000 --require-executable-gold --teacher-requests-output data/sft/spider_teacher_requests_2000.jsonl
```

Build GRPO train prompt data:

```bash
python3 rl_data.py --spider-root data/spider --output data/rl/spider_grpo_train_2000.jsonl --limit 2000
```

Build GRPO dev eval prompt data:

```bash
python3 rl_data.py --spider-root data/spider --split dev --output data/rl/spider_grpo_dev_100.jsonl --limit 100
```

Observed dev output:

```json
{
  "output_path": "data/rl/spider_grpo_dev_100.jsonl",
  "count": 100,
  "split": "dev",
  "seed": 42
}
```

Observed output:

```json
{
  "output_path": "data/rl/spider_grpo_train_2000.jsonl",
  "count": 2000,
  "split": "train_spider",
  "seed": 42
}
```

Observed label-path check:

```text
database/flight_1/flight_1.sqlite
absolute False
```

Start SFT on an OpenRLHF server:

```bash
bash scripts/train_sft_openrlhf.sh
```

Start GRPO on an OpenRLHF server:

```bash
START_RAY=1 bash scripts/train_grpo_openrlhf.sh
```

## Notes

- The OpenRLHF commands follow the current dot-argument style used by upstream examples.
- GRPO is selected with `--algo.advantage.estimator group_norm`.
- GRPO uses `--algo.kl.use_loss` and `--algo.kl.estimator k3` by default, matching OpenRLHF's GRPO KL-loss guidance.
- The reward function uses `0.1` format reward, `0.2` no-error reward, and `1.0` execution-match reward.
- GRPO defaults to `rollout.batch_size=32` and `n_samples_per_prompt=4` to start with 128 generations per step on a 24GB 4090.
- The GRPO script uses `--eval.steps 50` when `data/rl/spider_grpo_dev_100.jsonl` exists.
- GRPO labels store database paths relative to `SPIDER_ROOT`, so generated prompt JSONL can move from Mac to server.
- `data/` remains ignored by git, so generated SFT and GRPO JSONL files must be generated or copied separately on the server.
- OpenRLHF's default generation loop remains single-turn; the project-specific multi-turn behavior is implemented in `agent_rollout.py` and should be wired into a custom OpenRLHF agent function or used for rollout analysis before claiming full agentic RL training.

## Validation

```bash
python3 -m py_compile rl_data.py openrlhf_reward.py agent_rollout.py tests/test_rl_data.py tests/test_openrlhf_reward.py tests/test_agent_rollout.py
python3 -m pytest tests -q
bash -n scripts/train_sft_openrlhf.sh
bash -n scripts/train_grpo_openrlhf.sh
```

Observed test result:

```text
...........................                                              [100%]
27 passed in 1.03s
```

## Review Fix

- Added a no-error-but-wrong-result reward test for the expected `0.3` score.
- Added cached gold execution so the same `(db_path, gold_sql, timeout)` is not re-run for every rollout sample.
- Lowered default rollout batch size from `128` to `32` while keeping `n_samples_per_prompt=4`.
- Added OpenRLHF eval hook arguments for a 100-example Spider dev prompt set every 50 steps.
- Added `agent_rollout.py` to model the planned two-turn self-correction loop with SQLite error feedback.
