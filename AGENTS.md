# AGENTS.md

## Project

- Repo: `Log-TIR`
- Goal: build a fully local Text-to-SQL self-correction agent with GRPO on top of `Qwen2.5-Coder-3B-Instruct`
- Primary benchmark: Spider
- Secondary transfer benchmark: BIRD
- Intended outcome: a strong resume-grade LLM/agent/RL systems project for algorithm internship applications

## Core Thesis

- The agent should read a natural-language question, generate SQL, execute it in a sandbox, read traceback or empty-result feedback, and retry.
- The project must stay fully local and avoid external inference APIs during training and evaluation.
- Reward shaping is staged as `format -> no-error -> exec-match`, with `exec-match` dominating.
- A short SFT cold start is required before GRPO. Direct GRPO from the base model is expected to stall.

## Training Defaults

- Reward weights: `format = 0.1`, `no-error = 0.2`, `exec-match = 1.0`
- SFT cold start target: `2000` formatted trajectories with `<thought>` and `<action>`
- GRPO group size: `G = 4-8`
- Self-correction budget: at most `2` turns per episode
- Primary base model: `Qwen2.5-Coder-3B-Instruct`

## Day 1 Plan

Deliverables:

1. Download and unpack Spider with executable SQLite databases.
2. Implement `sandbox.py` with read-only SQLite access, subprocess isolation, and a 3 second timeout.
3. Implement `eval.py` for execution-match evaluation.
4. Validate the evaluator by running Spider dev gold SQL and target `>=95%` execution accuracy.

Suggested order:

1. Set up `data/spider/`.
2. Write the SQL runner and normalization helpers.
3. Run a small gold-SQL smoke test.
4. Run full Spider dev evaluation and inspect failures.

## Day 2 Focus

Deliverables:

1. Build a Spider-to-SFT conversion script for `<thought>/<action>` cold-start data.
2. Keep deterministic SFT responses explicitly `format_only`; do not synthesize fake reasoning from gold SQL.
3. Generate an initial `2000`-example JSONL file from executable Spider training examples.
4. Optionally export teacher-model prompt requests for real reasoning generation from schema and question only.
5. Add tests for schema rendering, tagged-response validation, and dataset generation.

## Day 3 Focus

Deliverables:

1. Add OpenRLHF SFT startup script using generated Spider SFT JSONL.
2. Add OpenRLHF GRPO startup script using `train_ppo_ray` with `--algo.advantage.estimator group_norm`.
3. Add GRPO prompt/label JSONL generation for reward-time execution matching.
4. Add local reward function with `format = 0.1`, `no-error = 0.2`, `exec-match = 1.0`.
5. Add a two-turn self-correction rollout helper that feeds execution feedback into the second prompt.
6. Keep launch-time settings configurable through `configs/openrlhf_day3.env`.

## How To Run

Spider evaluator:

```bash
python3 eval.py --spider-root data/spider --use-gold-predictions --limit 100
python3 eval.py --spider-root data/spider --use-gold-predictions --failures-out spider_gold_failures.json
python3 eval.py --dataset bird --data-root data/bird --use-gold-predictions --limit 100
```

SFT cold-start data generation:

```bash
python3 sft_data.py --spider-root data/spider --output data/sft/spider_sft_2000.jsonl --limit 2000 --require-executable-gold --teacher-requests-output data/sft/spider_teacher_requests_2000.jsonl
```

GRPO prompt data generation:

```bash
python3 rl_data.py --spider-root data/spider --output data/rl/spider_grpo_train_2000.jsonl --limit 2000
python3 rl_data.py --spider-root data/spider --split dev --output data/rl/spider_grpo_dev_100.jsonl --limit 100
```

OpenRLHF launch scripts:

```bash
bash scripts/train_sft_openrlhf.sh
START_RAY=1 bash scripts/train_grpo_openrlhf.sh
GRPO_MULTI_TURN=0 START_RAY=1 bash scripts/train_grpo_openrlhf.sh
```

Standalone checkpoint inference and evaluation:

```bash
python3 infer_eval.py --backend vllm --dataset spider --data-root data/spider --model checkpoints/qwen2.5-coder-3b-logtir-sft --output logs/infer_eval_spider_dev.json
python3 infer_eval.py --backend vllm --dataset bird --data-root data/bird --model checkpoints/qwen2.5-coder-3b-logtir-grpo --output logs/infer_eval_bird_dev.json
python3 infer_eval.py --backend transformers --dataset spider --data-root data/spider --model checkpoints/qwen2.5-coder-3b-logtir-sft --limit 10
```

Run logging and remote log sync:

```bash
LOGTIR_ENABLE_RUN_LOGS=1 bash scripts/train_sft_openrlhf.sh
LOGTIR_WANDB=1 WANDB_API_KEY=... bash scripts/train_sft_openrlhf.sh
scripts/sync_logs.sh user@your.server:~/Log-TIR/logs remote-logs
```

Wandb preflight:

- Before every SFT or GRPO training run that enables wandb, test wandb connectivity first and record the result in the run log.
- Prefer `WANDB_MODE=offline` for long domestic-server GRPO runs unless the preflight proves online wandb is stable; wandb init timeouts must not be allowed to kill Ray actors mid-launch.
- `scripts/train_grpo_openrlhf.sh` performs this preflight by default when `LOGTIR_WANDB=1`; set `LOGTIR_WANDB_PREFLIGHT_STRICT=1` only when training should fail instead of falling back from an unhealthy wandb connection.

Direct sandbox smoke tests:

```bash
python3 sandbox.py --db path/to/db.sqlite --sql "select count(*) from some_table"
python3 sandbox.py --db path/to/db.sqlite --sql "drop table some_table"
```

## Repository Layout Target

Expected layout:

```text
Log-TIR/
├── AGENTS.md
├── CLAUDE.md
├── data/
│   ├── spider/
│   └── bird/
├── logs/
├── remote-logs/
├── sandbox.py
├── eval.py
└── ...
```

Training logs should follow:

```text
logs/
├── run_YYYYMMDD_HHMM/
│   ├── config.json
│   ├── metrics.jsonl
│   ├── trajectories.jsonl
│   ├── train.stdout.log
│   └── train.stderr.log
└── latest -> run_YYYYMMDD_HHMM
```

## Evaluation Rules

- Main metric: execution match on Spider and BIRD.
- Normalize execution outputs so numeric and string formatting noise does not dominate reward.
- Preserve row order for `ORDER BY` queries; otherwise compare as multisets.
- Do not trust model-side metrics until the gold SQL baseline is validated on the dev split.
- Do not train on fake chain-of-thought synthesized from gold SQL. Use `thought_mode=format_only` for deterministic SFT, or generate real teacher reasoning from schema and question only.

## Testing Strategy

- Keep `tests/` focused on small SQLite fixtures built at test time.
- `sandbox.py` must be covered for timeout handling, syntax errors, read-only rejection, and valid read-only queries.
- `eval.py` must be covered for `ORDER BY` sensitivity, numeric normalization, and Spider-style dataset evaluation on a tiny synthetic fixture.
- `sft_data.py` must be covered for schema rendering, `<thought>/<action>` validation, and JSONL generation on a tiny Spider-style fixture.
- Before changing reward logic or rollout formatting, re-run the Spider gold-SQL evaluator and `pytest`.

## Engineering Rules

- Keep reward functions pure and unit-testable.
- Create the run directory, update `logs/latest`, dump config, and initialize tracking at the start of every training run.
- `config.json` should capture hyperparameters, git commit and branch and dirty flag and diff, hostname, python and torch and cuda versions, GPU type, and launch command.
- Store logs as JSONL rather than large free-form text files.

## Infrastructure Notes

- Preferred training stack: `OpenRLHF` + Ray
- Main training machine: domestic server
- Backup compute: vast.ai
- Mac is the control plane and should pull logs from the server instead of relying on reverse sync

## Constraints

- Do not revive the synthetic Gemini-style personal-log dataset as the main training source.
- Do not treat a weak evaluator as ground truth; evaluator validation is a hard gate.
- Do not over-reward trivial no-error outputs such as `SELECT 1`.

## Resume-Level Claims To Preserve

- Zero-API local sandbox execution for scalable rollouts
- Multi-stage reward shaping to reduce sparse-reward collapse
- Multi-turn self-correction using traceback feedback

## Continuation Checklist

When resuming later:

1. Read `AGENTS.md`.
2. Inspect the repo tree and current missing pieces.
3. Check recent git history if git is initialized.
4. Verify whether Spider data is already present.
5. Re-run the Spider gold-SQL evaluator before changing reward logic.
