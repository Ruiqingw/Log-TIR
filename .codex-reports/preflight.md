# Preflight Infrastructure Report

## Scope

- Verified the current progress assessment against the repository.
- Added reproducible run logging utilities.
- Added standalone checkpoint inference and execution-match evaluation.
- Added Spider/BIRD dataset adapter for evaluation.
- Switched standalone inference to vLLM by default while retaining a transformers fallback.
- Downloaded official BIRD dev package locally under `data/bird/`.
- Added optional wandb initialization hook and remote log sync script.
- Wired SFT and GRPO launch scripts to create `logs/run_*` directories.

## Files Added Or Updated

- `dataset_adapter.py`
- `run_logging.py`
- `infer_eval.py`
- `scripts/sync_logs.sh`
- `scripts/train_sft_openrlhf.sh`
- `scripts/train_grpo_openrlhf.sh`
- `configs/openrlhf_day3.env.example`
- `eval.py`
- `AGENTS.md`
- `.gitignore`
- `tests/test_dataset_adapter.py`
- `tests/test_infer_eval.py`
- `tests/test_run_logging.py`
- `tests/test_eval.py`

## Progress Check

- Day 1 sandbox and Spider evaluator are present and tested.
- Day 2 SFT data generation is present and tested.
- Day 3 GRPO data, reward function, launch scripts, and independent self-correction helper are present and tested.
- Missing pre-experiment infrastructure was real: run logging, standalone inference/evaluation, BIRD evaluation adapter, and log sync were not implemented before this pass.

## Commands

Run all tests:

```bash
python3 -m pytest tests -q
```

Observed:

```text
39 passed in 2.99s
```

Validate scripts:

```bash
bash -n scripts/train_sft_openrlhf.sh
bash -n scripts/train_grpo_openrlhf.sh
bash -n scripts/sync_logs.sh
```

Validate Spider evaluator still works:

```bash
python3 eval.py --spider-root data/spider --use-gold-predictions --limit 100
```

Observed:

```json
{
  "total": 100,
  "matched": 100,
  "accuracy": 1.0
}
```

Validate BIRD evaluator on downloaded official dev data:

```bash
python3 eval.py --dataset bird --data-root data/bird --use-gold-predictions --limit 20
```

Observed:

```json
{
  "total": 20,
  "matched": 20,
  "accuracy": 1.0
}
```

BIRD local data:

```text
data/bird/dev_20240627/dev.json: 1534 examples
data/bird/dev_20240627/dev_databases: 1.4G
data/bird total: 2.0G
```

## New Run Commands

Standalone Spider checkpoint eval with vLLM:

```bash
python3 infer_eval.py --backend vllm --dataset spider --data-root data/spider --model checkpoints/qwen2.5-coder-3b-logtir-sft --output logs/infer_eval_spider_dev.json
```

Small fallback eval without vLLM:

```bash
python3 infer_eval.py --backend transformers --dataset spider --data-root data/spider --model checkpoints/qwen2.5-coder-3b-logtir-sft --limit 10
```

Standalone BIRD checkpoint eval with vLLM:

```bash
python3 infer_eval.py --backend vllm --dataset bird --data-root data/bird --model checkpoints/qwen2.5-coder-3b-logtir-grpo --output logs/infer_eval_bird_dev.json
```

Remote log sync:

```bash
scripts/sync_logs.sh user@your.server:~/Log-TIR/logs remote-logs
```

## Remaining Caveats

- OpenRLHF training was not run locally.
- wandb is optional through `LOGTIR_WANDB=1` plus `WANDB_API_KEY`; the scripts pass OpenRLHF's `--use_wandb` flag when enabled.
- BIRD support is an adapter for common BIRD dev formats and should be validated against the downloaded official BIRD dev files.
- Multi-turn self-correction is now wired through `openrlhf_agent.py` and `--agent_func_path`; it still needs server-side OpenRLHF/Ray/vLLM smoke validation.
