# Log-TIR Remaining Tasks

This file is the current server-side Codex handoff. Completed tasks from the
previous checklist have been collapsed into a short status summary. Execute the
remaining tasks below in order.

## Completed Status

Official Spider results already recorded in `docs/experiment-results.md`:

- SFT baseline full Spider dev: `724 / 1034 = 70.02%`.
- Multi-turn GRPO best checkpoint full Spider dev:
  `742 / 1034 = 71.76%`.
- Selected multi-turn GRPO checkpoint:
  `checkpoints/qwen2.5-coder-3b-logtir-grpo-ckpt/best_global_step150_hf`.
- Same-checkpoint ablation on the selected multi-turn GRPO checkpoint:
  - `max_turns=1`: `745 / 1034 = 72.05%`
  - `max_turns=2`: `770 / 1034 = 74.47%`
  - raw multi-turn gain: `+2.42 pp`
  - timeout-excluded gain: about `+2.43 pp`
  - turn-2 rescues: `25`
  - timeout rescues: `0`
  - non-timeout rescues: `13` wrong-result, `8` schema/hallucination,
    `4` empty-result

This supports an inference-time self-correction claim for the selected
checkpoint. It does not yet prove that multi-turn GRPO training is better than
single-turn GRPO training. The next priority is the control run below.

## Task 1: Single-Turn GRPO Control Run

Goal: test whether multi-turn GRPO training matters beyond ordinary single-turn
GRPO reward optimization.

Run a separate GRPO training job from the same SFT checkpoint with multi-turn
disabled. Keep all other hyperparameters as close as possible to the successful
multi-turn GRPO run.

Do not overwrite the existing multi-turn checkpoint directories.

Recommended launch:

```bash
git pull

export GRPO_MULTI_TURN=0
export GRPO_ACTOR_MODEL=checkpoints/qwen2.5-coder-3b-logtir-sft
export GRPO_OUTPUT_DIR=checkpoints/qwen2.5-coder-3b-logtir-grpo-singleturn
export GRPO_CKPT_DIR=checkpoints/qwen2.5-coder-3b-logtir-grpo-singleturn-ckpt
export WANDB_RUN_NAME=grpo-singleturn-control

START_RAY=1 bash scripts/train_grpo_openrlhf.sh
```

If the server uses a local `configs/openrlhf_day3.env`, keep that file's
successful GPU, batch, max sample, timeout, wandb, and Ray settings. Override
only the variables above unless there is a concrete launch error.

During training, track the same checkpoint-selection signal used for the
multi-turn run:

- `eval_spider_pass1` by global step
- best global step
- whether later steps regress
- final selected checkpoint path

Expected output naming:

```text
checkpoints/qwen2.5-coder-3b-logtir-grpo-singleturn-ckpt/
logs/infer_eval_spider_grpo_singleturn_best_dev.json
logs/infer_eval_spider_grpo_singleturn_best_dev_turn1.json
logs/infer_eval_spider_grpo_singleturn_best_dev_turn2.json
logs/trajectories_spider_grpo_singleturn_best_dev_turn1.jsonl
logs/trajectories_spider_grpo_singleturn_best_dev_turn2.jsonl
```

After training, evaluate the best single-turn-GRPO checkpoint on full Spider dev:

```bash
SINGLE_TURN_BEST=checkpoints/qwen2.5-coder-3b-logtir-grpo-singleturn-ckpt/<best_step_hf>

python3 infer_eval.py \
  --backend vllm \
  --dataset spider \
  --data-root data/spider \
  --model "$SINGLE_TURN_BEST" \
  --output logs/infer_eval_spider_grpo_singleturn_best_dev.json

python3 multi_turn_infer_eval.py \
  --backend vllm \
  --dataset spider \
  --data-root data/spider \
  --model "$SINGLE_TURN_BEST" \
  --max-turns 1 \
  --output logs/infer_eval_spider_grpo_singleturn_best_dev_turn1.json \
  --trajectories-out logs/trajectories_spider_grpo_singleturn_best_dev_turn1.jsonl

python3 multi_turn_infer_eval.py \
  --backend vllm \
  --dataset spider \
  --data-root data/spider \
  --model "$SINGLE_TURN_BEST" \
  --max-turns 2 \
  --output logs/infer_eval_spider_grpo_singleturn_best_dev_turn2.json \
  --trajectories-out logs/trajectories_spider_grpo_singleturn_best_dev_turn2.jsonl
```

Acceptance criteria:

- The run uses `GRPO_MULTI_TURN=0`.
- The single-turn control uses a distinct checkpoint directory.
- The selected best checkpoint is evaluated on full Spider dev.
- The report compares these rows under the same evaluator where possible:
  - SFT baseline
  - multi-turn GRPO best150, `max_turns=1`
  - multi-turn GRPO best150, `max_turns=2`
  - single-turn GRPO best, `max_turns=1`
  - single-turn GRPO best, `max_turns=2`

Interpretation:

- If single-turn GRPO `max_turns=1` is close to multi-turn GRPO `max_turns=1`,
  then most training gain may come from GRPO reward optimization rather than
  multi-turn training.
- If multi-turn GRPO `max_turns=2` clearly beats single-turn GRPO `max_turns=2`,
  then the project has stronger evidence that multi-turn training itself helped.

## Task 2: Add Stable Timeout-Excluded Field Aliases

The current multi-turn evaluator works, but its output field names do not fully
match the previous task spec. Add stable aliases before more reports depend on
the JSON schema.

In `multi_turn_infer_eval.py`, keep existing fields and add these aliases:

- `accuracy_excluding_first_turn_timeout`
- `turn1_accuracy_excluding_timeout`
- `final_accuracy_excluding_first_turn_timeout`
- `turn2_rescue_rate_excluding_first_turn_timeout`

The existing plural field `final_accuracy_excluding_first_turn_timeouts` can
remain for backward compatibility.

Add or update tests in `tests/test_multi_turn_infer_eval.py`, then run:

```bash
python3 -m pytest tests/test_multi_turn_infer_eval.py
```

Acceptance criteria:

- The test passes.
- New summary JSONs contain both old and new timeout-excluded field names.

## Task 3: BIRD Transfer Evaluation

Run BIRD transfer after confirming the server has usable BIRD data. The dataset
must exist on the server-local filesystem under `data/bird`; a Mac-only copy is
not sufficient for server evaluation. On the Mac control plane, `data/bird`
exists, but the previous server run reported that the server path was missing.

First verify on the server:

```bash
test -e data/bird
find data/bird -maxdepth 3 -type f | head
```

If BIRD is missing on the server, download it or sync the Mac copy to the
server-local path `data/bird` before evaluating. Do not skip BIRD just because
the path is missing; first attempt to make the dataset available locally on the
server. Only record a missing-data caveat if download/sync/unpack is blocked.

Evaluate SFT and multi-turn GRPO best150:

```bash
python3 infer_eval.py \
  --backend vllm \
  --dataset bird \
  --data-root data/bird \
  --model checkpoints/qwen2.5-coder-3b-logtir-sft \
  --output logs/infer_eval_bird_sft_dev.json

python3 infer_eval.py \
  --backend vllm \
  --dataset bird \
  --data-root data/bird \
  --model checkpoints/qwen2.5-coder-3b-logtir-grpo-ckpt/best_global_step150_hf \
  --output logs/infer_eval_bird_grpo_best150_dev.json
```

After Task 1 finishes, also evaluate the selected single-turn-GRPO checkpoint on
BIRD:

```bash
python3 infer_eval.py \
  --backend vllm \
  --dataset bird \
  --data-root data/bird \
  --model "$SINGLE_TURN_BEST" \
  --output logs/infer_eval_bird_grpo_singleturn_best_dev.json
```

Acceptance criteria:

- If BIRD data exists, record SFT, multi-turn GRPO, and single-turn GRPO results.
- If BIRD data is still unavailable, record the exact missing path or file.

## Task 4: Update Experiment Records

After Tasks 1-3, update `docs/experiment-results.md` with:

- Single-turn GRPO training config summary.
- Single-turn GRPO best global step and checkpoint path.
- Spider full-dev result from `infer_eval.py`.
- Spider `max_turns=1` and `max_turns=2` results from
  `multi_turn_infer_eval.py`.
- Comparison against multi-turn GRPO best150.
- BIRD transfer results, if available.
- Any missing-data caveats.

Commit and push:

```bash
git add docs/experiment-results.md tasks.md multi_turn_infer_eval.py tests/test_multi_turn_infer_eval.py
git commit -m "Record single-turn GRPO control plan and results"
git push origin main
```

## Current Claim Boundary

Safe claim now:

```text
GRPO improved Spider dev execution match from 70.0% to 71.8% over an SFT
cold-start baseline, and same-checkpoint execution-feedback self-correction
improved the selected GRPO checkpoint from 72.1% to 74.5% on Spider dev without
timeout-rescue inflation.
```

Do not claim that multi-turn GRPO training itself is better than single-turn GRPO
training until Task 1 is complete.
