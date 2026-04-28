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
single-turn GRPO training. The next priorities are the inference-time turn sweep
and the single-turn control run below.

## Task 1: Inference-Time More-Turn Sweep

Goal: test whether allowing more inference-time repair attempts gives additional
self-correction gains beyond the current 2-turn result. This is inference only;
do not retrain a new model for this task.

First extend `multi_turn_infer_eval.py` so it supports arbitrary
`--max-turns >= 1` instead of only `1` or `2`.

Implementation requirements:

- Reuse the same checkpoint and decoding settings across all turn counts.
- For each failed turn, append the latest execution feedback and generate the
  next repair prompt.
- Stop early when a turn reaches execution match.
- Preserve one trajectory row per example with all turns in order.
- Report cumulative final accuracy and marginal rescues by turn.
- Keep timeout rescues separate from non-timeout semantic/error rescues.

The summary JSON must include at least:

- `total`
- `matched`
- `accuracy`
- `max_turns`
- `turn1_matched`
- `turn1_accuracy`
- `final_matched`
- `final_accuracy`
- `final_accuracy_excluding_first_turn_timeout`
- `rescued_by_turn2`
- `rescued_by_turn3`
- `rescued_by_turn4`
- `rescue_rate_by_turn`
- `marginal_accuracy_gain_by_turn`
- `timeout_rescue_by_turn`
- `non_timeout_rescue_by_turn`
- `first_turn_error_counts`
- `rescue_by_first_turn_error`

Run a full Spider dev turn sweep for the selected multi-turn GRPO checkpoint:

```bash
for TURNS in 1 2 3 4; do
  python3 multi_turn_infer_eval.py \
    --backend vllm \
    --dataset spider \
    --data-root data/spider \
    --model checkpoints/qwen2.5-coder-3b-logtir-grpo-ckpt/best_global_step150_hf \
    --max-turns "$TURNS" \
    --output "logs/infer_eval_spider_grpo_best150_dev_turn${TURNS}.json" \
    --trajectories-out "logs/trajectories_spider_grpo_best150_dev_turn${TURNS}.jsonl"
done
```

Acceptance criteria:

- Full Spider dev finishes for `max_turns=1,2,3,4`.
- The `max_turns=1` and `max_turns=2` numbers reproduce the existing results
  within normal deterministic-decoding tolerance.
- The report states marginal gains:

```text
turn 1 -> 2 gain
turn 2 -> 3 gain
turn 3 -> 4 gain
```

- Timeout rescues are reported separately and are not counted as semantic
  self-correction evidence.

Interpretation:

- If gains saturate after 2 turns, keep the project claim focused on 2-turn
  self-correction.
- If turns 3 or 4 add meaningful non-timeout rescues, report that inference-time
  self-correction has additional headroom, especially on harder examples.

After the single-turn GRPO control checkpoint is selected, run the same
`max_turns=1,2,3,4` sweep on that checkpoint too. This will show whether extra
inference turns help only the multi-turn-trained model or also help the
single-turn-trained model.

## Task 2: Single-Turn GRPO Control Run

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
logs/infer_eval_spider_grpo_singleturn_best_dev_turn3.json
logs/infer_eval_spider_grpo_singleturn_best_dev_turn4.json
logs/trajectories_spider_grpo_singleturn_best_dev_turn1.jsonl
logs/trajectories_spider_grpo_singleturn_best_dev_turn2.jsonl
logs/trajectories_spider_grpo_singleturn_best_dev_turn3.jsonl
logs/trajectories_spider_grpo_singleturn_best_dev_turn4.jsonl
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

for TURNS in 1 2 3 4; do
  python3 multi_turn_infer_eval.py \
    --backend vllm \
    --dataset spider \
    --data-root data/spider \
    --model "$SINGLE_TURN_BEST" \
    --max-turns "$TURNS" \
    --output "logs/infer_eval_spider_grpo_singleturn_best_dev_turn${TURNS}.json" \
    --trajectories-out "logs/trajectories_spider_grpo_singleturn_best_dev_turn${TURNS}.jsonl"
done
```

Acceptance criteria:

- The run uses `GRPO_MULTI_TURN=0`.
- The single-turn control uses a distinct checkpoint directory.
- The selected best checkpoint is evaluated on full Spider dev.
- The report compares these rows under the same evaluator where possible:
  - SFT baseline
  - multi-turn GRPO best150, `max_turns=1,2,3,4`
  - single-turn GRPO best, `max_turns=1,2,3,4`

Interpretation:

- If single-turn GRPO `max_turns=1` is close to multi-turn GRPO `max_turns=1`,
  then most training gain may come from GRPO reward optimization rather than
  multi-turn training.
- If multi-turn GRPO clearly beats single-turn GRPO under the same inference
  turn budget, especially at `max_turns=2,3,4`, then the project has stronger
  evidence that multi-turn training itself helped.

## Task 3: Add Stable Timeout-Excluded Field Aliases

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

## Task 4: BIRD Transfer Evaluation

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

After Task 2 finishes, also evaluate the selected single-turn-GRPO checkpoint on
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

## Task 5: Update Experiment Records

After Tasks 1-4, update `docs/experiment-results.md` with:

- Multi-turn GRPO best150 inference-time turn sweep for `max_turns=1,2,3,4`.
- Marginal gain from each extra inference turn.
- Single-turn GRPO training config summary.
- Single-turn GRPO best global step and checkpoint path.
- Spider full-dev result from `infer_eval.py`.
- Spider `max_turns=1,2,3,4` results from
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
training until the inference-time turn sweep and single-turn GRPO control are
both complete.
