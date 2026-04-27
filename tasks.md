# Log-TIR Next Tasks

This file is the handoff checklist for the server-side Codex run.

## Current Official State

- Official SFT baseline on full Spider dev: `724 / 1034 = 0.7002`.
- Official GRPO candidate on full Spider dev: `742 / 1034 = 0.7176`.
- Absolute gain over SFT: `+1.74 pp`.
- Selected checkpoint:
  `checkpoints/qwen2.5-coder-3b-logtir-grpo-ckpt/best_global_step150_hf`.
- Official GRPO eval artifact on the server:
  `logs/infer_eval_spider_grpo_best150_dev.json`.
- Local synced artifact path used by the Mac control plane:
  `remote-logs/infer_eval_spider_grpo_best150_dev.json`.
- Training internal eval selected step 150:
  - step 50: `eval_spider_pass1 = 0.8892`
  - step 100: `eval_spider_pass1 = 0.9210`
  - step 150: `eval_spider_pass1 = 0.9315`, selected best
  - step 200: `eval_spider_pass1 = 0.9204`, regressed

Do not continue training before the evaluation tasks below are done. The next
priority is to prove where the GRPO gain comes from.

## Task 1: Rehydrate And Verify The Current Result

On the server:

```bash
git pull
jq '{dataset,total,matched,accuracy}' logs/infer_eval_spider_grpo_best150_dev.json
```

Expected result:

```json
{
  "dataset": "spider",
  "total": 1034,
  "matched": 742,
  "accuracy": 0.7176015473887815
}
```

Also verify that the selected checkpoint exists:

```bash
test -d checkpoints/qwen2.5-coder-3b-logtir-grpo-ckpt/best_global_step150_hf
```

Acceptance criteria:

- The checkpoint directory exists.
- The eval artifact reports `742 / 1034`.
- `docs/experiment-results.md` matches the verified result.

## Task 2: Same-Checkpoint Multi-Turn Ablation

Goal: isolate the effect of self-correction without changing the model.

Run the same checkpoint on the same Spider dev split under two settings:

- `max_turns = 1`: first SQL only; no repair opportunity.
- `max_turns = 2`: if turn 1 fails, feed execution feedback back to the model
  and allow one repair attempt.

If `infer_eval.py` does not support live multi-turn evaluation yet, extend it or
create `multi_turn_infer_eval.py`. Reuse existing code instead of duplicating
logic:

- Prompt building and model generation from `infer_eval.py`.
- Feedback prompt style from `agent_rollout.py`.
- SQL scoring and error categorization from `openrlhf_reward.py`.
- SQLite execution from `sandbox.py`.

Required CLI shape:

```bash
python3 multi_turn_infer_eval.py \
  --backend vllm \
  --dataset spider \
  --data-root data/spider \
  --model checkpoints/qwen2.5-coder-3b-logtir-grpo-ckpt/best_global_step150_hf \
  --max-turns 1 \
  --output logs/infer_eval_spider_grpo_best150_dev_turn1.json \
  --trajectories-out logs/trajectories_spider_grpo_best150_dev_turn1.jsonl

python3 multi_turn_infer_eval.py \
  --backend vllm \
  --dataset spider \
  --data-root data/spider \
  --model checkpoints/qwen2.5-coder-3b-logtir-grpo-ckpt/best_global_step150_hf \
  --max-turns 2 \
  --output logs/infer_eval_spider_grpo_best150_dev_turn2.json \
  --trajectories-out logs/trajectories_spider_grpo_best150_dev_turn2.jsonl
```

The summary JSON must include at least:

- `total`
- `matched`
- `accuracy`
- `accuracy_excluding_first_turn_timeout`
- `turn1_matched`
- `turn1_accuracy`
- `turn1_accuracy_excluding_timeout`
- `final_matched`
- `final_accuracy`
- `final_accuracy_excluding_first_turn_timeout`
- `rescued_by_turn2`
- `turn2_rescue_rate_all`
- `turn2_rescue_rate_among_turn1_failures`
- `turn2_rescue_rate_excluding_first_turn_timeout`
- `timeout_first_turn`
- `timeout_rescued_by_turn2`
- `non_timeout_rescued_by_turn2`
- `syntax_rescued_by_turn2`
- `execution_error_rescued_by_turn2`
- `wrong_result_rescued_by_turn2`

Each trajectory JSONL row should include enough evidence for later analysis:

- example id or index
- `db_id`
- question
- gold SQL
- per-turn model response
- parsed SQL action
- error category
- raw execution error, if any
- `exec_match`
- whether turn 2 rescued a turn 1 failure

Acceptance criteria:

- Full Spider dev finishes for both `max_turns = 1` and `max_turns = 2`.
- The two outputs use the same checkpoint, same data split, same decoding
  settings, and same SQL timeout.
- Timeout rescues must be reported separately and must not be counted as
  evidence of semantic self-correction.
- The final report must state both raw and timeout-excluded gains:

```text
same-checkpoint raw multi-turn gain =
accuracy(max_turns=2) - accuracy(max_turns=1)

same-checkpoint timeout-excluded multi-turn gain =
accuracy_excluding_first_turn_timeout(max_turns=2)
- accuracy_excluding_first_turn_timeout(max_turns=1)
```

This is the key number for the self-correction claim.

## Task 3: Timeout And Error Attribution

The current self-correction evidence may be inflated by timeout and formatting
retries. Make the ablation report separate these categories.

Add or verify the following fields in the multi-turn eval summary:

- turn 1 timeout count
- turn 1 invalid-format count
- turn 1 SQLite syntax error count
- turn 1 schema or hallucination error count
- turn 1 wrong-result count
- turn 2 rescue count by each first-turn error category
- final accuracy including timeouts
- final accuracy excluding first-turn timeout cases

Acceptance criteria:

- We can tell whether turn 2 mainly fixes semantic SQL errors or mostly recovers
  from timeout/format noise.
- Do not make a strong semantic self-correction claim unless non-timeout,
  non-format rescues are visible.

## Task 4: BIRD Transfer Evaluation

Run transfer evaluation for the same SFT baseline and GRPO best checkpoint if
BIRD data is available on the server.

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

Acceptance criteria:

- If BIRD data exists, record SFT vs GRPO transfer results.
- If BIRD data is missing, write down exactly what path or file is missing.

## Task 5: Optional Single-Turn GRPO Control Run

Only do this after Tasks 1-4.

If compute budget allows, run a separate GRPO control from the SFT checkpoint
with multi-turn disabled:

```bash
GRPO_MULTI_TURN=0 START_RAY=1 bash scripts/train_grpo_openrlhf.sh
```

Requirements:

- Use a separate run directory.
- Do not overwrite the existing best step 150 checkpoint.
- Evaluate the best single-turn-GRPO checkpoint on full Spider dev.
- Compare:
  - SFT baseline
  - single-turn GRPO
  - multi-turn GRPO best150

This is the strongest evidence for whether multi-turn training itself mattered,
but it is more expensive than the same-checkpoint ablation.

## Task 6: Update Experiment Records

After Tasks 2-4, update `docs/experiment-results.md` with:

- Spider `max_turns=1` result.
- Spider `max_turns=2` result.
- Same-checkpoint multi-turn gain.
- Rescue breakdown by first-turn error category.
- BIRD SFT result, if available.
- BIRD GRPO best150 result, if available.
- Any missing-data or timeout caveats.

Then commit and push:

```bash
git add docs/experiment-results.md tasks.md
git commit -m "Record next evaluation tasks and ablations"
git push origin main
```

## Resume-Safe Claim Template

Use this claim now:

```text
Trained a local Text-to-SQL self-correction agent with sandbox execution rewards;
GRPO improved Spider dev execution match from 70.0% to 71.8% over an SFT
cold-start baseline, with checkpoint selection based on internal eval.
```

Do not claim that multi-turn self-correction caused the gain until the
same-checkpoint `max_turns=1` vs `max_turns=2` ablation is complete.
