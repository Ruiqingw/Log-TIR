# Log-TIR Remaining Tasks

This file is the current server-side handoff. Earlier turn-sweep,
single-turn-control, and BIRD-transfer tasks are complete; do not rerun them
unless a new experiment explicitly asks for it.

Run commands from the repository root.

## Completed Results Snapshot

Official results are recorded in `docs/experiment-results.md`.

Spider:

- SFT cold start: `724 / 1034 = 70.02%`.
- Multi-turn GRPO best150, official `infer_eval.py`: `742 / 1034 = 71.76%`.
- Multi-turn GRPO best150, inference-time sweep:
  - `max_turns=1`: `745 / 1034 = 72.05%`
  - `max_turns=2`: `771 / 1034 = 74.56%`
  - `max_turns=3`: `772 / 1034 = 74.66%`
  - `max_turns=4`: `778 / 1034 = 75.24%`
- Single-turn GRPO control, official `infer_eval.py`: `728 / 1034 = 70.41%`.
- Single-turn GRPO control, inference-time sweep:
  - `max_turns=1`: `729 / 1034 = 70.50%`
  - `max_turns=2`: `753 / 1034 = 72.82%`
  - `max_turns=3`: `749 / 1034 = 72.44%`
  - `max_turns=4`: `750 / 1034 = 72.53%`

BIRD transfer:

- Timeout-3 original eval:
  - SFT: `357 / 1534 = 23.27%`
  - Multi-turn GRPO best150: `366 / 1534 = 23.86%`
  - Single-turn GRPO control: `354 / 1534 = 23.08%`
- Timeout-30 saved-response re-eval:
  - SFT: `365 / 1534 = 23.79%`
  - Multi-turn GRPO best150: `375 / 1534 = 24.45%`
  - Single-turn GRPO control: `363 / 1534 = 23.66%`

BIRD remains timeout-limited. Standalone gold-SQL validation reached
`1350 / 1534 = 88.01%` at timeout 30, but still had 184 gold timeout failures;
the saved-response re-eval cache pass observed 216 gold timeouts under
`workers=8`. Treat BIRD as directional transfer evidence, not the clean primary
benchmark.

## Task 1: Pre-SFT Base Model Baseline

Goal: measure the raw `Qwen2.5-Coder-3B-Instruct` model before the SFT cold
start. This gives a clean baseline for the claim that SFT is a necessary cold
start before GRPO.

This is inference only. Do not train, do not overwrite any SFT or GRPO
checkpoint directory, and do not change decoding settings unless vLLM requires
a launch-specific compatibility flag.

Recommended launch:

```bash
git pull --rebase origin main

export BASE_MODEL="${BASE_MODEL:-Qwen/Qwen2.5-Coder-3B-Instruct}"

python3 infer_eval.py \
  --backend vllm \
  --dataset spider \
  --data-root data/spider \
  --model "$BASE_MODEL" \
  --output logs/infer_eval_spider_base_qwen25_coder3b_dev.json

python3 infer_eval.py \
  --backend vllm \
  --dataset bird \
  --data-root data/bird \
  --model "$BASE_MODEL" \
  --output logs/infer_eval_bird_base_qwen25_coder3b_dev.json
```

If the Hugging Face model name cannot be loaded because the server is offline,
set `BASE_MODEL` to the local cache path for the same base checkpoint and rerun
the two commands above. Do not substitute the SFT checkpoint.

For BIRD, also run the same saved-response timeout-30 re-evaluation used for
the SFT and GRPO BIRD reports:

```bash
python3 scripts/reevaluate_saved_infer_eval.py \
  --dataset bird \
  --data-root data/bird \
  --timeout 30 \
  --workers 8 \
  --input-output \
    logs/infer_eval_bird_base_qwen25_coder3b_dev.json \
    logs/infer_eval_bird_base_qwen25_coder3b_dev_timeout30.json
```

Acceptance criteria:

- `logs/infer_eval_spider_base_qwen25_coder3b_dev.json` exists and reports
  `total`, `matched`, and `accuracy`.
- `logs/infer_eval_bird_base_qwen25_coder3b_dev.json` exists and reports
  `total`, `matched`, and `accuracy`.
- `logs/infer_eval_bird_base_qwen25_coder3b_dev_timeout30.json` exists and
  reports `total`, `matched`, `accuracy`, `gold_failures`, and
  `gold_timeout_count`.
- Record the three summary JSON outputs in the server note or terminal log so
  the Mac control plane can update `docs/experiment-results.md`.

## Task 2: Package Results After Base Baseline

Owner: Mac control plane / local Codex after the server baseline artifacts are
available.

Use the base-model baseline to package the project around a defensible story:

- Base model -> SFT cold start -> multi-turn GRPO -> inference-time
  self-correction.
- Primary claim should be Spider, because BIRD is timeout-limited.
- Keep BIRD as transfer evidence only: multi-turn GRPO is directionally best,
  but the absolute BIRD number is not the headline.
- Separate three effects in the writeup:
  - SFT cold start versus raw base model.
  - GRPO reward optimization versus SFT.
  - Multi-turn/self-correction benefit versus single-turn control under matched
    inference budgets.

Expected packaging updates after the base baseline is synced back:

- Add base-model rows to `docs/experiment-results.md`.
- Add a concise result table and claim boundary to `README.md` if the README is
  ready for project presentation.
- Draft resume bullets that emphasize local sandbox execution, GRPO reward
  shaping, multi-turn feedback, and measured Spider gains.
- Keep timeout caveats explicit for BIRD and do not overstate transfer.

## Current Safe Claim Before Base Baseline

```text
On Spider dev, the selected multi-turn GRPO checkpoint improves from the SFT
cold-start baseline's 70.02% execution match to 71.76% under the standalone
single-pass evaluator. With inference-time execution-feedback repair on the same
checkpoint, accuracy rises to 74.56% at two turns and 75.24% at four turns.
Compared with the single-turn GRPO control under the same four-turn evaluator,
multi-turn GRPO is higher by 2.71 percentage points.
```

After Task 1, update this claim with the raw base-model baseline so the
cold-start story is complete.
