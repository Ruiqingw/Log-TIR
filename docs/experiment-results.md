# Experiment Results

This file records externally evaluated model results. Training-time loss-only logs
are not treated as benchmark results unless an explicit evaluator was run.

## Spider

| Date | Stage | Checkpoint | Evaluation artifact | Split | Total | Matched | Exec match | Notes |
| --- | --- | --- | --- | --- | ---: | ---: | ---: | --- |
| 2026-04-27 | SFT cold start | `checkpoints/qwen2.5-coder-3b-logtir-sft` | `remote-logs/infer_eval_spider_sft_dev.json` | dev | 1034 | 724 | 70.02% | SFT training used `--eval.steps -1`; this result came from a separate `infer_eval.py` run. |
| 2026-04-27 | GRPO best checkpoint | `checkpoints/qwen2.5-coder-3b-logtir-grpo-ckpt/best_global_step150_hf` | `remote-logs/infer_eval_spider_grpo_best150_dev.json` | dev | 1034 | 742 | 71.76% | Server artifact path: `logs/infer_eval_spider_grpo_best150_dev.json`. Selected from step 150; improves SFT by +1.74 pp. |
| 2026-04-28 | GRPO best checkpoint, first-turn-only ablation | `checkpoints/qwen2.5-coder-3b-logtir-grpo-ckpt/best_global_step150_hf` | `logs/infer_eval_spider_grpo_best150_dev_turn1.json` | dev | 1034 | 745 | 72.05% | Same checkpoint as the official GRPO candidate; `max_turns=1`, no repair opportunity. |
| 2026-04-28 | GRPO best checkpoint, two-turn ablation | `checkpoints/qwen2.5-coder-3b-logtir-grpo-ckpt/best_global_step150_hf` | `logs/infer_eval_spider_grpo_best150_dev_turn2.json` | dev | 1034 | 770 | 74.47% | Same checkpoint and split as the first-turn-only ablation; allows one repair from execution feedback. |

### Spider Main Comparison

| Date | Baseline | Candidate | Baseline exec match | Candidate exec match | Absolute delta | Error reduction | Notes |
| --- | --- | --- | ---: | ---: | ---: | ---: | --- |
| 2026-04-27 | SFT cold start | GRPO step 150 best checkpoint | 70.02% | 71.76% | +1.74 pp | 5.81% | 18 fewer Spider dev execution mismatches, from 310 down to 292. |

### Same-Checkpoint Multi-Turn Ablation

This ablation isolates inference-time self-correction by evaluating the selected
step-150 GRPO checkpoint on Spider dev with the same checkpoint and split under
`max_turns=1` and `max_turns=2`.

| Date | Artifact | Max turns | Total | Matched | Exec match | Notes |
| --- | --- | ---: | ---: | ---: | ---: | --- |
| 2026-04-28 | `logs/infer_eval_spider_grpo_best150_dev_turn1.json` | 1 | 1034 | 745 | 72.05% | First SQL only. |
| 2026-04-28 | `logs/infer_eval_spider_grpo_best150_dev_turn2.json` | 2 | 1034 | 770 | 74.47% | One repair attempt after execution feedback. |

Same-checkpoint multi-turn gain:

```text
accuracy(max_turns=2) - accuracy(max_turns=1)
= 0.7446808511 - 0.7205029014
= +2.42 pp
```

Key summary fields from `logs/infer_eval_spider_grpo_best150_dev_turn2.json`:

| Field | Value |
| --- | ---: |
| `turn1_exec_match` | 72.05% |
| `final_exec_match_with_2_turns` | 74.47% |
| `rescued_by_turn2` | 25 |
| `turn2_rescue_rate_all` | 2.42% |
| `turn2_rescue_rate_among_turn1_failures` | 8.65% |
| `timeout_rescue` | 0 |
| `syntax_error_rescue` | 0 |
| `wrong_result_rescue` | 13 |

Turn-2 rescue attribution by first-turn failure category:

| First-turn category | Turn-1 count | Rescued by turn 2 |
| --- | ---: | ---: |
| `timeout` | 4 | 0 |
| `invalid_format` | 0 | 0 |
| `syntax_error` | 0 | 0 |
| `schema_hallucination` | 62 | 8 |
| `wrong_result` | 181 | 13 |
| `empty_result` | 37 | 4 |
| `execution_error` | 5 | 0 |

Final accuracy excluding first-turn timeout cases was 74.76%. The observed
rescues are non-timeout and non-format rescues: 13 wrong-result fixes, 8
schema/hallucination fixes, and 4 empty-result fixes. Both ablation runs used
the same checkpoint, data split, backend, and SQL timeout; the first-turn
matched count was identical across the independent `max_turns=1` and
`max_turns=2` runs.

### BIRD Transfer

BIRD transfer evaluation was not run on 2026-04-28 because the expected server
data path `data/bird` does not exist (`test -e data/bird` returned exit code
1). No `logs/infer_eval_bird_sft_dev.json` or
`logs/infer_eval_bird_grpo_best150_dev.json` artifact was generated in this
run.

### GRPO Checkpoint Selection

Training internal eval is used for checkpoint selection only. The full Spider dev
`infer_eval.py` artifact above is the official benchmark result.

| Date | Candidate run | Global step | Metric | Value | Decision |
| --- | --- | ---: | --- | ---: | --- |
| 2026-04-27 | GRPO | 50 | `eval_spider_pass1` | 0.8892 | Improving |
| 2026-04-27 | GRPO | 100 | `eval_spider_pass1` | 0.9210 | Improving |
| 2026-04-27 | GRPO | 150 | `eval_spider_pass1` | 0.9315 | Best; selected as `best_global_step150_hf` |
| 2026-04-27 | GRPO | 200 | `eval_spider_pass1` | 0.9204 | Regressed from step 150; do not use final checkpoint by default |

### Smoke Checks

| Date | Stage | Evaluation artifact | Split/limit | Total | Matched | Exec match | Notes |
| --- | --- | --- | --- | ---: | ---: | ---: | --- |
| 2026-04-27 | SFT cold start | `remote-logs/infer_eval_spider_sft_10.json` | dev / 10 | 10 | 8 | 80.00% | Smoke test only; do not use as the main SFT result. |
