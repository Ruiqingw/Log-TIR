# Experiment Results

This file records externally evaluated model results. Training-time loss-only logs
are not treated as benchmark results unless an explicit evaluator was run.

## Spider

| Date | Stage | Checkpoint | Evaluation artifact | Split | Total | Matched | Exec match | Notes |
| --- | --- | --- | --- | --- | ---: | ---: | ---: | --- |
| 2026-04-27 | SFT cold start | `checkpoints/qwen2.5-coder-3b-logtir-sft` | `remote-logs/infer_eval_spider_sft_dev.json` | dev | 1034 | 724 | 70.02% | SFT training used `--eval.steps -1`; this result came from a separate `infer_eval.py` run. |
| 2026-04-27 | Multi-turn GRPO best checkpoint | `checkpoints/qwen2.5-coder-3b-logtir-grpo-ckpt/best_global_step150_hf` | `remote-logs/infer_eval_spider_grpo_best150_dev.json` | dev | 1034 | 742 | 71.76% | Server artifact path: `logs/infer_eval_spider_grpo_best150_dev.json`. Selected from step 150; improves SFT by +1.74 pp. |
| 2026-04-28 | Multi-turn GRPO best checkpoint, first-turn-only ablation | `checkpoints/qwen2.5-coder-3b-logtir-grpo-ckpt/best_global_step150_hf` | `logs/infer_eval_spider_grpo_best150_dev_turn1.json` | dev | 1034 | 745 | 72.05% | Same checkpoint as the official GRPO candidate; `max_turns=1`, no repair opportunity. |
| 2026-04-28 | Multi-turn GRPO best checkpoint, two-turn ablation | `checkpoints/qwen2.5-coder-3b-logtir-grpo-ckpt/best_global_step150_hf` | `logs/infer_eval_spider_grpo_best150_dev_turn2.json` | dev | 1034 | 770 | 74.47% | Same checkpoint and split as the first-turn-only ablation; allows one repair from execution feedback. |
| 2026-04-28 | Single-turn GRPO control best checkpoint | `checkpoints/qwen2.5-coder-3b-logtir-grpo-singleturn-timeout10-ckpt/best_global_step150_hf` | `logs/infer_eval_spider_grpo_singleturn_best_dev.json` | dev | 1034 | 728 | 70.41% | `GRPO_MULTI_TURN=0`; selected by internal step-150 eval. Full-dev gain over SFT is small: +0.39 pp. |
| 2026-04-28 | Single-turn GRPO control, first-turn-only ablation | `checkpoints/qwen2.5-coder-3b-logtir-grpo-singleturn-timeout10-ckpt/best_global_step150_hf` | `logs/infer_eval_spider_grpo_singleturn_best_dev_turn1.json` | dev | 1034 | 730 | 70.60% | `max_turns=1`, no repair opportunity. |
| 2026-04-28 | Single-turn GRPO control, two-turn ablation | `checkpoints/qwen2.5-coder-3b-logtir-grpo-singleturn-timeout10-ckpt/best_global_step150_hf` | `logs/infer_eval_spider_grpo_singleturn_best_dev_turn2.json` | dev | 1034 | 747 | 72.24% | `max_turns=2`; one repair attempt after execution feedback. |

### Spider Main Comparison

| Date | Baseline | Candidate | Baseline exec match | Candidate exec match | Absolute delta | Error reduction | Notes |
| --- | --- | --- | ---: | ---: | ---: | ---: | --- |
| 2026-04-27 | SFT cold start | Multi-turn GRPO step-150 best checkpoint | 70.02% | 71.76% | +1.74 pp | 5.81% | 18 fewer Spider dev execution mismatches, from 310 down to 292. |
| 2026-04-28 | SFT cold start | Single-turn GRPO control step-150 best checkpoint | 70.02% | 70.41% | +0.39 pp | 1.29% | 4 fewer Spider dev execution mismatches, from 310 down to 306. |
| 2026-04-28 | Single-turn GRPO control, `max_turns=1` | Multi-turn GRPO best150, `max_turns=1` | 70.60% | 72.05% | +1.45 pp | 4.92% | Same split and evaluator family; independent vLLM generations can vary slightly. |
| 2026-04-28 | Single-turn GRPO control, `max_turns=2` | Multi-turn GRPO best150, `max_turns=2` | 72.24% | 74.47% | +2.22 pp | 8.01% | 23 fewer execution mismatches under the two-turn evaluator, from 287 down to 264. |

### Single-Turn GRPO Control

This run tested whether ordinary single-turn GRPO reward optimization explains
the multi-turn GRPO result.

Run configuration:

| Field | Value |
| --- | --- |
| Run directory | `logs/run_20260428_081306_grpo` |
| Training mode | `GRPO_MULTI_TURN=0`; `agent_func_path=None` |
| Actor start checkpoint | `checkpoints/qwen2.5-coder-3b-logtir-sft` |
| Prompt data | `data/rl/spider_grpo_train_2000_singleturn.jsonl` |
| Eval data | `data/rl/spider_grpo_dev_100_singleturn.jsonl` |
| Output directory | `checkpoints/qwen2.5-coder-3b-logtir-grpo-singleturn-timeout10` |
| Checkpoint directory | `checkpoints/qwen2.5-coder-3b-logtir-grpo-singleturn-timeout10-ckpt` |
| GPU layout | 4 GPUs total: actor/ref colocated on 3 GPUs plus 1 vLLM GPU |
| Batch / group | train batch 6, rollout batch 6, `n_samples_per_prompt=4` |
| Rollout | temperature 0.3, top-p 1.0, max new tokens 512 |
| Reward timeout | `LOGTIR_REWARD_TIMEOUT=10` |
| Tracking | wandb offline run `grpo-singleturn-control-timeout10` |

The reward timeout was increased to 10 seconds because the earlier 3 second
reward-time SQL timeout produced false timeouts under Ray reward concurrency.
Checkpoint saves were very slow and left GPU utilization at 0% during save; that
was checkpoint I/O, not a stalled rollout.

Checkpoint-selection signal:

| Global step | `eval_spider_pass1` | Response length mean | Truncated rate | Decision |
| ---: | ---: | ---: | ---: | --- |
| 50 | 0.8745 | 71.80 | 0.50% | New best; saved `best_global_step50_hf`. |
| 100 | 0.8760 | 69.04 | 0.50% | New best; saved `best_global_step100_hf`. |
| 150 | 0.9485 | 68.60 | 1.00% | Best selected checkpoint: `best_global_step150_hf`. |
| 200 | 0.9130 | 61.98 | 0.25% | Regressed from step 150; training was stopped after this eval. |

Full Spider dev evaluations for the selected single-turn control checkpoint:

| Artifact | Evaluator | Max turns | Total | Matched | Exec match | Timeout-excluded exec match | Turn-2 rescues |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |
| `logs/infer_eval_spider_grpo_singleturn_best_dev.json` | `infer_eval.py` | 1 | 1034 | 728 | 70.41% | n/a | n/a |
| `logs/infer_eval_spider_grpo_singleturn_best_dev_turn1.json` | `multi_turn_infer_eval.py` | 1 | 1034 | 730 | 70.60% | 70.94% | 0 |
| `logs/infer_eval_spider_grpo_singleturn_best_dev_turn2.json` | `multi_turn_infer_eval.py` | 2 | 1034 | 747 | 72.24% | 72.67% | 22 |

Key summary fields from
`logs/infer_eval_spider_grpo_singleturn_best_dev_turn2.json`:

| Field | Value |
| --- | ---: |
| `turn1_exec_match` | 70.12% |
| `final_exec_match_with_2_turns` | 72.24% |
| `rescued_by_turn2` | 22 |
| `turn2_rescue_rate_all` | 2.13% |
| `turn2_rescue_rate_among_turn1_failures` | 7.12% |
| `accuracy_excluding_first_turn_timeout` | 72.67% |
| `final_accuracy_excluding_first_turn_timeout` | 72.67% |
| `turn2_rescue_rate_excluding_first_turn_timeout` | 2.14% |
| `timeout_rescue` | 0 |
| `syntax_error_rescue` | 0 |
| `wrong_result_rescue` | 13 |

Turn-2 rescue attribution for the single-turn control:

| First-turn category | Turn-1 count | Rescued by turn 2 |
| --- | ---: | ---: |
| `timeout` | 6 | 0 |
| `invalid_format` | 0 | 0 |
| `syntax_error` | 1 | 0 |
| `schema_hallucination` | 73 | 6 |
| `wrong_result` | 186 | 13 |
| `empty_result` | 41 | 3 |
| `execution_error` | 2 | 0 |

Interpretation: the single-turn GRPO control did not match the selected
multi-turn GRPO checkpoint on Spider. Under the same two-turn evaluator, the
multi-turn GRPO checkpoint reached 74.47% while the single-turn control reached
72.24%. This is stronger evidence that the multi-turn training/feedback setup
adds value beyond ordinary single-turn GRPO, though the BIRD transfer gain below
is small.

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

Newer `multi_turn_infer_eval.py` summaries keep the old plural
`final_accuracy_excluding_first_turn_timeouts` field and also emit stable alias
fields: `accuracy_excluding_first_turn_timeout`,
`turn1_accuracy_excluding_timeout`,
`final_accuracy_excluding_first_turn_timeout`, and
`turn2_rescue_rate_excluding_first_turn_timeout`.

## BIRD Transfer

BIRD dev was downloaded and unpacked locally on the server under
`data/bird/dev_20240627`. A BIRD gold-SQL smoke check passed on 10 examples
before model evaluation.

| Date | Stage | Checkpoint | Evaluation artifact | Split | Total | Matched | Exec match | Notes |
| --- | --- | --- | --- | --- | ---: | ---: | ---: | --- |
| 2026-04-28 | SFT cold start | `checkpoints/qwen2.5-coder-3b-logtir-sft` | `logs/infer_eval_bird_sft_dev.json` | dev | 1534 | 357 | 23.27% | Transfer baseline. |
| 2026-04-28 | Multi-turn GRPO best checkpoint | `checkpoints/qwen2.5-coder-3b-logtir-grpo-ckpt/best_global_step150_hf` | `logs/infer_eval_bird_grpo_best150_dev.json` | dev | 1534 | 366 | 23.86% | +0.59 pp over SFT, 9 fewer mismatches. |
| 2026-04-28 | Single-turn GRPO control best checkpoint | `checkpoints/qwen2.5-coder-3b-logtir-grpo-singleturn-timeout10-ckpt/best_global_step150_hf` | `logs/infer_eval_bird_grpo_singleturn_best_dev.json` | dev | 1534 | 354 | 23.08% | -0.19 pp vs SFT and -0.78 pp vs multi-turn GRPO. |

The BIRD transfer signal is weak but directionally consistent with Spider:
multi-turn GRPO is best among the three local checkpoints, while the
single-turn GRPO control does not improve transfer.

## GRPO Checkpoint Selection

Training internal eval is used for checkpoint selection only. The full Spider dev
`infer_eval.py` artifacts above are the official benchmark results.

| Date | Candidate run | Global step | Metric | Value | Decision |
| --- | --- | ---: | --- | ---: | --- |
| 2026-04-27 | Multi-turn GRPO | 50 | `eval_spider_pass1` | 0.8892 | Improving |
| 2026-04-27 | Multi-turn GRPO | 100 | `eval_spider_pass1` | 0.9210 | Improving |
| 2026-04-27 | Multi-turn GRPO | 150 | `eval_spider_pass1` | 0.9315 | Best; selected as `best_global_step150_hf` |
| 2026-04-27 | Multi-turn GRPO | 200 | `eval_spider_pass1` | 0.9204 | Regressed from step 150; do not use final checkpoint by default |
| 2026-04-28 | Single-turn GRPO control | 50 | `eval_spider_pass1` | 0.8745 | Improving |
| 2026-04-28 | Single-turn GRPO control | 100 | `eval_spider_pass1` | 0.8760 | Improving |
| 2026-04-28 | Single-turn GRPO control | 150 | `eval_spider_pass1` | 0.9485 | Best; selected as `best_global_step150_hf` |
| 2026-04-28 | Single-turn GRPO control | 200 | `eval_spider_pass1` | 0.9130 | Regressed from step 150; training stopped after this eval |

## Smoke Checks

| Date | Stage | Evaluation artifact | Split/limit | Total | Matched | Exec match | Notes |
| --- | --- | --- | --- | ---: | ---: | ---: | --- |
| 2026-04-27 | SFT cold start | `remote-logs/infer_eval_spider_sft_10.json` | Spider dev / 10 | 10 | 8 | 80.00% | Smoke test only; do not use as the main SFT result. |
| 2026-04-28 | BIRD gold SQL | command output only | BIRD dev / 10 | 10 | 10 | 100.00% | Verified local BIRD adapter and database paths before transfer eval. |
