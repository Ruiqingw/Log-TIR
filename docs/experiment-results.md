# Experiment Results

This file records externally evaluated model results. Training-time loss-only logs
are not treated as benchmark results unless an explicit evaluator was run.

## Spider

| Date | Stage | Checkpoint | Evaluation artifact | Split | Total | Matched | Exec match | Notes |
| --- | --- | --- | --- | --- | ---: | ---: | ---: | --- |
| 2026-04-27 | SFT cold start | `checkpoints/qwen2.5-coder-3b-logtir-sft` | `remote-logs/infer_eval_spider_sft_dev.json` | dev | 1034 | 724 | 70.02% | SFT training used `--eval.steps -1`; this result came from a separate `infer_eval.py` run. |
| 2026-04-27 | Multi-turn GRPO best checkpoint | `checkpoints/qwen2.5-coder-3b-logtir-grpo-ckpt/best_global_step150_hf` | `remote-logs/infer_eval_spider_grpo_best150_dev.json` | dev | 1034 | 742 | 71.76% | Server artifact path: `logs/infer_eval_spider_grpo_best150_dev.json`. Selected from step 150; improves SFT by +1.74 pp. |
| 2026-04-28 | Multi-turn GRPO best checkpoint, first-turn-only ablation | `checkpoints/qwen2.5-coder-3b-logtir-grpo-ckpt/best_global_step150_hf` | `logs/infer_eval_spider_grpo_best150_dev_turn1.json` | dev | 1034 | 745 | 72.05% | Same checkpoint as the official GRPO candidate; `max_turns=1`, no repair opportunity. |
| 2026-04-28 | Multi-turn GRPO best checkpoint, two-turn ablation | `checkpoints/qwen2.5-coder-3b-logtir-grpo-ckpt/best_global_step150_hf` | `logs/infer_eval_spider_grpo_best150_dev_turn2.json` | dev | 1034 | 771 | 74.56% | Refreshed arbitrary-turn evaluator run; allows one repair from execution feedback. |
| 2026-04-28 | Multi-turn GRPO best checkpoint, three-turn sweep | `checkpoints/qwen2.5-coder-3b-logtir-grpo-ckpt/best_global_step150_hf` | `logs/infer_eval_spider_grpo_best150_dev_turn3.json` | dev | 1034 | 772 | 74.66% | Two repair attempts; third turn adds a small gain. |
| 2026-04-28 | Multi-turn GRPO best checkpoint, four-turn sweep | `checkpoints/qwen2.5-coder-3b-logtir-grpo-ckpt/best_global_step150_hf` | `logs/infer_eval_spider_grpo_best150_dev_turn4.json` | dev | 1034 | 778 | 75.24% | Three repair attempts; no timeout rescues. |
| 2026-04-28 | Single-turn GRPO control best checkpoint | `checkpoints/qwen2.5-coder-3b-logtir-grpo-singleturn-timeout10-ckpt/best_global_step150_hf` | `logs/infer_eval_spider_grpo_singleturn_best_dev.json` | dev | 1034 | 728 | 70.41% | `GRPO_MULTI_TURN=0`; selected by internal step-150 eval. Full-dev gain over SFT is small: +0.39 pp. |
| 2026-04-28 | Single-turn GRPO control, first-turn-only ablation | `checkpoints/qwen2.5-coder-3b-logtir-grpo-singleturn-timeout10-ckpt/best_global_step150_hf` | `logs/infer_eval_spider_grpo_singleturn_best_dev_turn1.json` | dev | 1034 | 729 | 70.50% | `max_turns=1`, no repair opportunity. |
| 2026-04-28 | Single-turn GRPO control, two-turn ablation | `checkpoints/qwen2.5-coder-3b-logtir-grpo-singleturn-timeout10-ckpt/best_global_step150_hf` | `logs/infer_eval_spider_grpo_singleturn_best_dev_turn2.json` | dev | 1034 | 753 | 72.82% | `max_turns=2`; one repair attempt after execution feedback. |
| 2026-04-28 | Single-turn GRPO control, three-turn sweep | `checkpoints/qwen2.5-coder-3b-logtir-grpo-singleturn-timeout10-ckpt/best_global_step150_hf` | `logs/infer_eval_spider_grpo_singleturn_best_dev_turn3.json` | dev | 1034 | 749 | 72.44% | Independent run; use its `matched_by_turn` for within-run marginal gains. |
| 2026-04-28 | Single-turn GRPO control, four-turn sweep | `checkpoints/qwen2.5-coder-3b-logtir-grpo-singleturn-timeout10-ckpt/best_global_step150_hf` | `logs/infer_eval_spider_grpo_singleturn_best_dev_turn4.json` | dev | 1034 | 750 | 72.53% | Four-turn sweep saturates after turn 3 in this run. |

### Spider Main Comparison

| Date | Baseline | Candidate | Baseline exec match | Candidate exec match | Absolute delta | Error reduction | Notes |
| --- | --- | --- | ---: | ---: | ---: | ---: | --- |
| 2026-04-27 | SFT cold start | Multi-turn GRPO step-150 best checkpoint | 70.02% | 71.76% | +1.74 pp | 5.81% | 18 fewer Spider dev execution mismatches, from 310 down to 292. |
| 2026-04-28 | SFT cold start | Single-turn GRPO control step-150 best checkpoint | 70.02% | 70.41% | +0.39 pp | 1.29% | 4 fewer Spider dev execution mismatches, from 310 down to 306. |
| 2026-04-28 | Single-turn GRPO control, `max_turns=1` | Multi-turn GRPO best150, `max_turns=1` | 70.50% | 72.05% | +1.55 pp | 5.25% | Same split and evaluator family; independent vLLM generations can vary slightly. |
| 2026-04-28 | Single-turn GRPO control, `max_turns=2` | Multi-turn GRPO best150, `max_turns=2` | 72.82% | 74.56% | +1.74 pp | 6.41% | 18 fewer execution mismatches under the refreshed two-turn evaluator, from 281 down to 263. |
| 2026-04-28 | Single-turn GRPO control, `max_turns=3` | Multi-turn GRPO best150, `max_turns=3` | 72.44% | 74.66% | +2.22 pp | 8.07% | 23 fewer execution mismatches, from 285 down to 262. |
| 2026-04-28 | Single-turn GRPO control, `max_turns=4` | Multi-turn GRPO best150, `max_turns=4` | 72.53% | 75.24% | +2.71 pp | 9.86% | 28 fewer execution mismatches, from 284 down to 256. |

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
| `logs/infer_eval_spider_grpo_singleturn_best_dev_turn1.json` | `multi_turn_infer_eval.py` | 1 | 1034 | 729 | 70.50% | 70.91% | 0 |
| `logs/infer_eval_spider_grpo_singleturn_best_dev_turn2.json` | `multi_turn_infer_eval.py` | 2 | 1034 | 753 | 72.82% | 73.18% | 20 |
| `logs/infer_eval_spider_grpo_singleturn_best_dev_turn3.json` | `multi_turn_infer_eval.py` | 3 | 1034 | 749 | 72.44% | 72.72% | 20 |
| `logs/infer_eval_spider_grpo_singleturn_best_dev_turn4.json` | `multi_turn_infer_eval.py` | 4 | 1034 | 750 | 72.53% | 72.96% | 21 |

Each `max_turns` artifact is an independent vLLM generation run, so first-turn
counts can drift by a few examples. For marginal extra-turn analysis, prefer
`matched_by_turn` and `marginal_accuracy_gain_by_turn` within the same artifact.

Key summary fields from
`logs/infer_eval_spider_grpo_singleturn_best_dev_turn4.json`:

| Field | Value |
| --- | ---: |
| `turn1_exec_match` | 70.21% |
| `final_accuracy` | 72.53% |
| `rescued_by_turn2` | 21 |
| `rescued_by_turn3` | 3 |
| `rescued_by_turn4` | 0 |
| `marginal_accuracy_gain_by_turn.turn2` | 2.03% |
| `marginal_accuracy_gain_by_turn.turn3` | 0.29% |
| `marginal_accuracy_gain_by_turn.turn4` | 0.00% |
| `accuracy_excluding_first_turn_timeout` | 72.96% |
| `final_accuracy_excluding_first_turn_timeout` | 72.96% |
| `timeout_rescue_by_turn.turn2` | 0 |
| `timeout_rescue_by_turn.turn3` | 0 |
| `timeout_rescue_by_turn.turn4` | 0 |

Turn-2 rescue attribution for the single-turn control:

| First-turn category | Turn-1 count | Rescued by turn 2 |
| --- | ---: | ---: |
| `timeout` | 6 | 0 |
| `invalid_format` | 0 | 0 |
| `syntax_error` | 1 | 0 |
| `schema_hallucination` | 72 | 5 |
| `wrong_result` | 188 | 13 |
| `empty_result` | 39 | 3 |
| `execution_error` | 2 | 0 |

Cumulative rescue attribution by first-turn category in the four-turn
single-turn-control run:

| First-turn category | Rescued by turns 2-4 |
| --- | ---: |
| `empty_result` | 3 |
| `execution_error` | 1 |
| `schema_hallucination` | 5 |
| `wrong_result` | 15 |

Interpretation: the single-turn GRPO control did not match the selected
multi-turn GRPO checkpoint on Spider. Under the refreshed four-turn evaluator,
the multi-turn GRPO checkpoint reached 75.24% while the single-turn control
reached 72.53%. This is stronger evidence that the multi-turn training/feedback
setup adds value beyond ordinary single-turn GRPO, though the BIRD transfer gain
below is small.

### Multi-Turn GRPO Inference-Time Turn Sweep

This sweep isolates inference-time self-correction by evaluating the selected
step-150 GRPO checkpoint on Spider dev with the same checkpoint and split under
`max_turns=1,2,3,4`. Each `max_turns` artifact is an independent vLLM generation
run, so first-turn counts can drift slightly; use `matched_by_turn` inside the
same artifact for marginal turn analysis.

| Date | Artifact | Max turns | Total | Matched | Exec match | `matched_by_turn` | Notes |
| --- | --- | ---: | ---: | ---: | ---: | --- | --- |
| 2026-04-28 | `logs/infer_eval_spider_grpo_best150_dev_turn1.json` | 1 | 1034 | 745 | 72.05% | `{"turn1": 745}` | First SQL only. |
| 2026-04-28 | `logs/infer_eval_spider_grpo_best150_dev_turn2.json` | 2 | 1034 | 771 | 74.56% | `{"turn1": 746, "turn2": 771}` | One repair attempt after execution feedback. |
| 2026-04-28 | `logs/infer_eval_spider_grpo_best150_dev_turn3.json` | 3 | 1034 | 772 | 74.66% | `{"turn1": 744, "turn2": 769, "turn3": 772}` | Two repair attempts. |
| 2026-04-28 | `logs/infer_eval_spider_grpo_best150_dev_turn4.json` | 4 | 1034 | 778 | 75.24% | `{"turn1": 744, "turn2": 771, "turn3": 775, "turn4": 778}` | Three repair attempts; no timeout rescues. |

Marginal gains inside
`logs/infer_eval_spider_grpo_best150_dev_turn4.json`:

```text
turn 1 -> 2 gain: +27 examples, +2.61 pp
turn 2 -> 3 gain: +4 examples, +0.39 pp
turn 3 -> 4 gain: +3 examples, +0.29 pp
```

Key summary fields from `logs/infer_eval_spider_grpo_best150_dev_turn4.json`:

| Field | Value |
| --- | ---: |
| `turn1_exec_match` | 71.95% |
| `final_accuracy` | 75.24% |
| `rescued_by_turn2` | 27 |
| `rescued_by_turn3` | 4 |
| `rescued_by_turn4` | 3 |
| `marginal_accuracy_gain_by_turn.turn2` | 2.61% |
| `marginal_accuracy_gain_by_turn.turn3` | 0.39% |
| `marginal_accuracy_gain_by_turn.turn4` | 0.29% |
| `accuracy_excluding_first_turn_timeout` | 75.53% |
| `final_accuracy_excluding_first_turn_timeout` | 75.53% |
| `timeout_rescue_by_turn.turn2` | 0 |
| `timeout_rescue_by_turn.turn3` | 0 |
| `timeout_rescue_by_turn.turn4` | 0 |
| `non_timeout_rescue_by_turn.turn2` | 27 |
| `non_timeout_rescue_by_turn.turn3` | 4 |
| `non_timeout_rescue_by_turn.turn4` | 3 |

Cumulative rescue attribution by first-turn category in the four-turn
multi-turn-GRPO run:

| First-turn category | Turn-1 count | Rescued by turns 2-4 |
| --- | ---: | ---: |
| `timeout` | 4 | 0 |
| `invalid_format` | 0 | 0 |
| `syntax_error` | 0 | 0 |
| `schema_hallucination` | 62 | 8 |
| `wrong_result` | 182 | 19 |
| `empty_result` | 37 | 5 |
| `execution_error` | 5 | 2 |

Interpretation: gains still saturate strongly after turn 2, but turns 3 and 4
add 7 additional non-timeout rescues in the same four-turn run. The strongest
cost-effective project claim should stay focused on two-turn self-correction,
with optional extra inference turns reported as remaining headroom on harder
examples. Timeout rescues are zero, so these gains should be counted as
semantic/error self-correction rather than timeout recovery.

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
