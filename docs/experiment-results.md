# Experiment Results

This file records externally evaluated model results. Training-time loss-only logs
are not treated as benchmark results unless an explicit evaluator was run.

## Spider

| Date | Stage | Checkpoint | Evaluation artifact | Split | Total | Matched | Exec match | Notes |
| --- | --- | --- | --- | --- | ---: | ---: | ---: | --- |
| 2026-04-27 | SFT cold start | `checkpoints/qwen2.5-coder-3b-logtir-sft` | `remote-logs/infer_eval_spider_sft_dev.json` | dev | 1034 | 724 | 70.02% | SFT training used `--eval.steps -1`; this result came from a separate `infer_eval.py` run. |
| 2026-04-27 | GRPO best checkpoint | `checkpoints/qwen2.5-coder-3b-logtir-grpo-ckpt/best_global_step150_hf` | `remote-logs/infer_eval_spider_grpo_best150_dev.json` | dev | 1034 | 742 | 71.76% | Server artifact path: `logs/infer_eval_spider_grpo_best150_dev.json`. Selected from step 150; improves SFT by +1.74 pp. |

### Spider Main Comparison

| Date | Baseline | Candidate | Baseline exec match | Candidate exec match | Absolute delta | Error reduction | Notes |
| --- | --- | --- | ---: | ---: | ---: | ---: | --- |
| 2026-04-27 | SFT cold start | GRPO step 150 best checkpoint | 70.02% | 71.76% | +1.74 pp | 5.81% | 18 fewer Spider dev execution mismatches, from 310 down to 292. |

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
