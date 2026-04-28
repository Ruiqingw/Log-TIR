# Log-TIR

Log-TIR is a fully local Text-to-SQL self-correction agent built on
`Qwen2.5-Coder-3B-Instruct`. The system trains a short SFT cold start, applies
GRPO with execution-based rewards, and lets the agent repair SQL after sandbox
feedback from SQLite execution.

The core loop is local: generate SQL, execute it in a read-only SQLite sandbox,
classify the result or traceback, and retry within a fixed turn budget. No
external inference API is required for training or evaluation.

## Main Results

Primary benchmark: Spider dev execution match.

| Stage | Evaluation mode | Matched | Exec match |
| --- | --- | ---: | ---: |
| Raw `Qwen2.5-Coder-3B-Instruct` | single pass | 117 / 1034 | 11.32% |
| SFT cold start | single pass | 724 / 1034 | 70.02% |
| Single-turn GRPO control | single pass | 728 / 1034 | 70.41% |
| Multi-turn GRPO best checkpoint | single pass | 742 / 1034 | 71.76% |
| Multi-turn GRPO best checkpoint | 2-turn execution feedback | 771 / 1034 | 74.56% |
| Multi-turn GRPO best checkpoint | 4-turn execution feedback | 778 / 1034 | 75.24% |

Key deltas:

- SFT cold start improves the raw base model by `+58.70 pp` on Spider dev.
- Multi-turn GRPO improves the SFT single-pass baseline by `+1.74 pp`.
- Execution-feedback self-correction improves the selected GRPO checkpoint from
  `71.76%` single-pass to `74.56%` at two turns and `75.24%` at four turns.
- Under the same four-turn evaluator, multi-turn GRPO beats the single-turn
  GRPO control by `+2.71 pp`.

Secondary benchmark: BIRD dev transfer. BIRD remains timeout-limited in the
current evaluator, so it is reported as directional transfer evidence rather
than the headline result. Under timeout-30 saved-response re-evaluation, raw
base reaches `6.26%`, SFT reaches `23.79%`, and multi-turn GRPO reaches
`24.45%`.

Full experiment details and caveats are in
[`docs/experiment-results.md`](docs/experiment-results.md).

## Technical Pieces

- `sandbox.py`: read-only SQLite execution sandbox with timeout handling.
- `eval.py`: execution-match evaluator with row normalization.
- `sft_data.py`: Spider-to-SFT cold-start data generation.
- `rl_data.py`: GRPO prompt/label generation.
- `openrlhf_reward.py`: local reward function using format, no-error, and
  execution-match signals.
- `multi_turn_infer_eval.py`: inference-time self-correction evaluator with
  arbitrary turn budgets.

## Claim Boundary

The strongest claim is on Spider: local execution feedback and GRPO produce a
measured self-correcting Text-to-SQL agent, with clear gains over raw prompting,
SFT, and a single-turn GRPO control. BIRD should be described as a hard transfer
setting where SFT provides most of the lift and multi-turn GRPO is directionally
best but only modestly better.
