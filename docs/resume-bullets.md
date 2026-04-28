# Resume Bullets

Use the Spider numbers as the main evidence. Keep BIRD as transfer/caveat
context, not the headline.

## Strong Version

- Built a fully local Text-to-SQL self-correction agent on
  `Qwen2.5-Coder-3B-Instruct`, using read-only SQLite sandbox execution as the
  reward/evaluation signal and avoiding external inference APIs.
- Created a staged SFT-to-GRPO training pipeline with format, no-error, and
  execution-match rewards; improved Spider dev execution match from `11.32%`
  raw prompting to `70.02%` after SFT and `71.76%` after GRPO.
- Added multi-turn execution-feedback repair and a single-turn GRPO control;
  reached `75.24%` Spider dev execution match at four inference turns, `+2.71
  pp` above the single-turn GRPO control under the same turn budget.

## Short Version

- Built a local Text-to-SQL self-correction agent with SFT + GRPO and SQLite
  sandbox rewards, improving Spider dev execution match from `11.32%` raw base
  to `75.24%` with multi-turn execution feedback.

## Interview Framing

- The main engineering contribution is not just fine-tuning: it is the local
  execution environment, reward shaping, and controlled ablations that separate
  SFT cold start, GRPO reward optimization, and multi-turn self-correction.
- The strongest experimental claim is Spider. BIRD transfer is harder and
  timeout-limited, but still shows raw base `6.26%`, SFT `23.79%`, and
  multi-turn GRPO `24.45%` under timeout-30 re-evaluation.
