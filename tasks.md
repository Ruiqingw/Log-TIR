# Log-TIR Current Tasks

This file is the current handoff state. The base-model baseline, Spider
turn-sweep, single-turn GRPO control, and BIRD transfer checks are complete.

## Completed Baseline And Packaging Status

Raw base-model baseline:

- Spider dev, raw `Qwen/Qwen2.5-Coder-3B-Instruct`:
  `117 / 1034 = 11.32%`.
- BIRD dev, raw `Qwen/Qwen2.5-Coder-3B-Instruct`, timeout-3:
  `95 / 1534 = 6.19%`.
- BIRD dev, raw `Qwen/Qwen2.5-Coder-3B-Instruct`, timeout-30 saved-response
  re-eval: `96 / 1534 = 6.26%`.

Packaging files now updated:

- `docs/experiment-results.md`: full experiment record with raw base, SFT,
  single-turn GRPO, multi-turn GRPO, turn sweep, BIRD, and timeout caveats.
- `README.md`: concise project overview and headline results.
- `docs/resume-bullets.md`: resume and interview framing bullets.

## Current Safe Claim

```text
On Spider dev, raw Qwen2.5-Coder-3B-Instruct reaches 11.32% execution match.
The SFT cold start raises this to 70.02%. The selected multi-turn GRPO
checkpoint reaches 71.76% under the standalone single-pass evaluator, 74.56% at
two inference turns, and 75.24% at four inference turns. Under the same
four-turn evaluator, multi-turn GRPO is 2.71 percentage points above the
single-turn GRPO control.
```

BIRD should be framed only as directional transfer evidence because the current
evaluator is still timeout-limited. Under timeout-30 saved-response re-eval,
raw base is `6.26%`, SFT is `23.79%`, and multi-turn GRPO is `24.45%`.

## Optional Next Tasks

No additional training is required for the current resume-grade package.

Optional improvements:

- Produce a final clean BIRD evaluator run with lower concurrency and a longer
  timeout if BIRD becomes important for the writeup.
- Add a small qualitative appendix with representative self-correction cases
  from Spider turn-2 rescues.
- Polish `README.md` further if the repository will be shared publicly.
