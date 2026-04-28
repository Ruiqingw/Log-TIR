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

## 中文版本

### 强版本

- 基于 `Qwen2.5-Coder-3B-Instruct` 构建全本地 Text-to-SQL 自纠错 Agent，
  使用只读 SQLite 沙箱执行结果作为奖励和评估信号，训练与评估均不依赖外部
  推理 API。
- 设计 SFT cold start 到 GRPO 的分阶段训练流程，引入 format、no-error、
  execution-match 奖励；Spider dev execution match 从 raw base 的 `11.32%`
  提升到 SFT 后 `70.02%`，GRPO 后单轮达到 `71.76%`。
- 实现基于执行反馈的多轮 SQL 修复，并加入 single-turn GRPO 对照实验；四轮
  inference 下 Spider dev execution match 达到 `75.24%`，在相同 turn budget
  下比 single-turn GRPO control 高 `+2.71 pp`。

### 短版本

- 构建本地 Text-to-SQL 自纠错 Agent，使用 SFT + GRPO 与 SQLite 沙箱执行奖励，
  将 Spider dev execution match 从 raw base `11.32%` 提升到多轮执行反馈下的
  `75.24%`。

### 面试讲法

- 这个项目的重点不是单纯 fine-tuning，而是把本地 SQL 执行环境、reward
  shaping、self-correction rollout 和对照实验串成一个可验证系统，能够区分
  SFT cold start、GRPO reward optimization、multi-turn feedback 各自的作用。
- 最强结论放在 Spider：raw base `11.32%`，SFT `70.02%`，multi-turn GRPO
  单轮 `71.76%`，四轮执行反馈 `75.24%`。BIRD 作为 transfer 结果描述即可，
  因为当前 evaluator 仍受 timeout 影响；timeout-30 re-eval 下 raw base
  `6.26%`、SFT `23.79%`、multi-turn GRPO `24.45%`。
