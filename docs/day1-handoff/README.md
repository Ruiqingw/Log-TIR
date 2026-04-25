# Day 1 Handoff

这个文件夹是给第一次接手 `Log-TIR` 仓库的人准备的。

目标不是“简单记一下做了什么”，而是让一个不了解这个仓库的人也能在以下前提下快速接手：

- 他知道 Python 和命令行的基本用法
- 他可以借助 AI 解释不熟悉的概念，例如 `GRPO`、`Spider`、`execution match`
- 他需要知道这一天到底改了什么、为什么这样改、怎么重新验证

建议阅读顺序：

1. [01-project-context.md](./01-project-context.md)
2. [02-day1-scope-and-deliverables.md](./02-day1-scope-and-deliverables.md)
3. [03-data-setup-spider.md](./03-data-setup-spider.md)
4. [04-sandbox-design.md](./04-sandbox-design.md)
5. [05-evaluator-design.md](./05-evaluator-design.md)
6. [06-tests-and-validation.md](./06-tests-and-validation.md)
7. [07-rerun-playbook.md](./07-rerun-playbook.md)

如果你只想先知道“现在仓库里有哪些关键文件”，先看下面这个最短摘要：

- 项目目标写在仓库根目录的 `CLAUDE.md`
- 协作约定和默认参数写在仓库根目录的 `AGENTS.md`
- Day 1 核心代码是 `sandbox.py` 和 `eval.py`
- Day 1 测试在 `tests/test_sandbox.py` 和 `tests/test_eval.py`
- Day 1 的实际运行结果写在 `.codex-reports/day1.md`
- 本文件夹是面向人类阅读的拆解版说明，不替代代码和测试

如果你时间很少：

- 想知道项目做什么：看 `01`
- 想知道 Day 1 交付了什么：看 `02`
- 想知道怎么重跑：看 `07`
