# 02. Day 1 Scope And Deliverables

## Day 1 的明确目标

Day 1 要完成四件事：

1. 下载 Spider 数据集
2. 写 `sandbox.py`
3. 写 `eval.py`
4. 用 Spider dev 的 gold SQL 验证 evaluator 是对的

这四件事的关系是串联的，不是并列的：

1. 先有数据，才能测试执行和评估
2. 先有 sandbox，才能真正跑 SQL
3. 先有 evaluator，才能知道 SQL 是否答对
4. 只有 gold SQL 自检通过，后面的 reward 才值得信

## 仓库里实际新增了什么

本次新增或更新的关键文件如下：

- `AGENTS.md`
- `sandbox.py`
- `eval.py`
- `tests/test_sandbox.py`
- `tests/test_eval.py`
- `.codex-reports/day1.md`
- `docs/day1-handoff/` 这个说明文件夹

## 每个文件的职责

`AGENTS.md`
: 给后续协作者的项目约定，包括默认超参、Day 1 目标、如何运行、测试策略。

`sandbox.py`
: 安全执行 SQL 的入口。负责只读限制、超时控制、错误捕获、结果返回。

`eval.py`
: 评估 SQL 是否答对问题的入口。核心指标是 execution match。

`tests/test_sandbox.py`
: 验证 sandbox 的关键行为是否正确。

`tests/test_eval.py`
: 验证 evaluator 的关键行为是否正确。

`.codex-reports/day1.md`
: 记录 Day 1 实际跑过的命令和结果。

## Day 1 实际达成的状态

已经达成：

- Spider 已经下载并解压
- `sandbox.py` 能执行只读 SQL
- `sandbox.py` 会拒绝明显的危险 SQL
- `sandbox.py` 有超时机制
- `eval.py` 能跑 Spider 风格的数据
- `eval.py` 能区分需要保序和不需要保序的结果比较
- `eval.py` 已通过 Spider dev 的 gold SQL 自检
- pytest 已覆盖核心边界情况

尚未触及：

- 训练数据生成
- prompt 设计
- SFT
- GRPO
- 服务器侧训练工作流

## Day 1 的一句话总结

如果你只想记一句话：

Day 1 做的不是“训练模型”，而是“把训练前必须可信的执行环境和评估环境先搭好，并证明它们是可信的”。
