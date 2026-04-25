# Day 3 Context

## Day 3 要解决什么

Day 1 做了两件基础设施：

- `sandbox.py`：安全执行 SQLite SQL，支持只读、防危险操作、超时。
- `eval.py`：对比模型 SQL 和 gold SQL 的执行结果，得到 execution match。

Day 2 做了 SFT 冷启动数据：

- `sft_data.py`：从 Spider 生成 `<thought>...</thought><action>...</action>` 格式的监督微调样本。
- 重点约束：不要从 gold SQL 伪造推理，只用 `format_only` 模板教模型输出格式。

Day 3 要把这些东西接到训练框架：

- SFT：让 base model 先学会稳定输出 `<thought>` 和 `<action>`。
- GRPO：让模型采样多个 SQL，用 reward 判断哪一个更好。
- Reward：用 sandbox 执行 SQL，按格式、无错误、执行匹配分层给分。
- 自纠错：第一轮 SQL 失败时，把 SQLite 错误或空结果反馈给模型，让它第二轮修改。

## 为什么先 SFT 再 GRPO

直接对 base model 做 GRPO 很容易卡住，因为早期输出可能没有稳定标签格式，reward 大量为 0。SFT 的作用不是教会完整推理，而是让模型先学会固定输出协议：

```text
<thought>...</thought>
<action>SELECT ...</action>
```

这样 GRPO 阶段至少能稳定抽取 SQL 并执行，reward 才能发挥作用。

## OpenRLHF 在这里扮演什么角色

OpenRLHF 是训练框架，负责：

- 加载 actor model。
- 按 prompt 采样多个 response。
- 调用 reward function 打分。
- 根据 group 内 reward 做 GRPO 更新。
- 管理 Ray、vLLM、DeepSpeed 等训练组件。

本仓库负责提供：

- OpenRLHF 可读的数据 JSONL。
- OpenRLHF 可调用的 reward function。
- 启动脚本和默认超参数。
- 本地 sandbox/evaluator 作为 reward 的底层执行能力。

## 当前 Day 3 的边界

已经完成：

- SFT 启动脚本。
- GRPO prompt/label 数据生成。
- GRPO 启动脚本。
- 分层 reward function。
- gold SQL 执行缓存。
- 100 条 dev prompt eval 数据生成路径。
- 两轮自纠错 helper。
- 单元测试。

没有完成：

- 没有在服务器上真实跑完 SFT。
- 没有在服务器上真实跑完 GRPO。
- 两轮自纠错 helper 还没有接入 OpenRLHF 默认 rollout loop。
- 没有实现训练日志目录的完整 `logs/run_YYYYMMDD_HHMM/` 记录体系。

