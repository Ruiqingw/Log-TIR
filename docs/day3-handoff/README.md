# Day 3 Handoff

这个文件夹记录 `Log-TIR` Day 3 做了什么。

Day 3 的主题是把前两天做好的 Text-to-SQL 数据、执行器和 evaluator，接到 OpenRLHF 的 SFT 与 GRPO 训练入口上。这里的目标不是已经完成一次真实 GPU 训练，而是先把服务器上可以启动训练的代码路径、数据格式、reward 函数和自纠错 helper 建出来，并用本地单元测试验证关键逻辑。

建议阅读顺序：

1. [01-day3-context.md](./01-day3-context.md)
2. [02-openrlhf-sft-launch.md](./02-openrlhf-sft-launch.md)
3. [03-grpo-data-and-launch.md](./03-grpo-data-and-launch.md)
4. [04-reward-function.md](./04-reward-function.md)
5. [05-self-correction-rollout.md](./05-self-correction-rollout.md)
6. [06-server-runbook.md](./06-server-runbook.md)
7. [07-validation-and-known-limits.md](./07-validation-and-known-limits.md)

如果只想快速知道当前新增了什么：

- `scripts/train_sft_openrlhf.sh`：启动 OpenRLHF SFT 的脚本
- `rl_data.py`：把 Spider 样本转成 GRPO 用的 prompt/label JSONL
- `scripts/train_grpo_openrlhf.sh`：启动 OpenRLHF GRPO 的脚本
- `openrlhf_reward.py`：OpenRLHF 调用的本地 reward function
- `agent_rollout.py`：最多两轮的执行反馈自纠错 rollout helper
- `configs/openrlhf_day3.env.example`：服务器运行时的配置模板
- `tests/test_rl_data.py`、`tests/test_openrlhf_reward.py`、`tests/test_agent_rollout.py`：Day 3 新增测试
- `.codex-reports/day3.md`：Day 3 的机器可读/简短运行报告

最重要的现实状态：

- SFT 和 GRPO 启动脚本已经写好，但没有在本地 Mac 上进行 GPU 训练。
- GRPO 默认走 `--agent_func_path openrlhf_agent.py` 多轮 agent 模式。
- 单轮 reward function 仍保留，可用 `GRPO_MULTI_TURN=0` 回退。
- 训练数据仍在 `data/` 下生成，`data/` 被 git ignore，所以服务器需要重新生成或单独同步数据文件。
