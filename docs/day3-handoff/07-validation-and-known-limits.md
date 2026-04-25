# Validation And Known Limits

## Day 3 已做的验证

本地做过这些验证：

```bash
python3 -m py_compile \
  rl_data.py \
  openrlhf_reward.py \
  agent_rollout.py \
  tests/test_rl_data.py \
  tests/test_openrlhf_reward.py \
  tests/test_agent_rollout.py
```

```bash
python3 -m pytest tests -q
```

观察到：

```text
27 passed in 1.03s
```

训练脚本语法检查：

```bash
bash -n scripts/train_sft_openrlhf.sh
bash -n scripts/train_grpo_openrlhf.sh
```

GRPO 数据生成验证：

```bash
python3 rl_data.py \
  --spider-root data/spider \
  --output data/rl/spider_grpo_train_2000.jsonl \
  --limit 2000
```

观察到：

```json
{
  "output_path": "data/rl/spider_grpo_train_2000.jsonl",
  "count": 2000,
  "split": "train_spider",
  "seed": 42
}
```

dev 数据生成验证：

```bash
python3 rl_data.py \
  --spider-root data/spider \
  --split dev \
  --output data/rl/spider_grpo_dev_100.jsonl \
  --limit 100
```

观察到：

```json
{
  "output_path": "data/rl/spider_grpo_dev_100.jsonl",
  "count": 100,
  "split": "dev",
  "seed": 42
}
```

## Review 后已经修的点

Claude review 指出 Day 3 有几个缺口。当前已处理：

- 增加 `agent_rollout.py`，实现最多 2 轮自纠错 helper。
- 增加 dev eval 数据路径和 GRPO 脚本里的 `--eval.dataset` / `--eval.steps 50`。
- 把默认 rollout batch 从 128 降到 32，避免一开始就在 24GB GPU 上 OOM。
- 给 gold SQL 执行加 `@lru_cache`，减少同一 prompt 下重复执行 gold SQL。
- 增加“SQL 能执行但结果错误”时 reward 为 `0.3` 的测试。

## 仍然没有验证的内容

这些内容必须在服务器上验证：

- `scripts/train_sft_openrlhf.sh` 是否能完整跑完。
- `scripts/train_grpo_openrlhf.sh` 是否和当前服务器安装的 OpenRLHF 参数完全兼容。
- Ray job 的 runtime env 是否能正确包含本仓库代码。
- reward function 是否能被 OpenRLHF worker 正确加载。
- vLLM、DeepSpeed、Ray 组合是否在目标 GPU 上稳定。
- dev eval hook 是否和当前 OpenRLHF 版本参数名一致。

## Multi-Turn 状态

两轮自纠错已经通过 `openrlhf_agent.py` 接到 OpenRLHF multi-turn agent 入口。

当前更准确的表述：

```text
The project implements two-turn self-correction as an OpenRLHF multi-turn agent via --agent_func_path, with a single-turn reward-function fallback available through GRPO_MULTI_TURN=0.
```

仍然不能说：

```text
The multi-turn GRPO run has been validated end-to-end on the target server.
```

因为还没有在目标服务器上跑 OpenRLHF/Ray/vLLM 的真实 multi-turn smoke test。

## 训练日志体系还没补齐

`AGENTS.md` 里要求未来训练日志按：

```text
logs/
├── run_YYYYMMDD_HHMM/
│   ├── config.json
│   ├── metrics.jsonl
│   ├── trajectories.jsonl
│   ├── train.stdout.log
│   └── train.stderr.log
└── latest -> run_YYYYMMDD_HHMM
```

Day 3 还没有实现这个完整日志包装层。

目前脚本只负责启动 OpenRLHF。后续建议新增一个 wrapper：

1. 创建 run directory。
2. 保存当前 git commit、branch、dirty diff。
3. 保存 config。
4. 捕获 stdout 和 stderr。
5. 更新 `logs/latest`。
6. 再调用 SFT 或 GRPO 脚本。

## 下一步建议

优先级从高到低：

1. 在服务器上跑通 SFT 1 个短 sanity run。
2. 用 SFT checkpoint 跑 GRPO 10 到 50 step smoke test。
3. 把 OpenRLHF 实际报错和参数兼容问题写回 `.codex-reports/`。
4. 接入完整 run logging wrapper。
5. 决定自纠错是接入 OpenRLHF rollout，还是先作为 eval-only agentic analysis。
6. 开始记录训练曲线：format rate、no-error rate、exec-match rate、平均 reward。
