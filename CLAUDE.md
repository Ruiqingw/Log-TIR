# Log-TIR: Agentic RL for Text-to-SQL

基于 GRPO 的本地自纠错结构化数据查询 Agent 项目。目标是做成简历主项目，投递大模型算法实习岗位。

## 1. 项目定位

训练一个能读懂自然语言问题、生成 SQL 并在沙盒中执行、看懂 traceback 后自我纠错的 Agent。完全在本地闭环（无外部 API），用 GRPO 强化学习微调 `Qwen2.5-Coder-3B-Instruct`。

**三个亮点（简历话术）**
1. Zero-API 本地沙盒环境：`subprocess` 执行，省 API 钱，充分利用多卡并行 rollout
2. 三级阶梯式 reward：format → no-error → exec-match，解决 agent RL 早期奖励稀疏
3. 多轮自纠错对齐：traceback 回灌给模型，GRPO 优化多轮调试行为

## 2. 数据策略（关键：不用 Gemini 原计划的合成日志）

原 Gemini 计划用合成个人日志做 50-100 题，**外部效度为零**，在一线厂面试过不了关。改用学术基准：

- **Spider**（Yale，GitHub `taoyds/spider`）：~7000 train / ~1034 dev，200 个跨领域 DB，官方 SQLite 可执行
- **BIRD**（`AlibabaResearch/DAMO-ConvAI/tree/main/bird`）：1534 dev，真实脏数据

**评估指标**：Execution Match（执行后行集合匹配，注意 ORDER BY 时保序）。
训练前拿官方 gold SQL 跑一遍 dev，evaluator 应 ≥95% 准确，证明 evaluator 本身对。

## 3. 执行计划（9 天版）

| Phase | 天 | 产出 | 验收 |
|---|---|---|---|
| 1. 环境+数据 | 1-2 | `sandbox.py`（3s 超时 + 只读 DB）、`eval.py`（exec match）| evaluator 对 gold SQL ≥95% |
| 2. SFT 冷启动 | 3 | 2000 条格式化轨迹（`<thought>`/`<action>`），1 epoch SFT | dev exec match 30-40%，格式合规 >95% |
| 3. GRPO 主训练 | 4-6 | 多轮自纠错（最多 2 轮），每 50 步 dev 抽样 | dev exec match 提到 ~50%+ |
| 4. 消融+迁移 | 7-8 | 4 条对比曲线 + BIRD zero-shot + reward 消融 | 一张干净的对比表 |
| 5. 分析+写作 | 9 | 20 条"错→traceback→改对"轨迹、README/博客 | 可以直接贴进简历 |

**SFT 冷启动是 Gemini 漏掉的关键步骤**——裸 Qwen2.5-Coder-3B 的 `<thought>/<action>` 依从性差，直跳 GRPO 会长期 0 分不收敛。

**必须交付的对比表**（简历杀手锏）：
| 配置 | Spider dev EX | BIRD dev EX (zero-shot) |
|---|---|---|
| Base zero-shot / +SFT / +SFT+GRPO(单轮) / +SFT+GRPO(多轮) | | |

## 4. 算法与超参要点

- 算法：**GRPO**（无 critic，省显存，适配 24G 单卡）
- 框架：**OpenRLHF** + Ray
- Reward 权重：format 0.1 / no-error 0.2 / **exec-match 1.0**（主奖励必须压倒性大，否则 reward hacking）
- Group size G=4-8（4090 上折中）
- 自纠错：1 个 episode 最多 2 轮，第一轮错就把 stderr/空结果拼进 prompt 再给一次

**常见坑**：
- tier-2 分数过高 → 模型学会永远输出 `SELECT 1`
- exec-match 对 number/string/multi-row 要统一 normalize，否则 tier-3 噪声大
- SQL 执行必须 read-only + timeout + 连接池，防 DROP TABLE / 死循环

## 5. 基础设施

### 硬件 & 成本
- **国内服务器**（已有）：主力训练机
- **vast.ai 备用**：选美/欧节点，4×4090 ≈ $1.2-2.0/hr，整个项目预算 **$100-180**
- 国内服务器跑不通 Claude Code；vast.ai 可以（节点在 GFW 外）

### 开发工作流
```
Mac (Claude Code 主工作台)
  └── git push ──→  私仓 (GitHub/Gitee)
                       └── git pull ──→  国内服务器 (tmux 跑训练)
                                              │
Mac ←── rsync pull (每分钟) ────────────────┘
         (Claude Code 读 remote-logs/)
```

**方向关键**：永远 Mac 主动 SSH 到国内服务器拉日志，反向方向（国内→外）会被墙卡。

### SSH config（`~/.ssh/config`）
加 `ServerAliveInterval 30` + `ControlMaster auto` + `ControlPersist 10m`，多个 rsync 复用连接。

### 同步脚本 `~/bin/sync_logs.sh`
```bash
#!/bin/bash
REMOTE="user@your.china.server"
while true; do
  rsync -az --partial "$REMOTE:~/Log-TIR/logs/" \
    "$HOME/Documents/LLM Projects/Log-TIR/remote-logs/" 2>/dev/null
  sleep 60
done
```
`nohup ~/bin/sync_logs.sh &` 后台跑。

### 训练曲线
wandb.ai（国内一般可达）> SwanLab（国产备选）> TensorBoard（ssh 端口转发）。

## 6. 目录结构与日志约定

```
~/Log-TIR/
├── logs/
│   ├── run_20260424_1530/          # 每次训练独立时间戳目录
│   │   ├── config.json             # 启动时自动 dump
│   │   ├── metrics.jsonl           # 每步的 reward/loss/tier 命中率
│   │   ├── trajectories.jsonl      # 每 rollout 完整文本（采样 10%）
│   │   ├── train.stdout.log
│   │   └── train.stderr.log
│   └── latest -> run_20260424_1530 # 软链接永远指向最新
```

### 日志一律 JSONL，不写大 txt
Claude Code 可以用 `jq` 秒级切片，例如："最近 100 个 exec_match=0 的 trajectory"。

```json
// metrics.jsonl
{"step": 123, "reward_mean": 0.45, "format_match": 0.9, "exec_match": 0.3, "loss": 1.2}

// trajectories.jsonl
{"step": 123, "prompt": "...", "response": "...", "exec_stdout": "...", "exec_stderr": "...", "reward": 0.9}
```

### `config.json` 自动生成
训练启动时一行 `dump_config(run_dir, args)`，自动抓：
- 超参（argparse/YAML）
- git commit + branch + dirty flag + **完整 diff**（未提交改动也存，血泪经验）
- hostname、python/torch/cuda 版本、GPU 型号
- 启动命令 `sys.argv`

这份 config 同时喂给 `wandb.init(config=cfg, name=run_dir.name)`，网页上可按超参分组。

### `latest` 软链接
训练脚本启动时重建：删旧链接 → 指向新 run 目录（用相对路径，rsync 能同步）。调试时 `ssh china-server "tail -f ~/Log-TIR/logs/latest/train.stderr.log"` 实时看报错。

## 7. 代码规范

- 训练脚本开头必做：创建 run_dir、更新 latest 链接、dump_config、初始化 wandb
- Reward 函数纯函数设计，方便单元测试（手动传假 response 验证打分）
- 沙盒执行：`subprocess.run(..., timeout=3, capture_output=True)`，用 read-only SQLite URI `file:db.sqlite?mode=ro`
- 每个 run 开始前 `git status` 提示是否有未提交改动（不阻断，只警告）
