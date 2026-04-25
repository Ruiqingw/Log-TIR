# GRPO Data And Launch

## 相关文件

- `rl_data.py`
- `scripts/train_grpo_openrlhf.sh`
- `configs/openrlhf_day3.env.example`
- `openrlhf_reward.py`
- `data/rl/spider_grpo_train_2000.jsonl`
- `data/rl/spider_grpo_dev_100.jsonl`

## GRPO 需要什么数据

GRPO 不需要监督答案作为模型输出目标。它需要：

- `prompt`：让模型生成 SQL 的输入。
- `label`：reward function 用来判断输出好坏的隐藏信息。

`label` 不直接喂给模型，而是给本地 reward function 使用。

## `rl_data.py` 做了什么

`rl_data.py` 从 Spider JSON 中读取样本，再生成 OpenRLHF prompt/label JSONL。

推荐生成训练集：

```bash
python3 rl_data.py \
  --spider-root data/spider \
  --output data/rl/spider_grpo_train_2000.jsonl \
  --limit 2000
```

推荐生成 dev eval 集：

```bash
python3 rl_data.py \
  --spider-root data/spider \
  --split dev \
  --output data/rl/spider_grpo_dev_100.jsonl \
  --limit 100
```

输出里的单条记录结构：

```json
{
  "task": "text-to-sql-rl",
  "record_id": "train_spider:123",
  "datasource": "spider",
  "prompt": "You are a SQLite Text-to-SQL agent...\nSchema...\nQuestion...",
  "label": "{\"db_id\":\"...\",\"db_path\":\"database/.../...sqlite\",\"gold_sql\":\"SELECT ...\",\"question\":\"...\"}"
}
```

注意：`label` 是 JSON 字符串，不是 JSON object。这是为了和 OpenRLHF 的 dataset label key 更容易兼容。

## 为什么 `db_path` 存相对路径

Mac 和服务器上的仓库路径通常不同。如果把绝对路径写进 JSONL，例如：

```text
/Users/ruiqing/Documents/LLM Projects/Log-TIR/data/spider/spider_data/database/...
```

这个文件搬到服务器就不能用了。

所以 Day 3 存的是：

```text
database/<db_id>/<db_id>.sqlite
```

reward 执行时再用 `SPIDER_ROOT` 拼成真实路径。

默认 `SPIDER_ROOT` 是：

```text
data/spider/spider_data
```

如果服务器路径不同，设置：

```bash
export SPIDER_ROOT=/absolute/path/to/spider_data
```

## GRPO 启动脚本做了什么

`scripts/train_grpo_openrlhf.sh` 做这些事：

1. 进入仓库根目录。
2. 加载 `configs/openrlhf_day3.env`。
3. 检查 GRPO prompt 数据是否存在。
4. 检查 reward function 是否存在。
5. 如果 `START_RAY=1`，启动 Ray head。
6. 用 `ray job submit` 启动 `openrlhf.cli.train_ppo_ray`。
7. 设置 GRPO 相关参数。
8. 设置 reward function 文件。
9. 如果 dev eval 数据存在，每 50 步接一次 eval hook。

启动命令：

```bash
START_RAY=1 bash scripts/train_grpo_openrlhf.sh
```

如果 Ray 已经在服务器上启动：

```bash
bash scripts/train_grpo_openrlhf.sh
```

## GRPO 关键默认参数

```text
GRPO_ACTOR_MODEL=checkpoints/qwen2.5-coder-3b-logtir-sft
GRPO_REWARD_FUNC=openrlhf_reward.py
GRPO_TRAIN_BATCH_SIZE=32
GRPO_ROLLOUT_BATCH_SIZE=32
GRPO_N_SAMPLES_PER_PROMPT=4
GRPO_ACTOR_LR=5e-7
GRPO_INIT_KL_COEF=0.01
GRPO_USE_KL_LOSS=1
GRPO_KL_ESTIMATOR=k3
GRPO_EVAL_DATA=data/rl/spider_grpo_dev_100.jsonl
GRPO_EVAL_STEPS=50
```

这里 `GRPO_N_SAMPLES_PER_PROMPT=4` 对应项目设定里的 `G=4`。

`GRPO_ROLLOUT_BATCH_SIZE=32` 意味着每个 rollout batch 会产生：

```text
32 prompts * 4 samples = 128 generations
```

这个默认值是为了先在 24GB 级别 GPU 上保守启动。稳定后可以再调大。

## OpenRLHF 参数含义

脚本里最关键的几项：

```bash
--algo.advantage.estimator group_norm
```

表示用 group 内归一化的 reward 估计 advantage。这是 GRPO 的核心形式：同一个 prompt 采样多条回答，在同组里比较谁更好。

```bash
--algo.kl.use_loss
--algo.kl.estimator k3
```

表示使用 KL loss 控制 actor 不要偏离参考模型太远，`k3` 是 KL 估计方式。

```bash
--reward.remote_url "$GRPO_REWARD_FUNC"
```

表示 OpenRLHF 会调用本仓库的 `openrlhf_reward.py` 来给 response 打分。

```bash
--data.input_key prompt
--data.label_key label
```

表示 prompt JSONL 里用 `prompt` 作为模型输入，用 `label` 作为 reward 所需标签。

