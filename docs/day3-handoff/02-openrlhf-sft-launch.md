# OpenRLHF SFT Launch

## 相关文件

- `scripts/train_sft_openrlhf.sh`
- `configs/openrlhf_day3.env.example`
- `sft_data.py`
- `data/sft/spider_sft_2000.jsonl`

## SFT 数据从哪里来

SFT 数据由 Day 2 的 `sft_data.py` 生成。

推荐命令：

```bash
python3 sft_data.py \
  --spider-root data/spider \
  --output data/sft/spider_sft_2000.jsonl \
  --limit 2000 \
  --require-executable-gold \
  --teacher-requests-output data/sft/spider_teacher_requests_2000.jsonl
```

每条样本大致包含：

```json
{
  "task": "text-to-sql-sft",
  "split": "train_spider",
  "db_id": "some_database",
  "question": "natural language question",
  "prompt": "schema plus question plus tag rules",
  "response": "<thought>...</thought>\n<action>...</action>",
  "gold_sql": "SELECT ..."
}
```

OpenRLHF SFT 脚本只用两个字段：

- `prompt`：模型输入。
- `response`：监督学习目标。

## 启动脚本做了什么

`scripts/train_sft_openrlhf.sh` 做这些事：

1. 进入仓库根目录。
2. 如果存在 `configs/openrlhf_day3.env`，就加载里面的环境变量。
3. 检查 `SFT_DATA` 指向的数据文件是否存在。
4. 用 `deepspeed --module openrlhf.cli.train_sft` 启动训练。
5. 把 `prompt` 作为 input key，把 `response` 作为 output key。
6. 默认输出到 `checkpoints/qwen2.5-coder-3b-logtir-sft`。

默认关键参数：

```text
MODEL_NAME_OR_PATH=Qwen/Qwen2.5-Coder-3B-Instruct
SFT_DATA=data/sft/spider_sft_2000.jsonl
SFT_OUTPUT_DIR=checkpoints/qwen2.5-coder-3b-logtir-sft
SFT_MAX_LEN=4096
SFT_BATCH_SIZE=64
SFT_MICRO_BATCH_SIZE=1
SFT_MAX_EPOCHS=1
SFT_LR=5e-6
```

## 为什么 SFT 只训一轮

这里的 SFT 是 cold start，不是最终能力来源。它的目的主要是：

- 学会输出 `<thought>` 和 `<action>`。
- 让 `<action>` 里更稳定地出现单条 SQLite query。
- 降低 GRPO 初期的格式失败率。

如果 SFT 训得太久，模型可能过拟合 gold SQL 和固定格式，反而影响后续探索。

## 在服务器上怎么改配置

不要直接改脚本也可以运行。推荐复制一份配置文件：

```bash
cp configs/openrlhf_day3.env.example configs/openrlhf_day3.env
```

然后按服务器实际路径改：

```bash
MODEL_NAME_OR_PATH=/path/to/Qwen2.5-Coder-3B-Instruct
SFT_DATA=data/sft/spider_sft_2000.jsonl
SFT_OUTPUT_DIR=checkpoints/qwen2.5-coder-3b-logtir-sft
```

启动：

```bash
bash scripts/train_sft_openrlhf.sh
```

## 常见失败点

如果报 `Missing SFT data`：

- 说明 `data/sft/spider_sft_2000.jsonl` 不存在。
- 先重新运行 `sft_data.py`。
- 或者把 Mac 上生成的数据同步到服务器同一路径。

如果 OpenRLHF 或 DeepSpeed import 失败：

- 说明服务器训练环境没装好。
- 先确认 Python 环境里能执行 `python3 -m openrlhf.cli.train_sft --help`。

如果显存不够：

- 先减小 `SFT_BATCH_SIZE`。
- 保持 `SFT_MICRO_BATCH_SIZE=1`。
- 保持 gradient checkpointing 开启。

