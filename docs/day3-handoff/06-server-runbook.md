# Server Runbook

这个文件记录从一台新服务器接手 Day 3 到启动训练的推荐顺序。

## 1. 拉取代码

在服务器上进入你要放项目的位置：

```bash
git clone <your-repo-url> Log-TIR
cd Log-TIR
```

如果已经有仓库：

```bash
git pull
```

当前 Day 3 代码 commit 是：

```text
cabbd75 Add OpenRLHF SFT and GRPO launch pipeline
```

## 2. 准备 Python 训练环境

服务器需要安装：

- Python
- PyTorch with CUDA
- OpenRLHF
- Ray
- DeepSpeed
- vLLM
- transformers
- pytest

先验证这些命令是否能跑：

```bash
python3 -c "import torch; print(torch.cuda.is_available())"
python3 -m openrlhf.cli.train_sft --help
python3 -m openrlhf.cli.train_ppo_ray --help
ray --version
deepspeed --version
```

如果这些失败，先修环境，不要直接跑训练脚本。

## 3. 准备 Spider 数据

`data/` 不进 git，所以服务器必须有 Spider 数据。

目录应类似：

```text
data/spider/spider_data/
├── tables.json
├── train_spider.json
├── dev.json
└── database/
    └── <db_id>/
        └── <db_id>.sqlite
```

验证：

```bash
python3 eval.py --spider-root data/spider --use-gold-predictions --limit 100
```

如果 gold SQL evaluator 都跑不通，不要继续训练。

## 4. 生成 SFT 数据

```bash
python3 sft_data.py \
  --spider-root data/spider \
  --output data/sft/spider_sft_2000.jsonl \
  --limit 2000 \
  --require-executable-gold \
  --teacher-requests-output data/sft/spider_teacher_requests_2000.jsonl
```

检查文件：

```bash
wc -l data/sft/spider_sft_2000.jsonl
head -n 1 data/sft/spider_sft_2000.jsonl
```

期望：

- 行数是 2000。
- 每行是 JSON。
- 有 `prompt` 和 `response` 字段。
- `response` 包含 `<thought>` 和 `<action>`。

## 5. 生成 GRPO 数据

训练 prompt：

```bash
python3 rl_data.py \
  --spider-root data/spider \
  --output data/rl/spider_grpo_train_2000.jsonl \
  --limit 2000
```

dev eval prompt：

```bash
python3 rl_data.py \
  --spider-root data/spider \
  --split dev \
  --output data/rl/spider_grpo_dev_100.jsonl \
  --limit 100
```

检查：

```bash
wc -l data/rl/spider_grpo_train_2000.jsonl
wc -l data/rl/spider_grpo_dev_100.jsonl
```

期望：

- train 是 2000 行。
- dev 是 100 行。

## 6. 复制并修改配置

```bash
cp configs/openrlhf_day3.env.example configs/openrlhf_day3.env
```

按服务器实际情况修改：

```text
MODEL_NAME_OR_PATH=/path/to/Qwen2.5-Coder-3B-Instruct
SFT_OUTPUT_DIR=checkpoints/qwen2.5-coder-3b-logtir-sft
GRPO_ACTOR_MODEL=checkpoints/qwen2.5-coder-3b-logtir-sft
GRPO_GPUS_PER_NODE=1
GRPO_ROLLOUT_BATCH_SIZE=32
GRPO_N_SAMPLES_PER_PROMPT=4
```

如果 Spider 数据不在默认位置：

```bash
export SPIDER_ROOT=/absolute/path/to/spider_data
```

## 7. 先跑本地测试

在服务器训练前跑：

```bash
python3 -m pytest tests -q
bash -n scripts/train_sft_openrlhf.sh
bash -n scripts/train_grpo_openrlhf.sh
```

如果测试失败，先修测试。不要用失败状态启动长训练。

## 8. 启动 SFT

```bash
bash scripts/train_sft_openrlhf.sh
```

训练完成后检查：

```bash
ls checkpoints/qwen2.5-coder-3b-logtir-sft
```

期望有 Hugging Face checkpoint 相关文件。

## 9. 启动 GRPO

如果当前 shell 还没有 Ray：

```bash
START_RAY=1 bash scripts/train_grpo_openrlhf.sh
```

如果 Ray 已经启动：

```bash
bash scripts/train_grpo_openrlhf.sh
```

如果显存 OOM：

优先修改：

```text
GRPO_ROLLOUT_BATCH_SIZE=16
GRPO_TRAIN_BATCH_SIZE=16
```

先不要马上增大 `G`。等 `G=4` 稳定后，再考虑 `G=8`。

## 10. 训练时应该观察什么

至少观察：

- reward 均值是否从接近 0 上升。
- format reward 是否快速接近稳定。
- no-error 比例是否上升。
- exec-match 是否开始出现非零。
- 每 50 步 dev eval 是否能跑完。
- Ray job 是否反复重启。
- vLLM 是否 OOM。

如果所有 reward 长期为 0，优先检查：

- 模型是否输出 `<thought>` 和 `<action>`。
- reward 是否能读到 label。
- `SPIDER_ROOT` 是否正确。
- SQLite 数据库是否存在。
- OpenRLHF 传入 reward function 的 query/prompt 是否符合预期。

