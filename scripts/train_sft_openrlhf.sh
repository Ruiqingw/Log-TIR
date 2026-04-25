#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO_ROOT"

if [[ -f configs/openrlhf_day3.env ]]; then
  set -a
  source configs/openrlhf_day3.env
  set +a
fi

MODEL_NAME_OR_PATH="${MODEL_NAME_OR_PATH:-Qwen/Qwen2.5-Coder-3B-Instruct}"
SFT_DATA="${SFT_DATA:-data/sft/spider_sft_2000.jsonl}"
SFT_OUTPUT_DIR="${SFT_OUTPUT_DIR:-checkpoints/qwen2.5-coder-3b-logtir-sft}"
SFT_MAX_LEN="${SFT_MAX_LEN:-4096}"
SFT_BATCH_SIZE="${SFT_BATCH_SIZE:-64}"
SFT_MICRO_BATCH_SIZE="${SFT_MICRO_BATCH_SIZE:-1}"
SFT_MAX_EPOCHS="${SFT_MAX_EPOCHS:-1}"
SFT_LR="${SFT_LR:-5e-6}"
LOGTIR_ENABLE_RUN_LOGS="${LOGTIR_ENABLE_RUN_LOGS:-1}"

if [[ "$LOGTIR_ENABLE_RUN_LOGS" == "1" ]]; then
  LOGTIR_RUN_DIR="$(python3 run_logging.py prepare --kind sft --repo-root "$REPO_ROOT")"
  export LOGTIR_RUN_DIR
  if [[ "${LOGTIR_LOG_REDIRECTED:-0}" != "1" ]]; then
    export LOGTIR_LOG_REDIRECTED=1
    exec > >(tee -a "$LOGTIR_RUN_DIR/train.stdout.log") 2> >(tee -a "$LOGTIR_RUN_DIR/train.stderr.log" >&2)
  fi
  echo "Log-TIR run dir: $LOGTIR_RUN_DIR"
fi

WANDB_ARGS=()
if [[ "${LOGTIR_WANDB:-0}" == "1" ]]; then
  if [[ -z "${WANDB_API_KEY:-}" ]]; then
    echo "LOGTIR_WANDB=1 requires WANDB_API_KEY" >&2
    exit 1
  fi
  WANDB_ARGS=(--use_wandb "$WANDB_API_KEY")
fi

if [[ ! -f "$SFT_DATA" ]]; then
  echo "Missing SFT data: $SFT_DATA" >&2
  echo "Run: python3 sft_data.py --spider-root data/spider --output $SFT_DATA --limit 2000 --require-executable-gold" >&2
  exit 1
fi

deepspeed --module openrlhf.cli.train_sft \
  --data.max_len "$SFT_MAX_LEN" \
  --data.dataset "$SFT_DATA" \
  --data.input_key prompt \
  --data.output_key response \
  --train.batch_size "$SFT_BATCH_SIZE" \
  --train.micro_batch_size "$SFT_MICRO_BATCH_SIZE" \
  --data.max_samples 2000 \
  --model.model_name_or_path "$MODEL_NAME_OR_PATH" \
  --ckpt.output_dir "$SFT_OUTPUT_DIR" \
  --ckpt.save_steps -1 \
  --logger.logging_steps 1 \
  --eval.steps -1 \
  --ds.zero_stage 2 \
  --train.max_epochs "$SFT_MAX_EPOCHS" \
  --ds.param_dtype bf16 \
  --ds.attn_implementation flash_attention_2 \
  --adam.lr "$SFT_LR" \
  --ckpt.load_enable \
  --ds.packing_samples \
  --model.gradient_checkpointing_enable \
  "${WANDB_ARGS[@]}"
