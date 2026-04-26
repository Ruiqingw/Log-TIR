#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO_ROOT"

if [[ -f configs/openrlhf_day3.env ]]; then
  set -a
  source configs/openrlhf_day3.env
  set +a
fi

GRPO_PROMPT_DATA="${GRPO_PROMPT_DATA:-data/rl/spider_grpo_train_2000.jsonl}"
GRPO_OUTPUT_DIR="${GRPO_OUTPUT_DIR:-checkpoints/qwen2.5-coder-3b-logtir-grpo}"
GRPO_CKPT_DIR="${GRPO_CKPT_DIR:-checkpoints/qwen2.5-coder-3b-logtir-grpo-ckpt}"
GRPO_ACTOR_MODEL="${GRPO_ACTOR_MODEL:-checkpoints/qwen2.5-coder-3b-logtir-sft}"
GRPO_REWARD_FUNC="${GRPO_REWARD_FUNC:-openrlhf_reward.py}"
GRPO_AGENT_FUNC="${GRPO_AGENT_FUNC:-openrlhf_agent.py}"
GRPO_MULTI_TURN="${GRPO_MULTI_TURN:-1}"
LOGTIR_AGENT_MAX_TURNS="${LOGTIR_AGENT_MAX_TURNS:-2}"
LOGTIR_AGENT_TIMEOUT="${LOGTIR_AGENT_TIMEOUT:-3.0}"
LOGTIR_AGENT_TURN_PENALTY="${LOGTIR_AGENT_TURN_PENALTY:-0.05}"
GRPO_GPUS_PER_NODE="${GRPO_GPUS_PER_NODE:-1}"
GRPO_RAY_NUM_GPUS="${GRPO_RAY_NUM_GPUS:-$GRPO_GPUS_PER_NODE}"
GRPO_ACTOR_GPUS_PER_NODE="${GRPO_ACTOR_GPUS_PER_NODE:-$GRPO_GPUS_PER_NODE}"
GRPO_REF_GPUS_PER_NODE="${GRPO_REF_GPUS_PER_NODE:-$GRPO_ACTOR_GPUS_PER_NODE}"
GRPO_VLLM_ENGINES="${GRPO_VLLM_ENGINES:-1}"
GRPO_VLLM_TP="${GRPO_VLLM_TP:-1}"
GRPO_VLLM_ENFORCE_EAGER="${GRPO_VLLM_ENFORCE_EAGER:-0}"
GRPO_VLLM_ENABLE_SLEEP="${GRPO_VLLM_ENABLE_SLEEP:-0}"
GRPO_VLLM_GPU_MEMORY_UTILIZATION="${GRPO_VLLM_GPU_MEMORY_UTILIZATION:-}"
GRPO_COLOCATE_ALL="${GRPO_COLOCATE_ALL:-0}"
GRPO_TRAIN_BATCH_SIZE="${GRPO_TRAIN_BATCH_SIZE:-32}"
GRPO_ROLLOUT_BATCH_SIZE="${GRPO_ROLLOUT_BATCH_SIZE:-32}"
GRPO_N_SAMPLES_PER_PROMPT="${GRPO_N_SAMPLES_PER_PROMPT:-4}"
GRPO_MAX_NEW_TOKENS="${GRPO_MAX_NEW_TOKENS:-}"
GRPO_DYNAMIC_BATCH_ENABLE="${GRPO_DYNAMIC_BATCH_ENABLE:-1}"
GRPO_MAX_EPOCHS="${GRPO_MAX_EPOCHS:-1}"
GRPO_MAX_SAMPLES="${GRPO_MAX_SAMPLES:-2000}"
GRPO_MAX_LEN="${GRPO_MAX_LEN:-4096}"
GRPO_PACKING_SAMPLES="${GRPO_PACKING_SAMPLES:-1}"
GRPO_ACTOR_LR="${GRPO_ACTOR_LR:-5e-7}"
GRPO_INIT_KL_COEF="${GRPO_INIT_KL_COEF:-0.01}"
GRPO_USE_KL_LOSS="${GRPO_USE_KL_LOSS:-1}"
GRPO_KL_ESTIMATOR="${GRPO_KL_ESTIMATOR:-k3}"
GRPO_ADAM_OFFLOAD="${GRPO_ADAM_OFFLOAD:-0}"
GRPO_EVAL_DATA="${GRPO_EVAL_DATA:-data/rl/spider_grpo_dev_100.jsonl}"
GRPO_EVAL_STEPS="${GRPO_EVAL_STEPS:-50}"
RAY_ADDRESS="${RAY_ADDRESS:-http://127.0.0.1:8265}"
GRPO_USE_RAY_JOB="${GRPO_USE_RAY_JOB:-0}"
LOGTIR_RAY_TMPDIR="${LOGTIR_RAY_TMPDIR:-/Work21/2024/luyuheng/ray}"
LOGTIR_RAY_NUM_CPUS="${LOGTIR_RAY_NUM_CPUS:-8}"
LOGTIR_ENABLE_RUN_LOGS="${LOGTIR_ENABLE_RUN_LOGS:-1}"

case "$LOGTIR_RAY_TMPDIR" in
  /*) ;;
  *) LOGTIR_RAY_TMPDIR="$REPO_ROOT/$LOGTIR_RAY_TMPDIR" ;;
esac
mkdir -p "$LOGTIR_RAY_TMPDIR"
export RAY_TMPDIR="$LOGTIR_RAY_TMPDIR"
export LOGTIR_RAY_NUM_CPUS
export LOGTIR_RAY_NUM_GPUS="$GRPO_RAY_NUM_GPUS"
export PYTHONPATH="$REPO_ROOT:${PYTHONPATH:-}"
LOGTIR_LOCAL_NO_PROXY="127.0.0.1,localhost,::1,10.10.62.141"
export NO_PROXY="${NO_PROXY:+$NO_PROXY,}$LOGTIR_LOCAL_NO_PROXY"
export no_proxy="${no_proxy:+$no_proxy,}$LOGTIR_LOCAL_NO_PROXY"

if [[ "$LOGTIR_ENABLE_RUN_LOGS" == "1" ]]; then
  LOGTIR_RUN_DIR="$(python3 run_logging.py prepare --kind grpo --repo-root "$REPO_ROOT")"
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
    WANDB_API_KEY="$(conda run -n logtir python -c 'import wandb; print(wandb.api.api_key or "")' 2>/dev/null | tail -n 1)"
  fi
  if [[ -z "${WANDB_API_KEY:-}" ]]; then
    WANDB_API_KEY="$(python3 -c 'import netrc; auth = netrc.netrc().authenticators("api.wandb.ai"); print(auth[2] if auth else "")' 2>/dev/null || true)"
  fi
  if [[ -z "${WANDB_API_KEY:-}" ]]; then
    echo "LOGTIR_WANDB=1 requires WANDB_API_KEY or an existing wandb login" >&2
    exit 1
  fi
  export WANDB_API_KEY
  WANDB_ARGS=(
    --logger.wandb.key env
    --logger.wandb.project "${WANDB_PROJECT:-log-tir}"
    --logger.wandb.run_name "${WANDB_RUN_NAME:-$(basename "${LOGTIR_RUN_DIR:-grpo}")}"
  )
fi

if [[ ! -f "$GRPO_PROMPT_DATA" ]]; then
  echo "Missing GRPO prompt data: $GRPO_PROMPT_DATA" >&2
  echo "Run: python3 rl_data.py --spider-root data/spider --output $GRPO_PROMPT_DATA --limit 2000" >&2
  exit 1
fi

if [[ "$GRPO_MULTI_TURN" == "1" ]]; then
  if [[ ! -f "$GRPO_AGENT_FUNC" ]]; then
    echo "Missing agent function: $GRPO_AGENT_FUNC" >&2
    exit 1
  fi
else
  if [[ ! -f "$GRPO_REWARD_FUNC" ]]; then
    echo "Missing reward function: $GRPO_REWARD_FUNC" >&2
    exit 1
  fi
fi

case "$GRPO_PROMPT_DATA" in
  /*) ;;
  *) GRPO_PROMPT_DATA="$REPO_ROOT/$GRPO_PROMPT_DATA" ;;
esac

case "$GRPO_REWARD_FUNC" in
  /*) ;;
  *) GRPO_REWARD_FUNC="$REPO_ROOT/$GRPO_REWARD_FUNC" ;;
esac

case "$GRPO_AGENT_FUNC" in
  /*) ;;
  *) GRPO_AGENT_FUNC="$REPO_ROOT/$GRPO_AGENT_FUNC" ;;
esac

if [[ "$GRPO_MULTI_TURN" == "1" ]]; then
  export LOGTIR_AGENT_MAX_TURNS LOGTIR_AGENT_TIMEOUT LOGTIR_AGENT_TURN_PENALTY
  AGENT_ARGS=(--train.agent_func_path "$GRPO_AGENT_FUNC")
else
  AGENT_ARGS=(--reward.remote_url "$GRPO_REWARD_FUNC")
fi

if [[ -f "$GRPO_EVAL_DATA" ]]; then
  case "$GRPO_EVAL_DATA" in
    /*) ;;
    *) GRPO_EVAL_DATA="$REPO_ROOT/$GRPO_EVAL_DATA" ;;
  esac
  EVAL_ARGS=(--eval.dataset "$GRPO_EVAL_DATA" --eval.steps "$GRPO_EVAL_STEPS")
else
  echo "Warning: eval dataset not found, skipping OpenRLHF eval hook: $GRPO_EVAL_DATA" >&2
  EVAL_ARGS=()
fi

KL_ARGS=(--algo.kl.estimator "$GRPO_KL_ESTIMATOR")
if [[ "$GRPO_USE_KL_LOSS" == "1" ]]; then
  KL_ARGS+=(--algo.kl.use_loss)
fi

if [[ "$GRPO_USE_RAY_JOB" == "1" && "${START_RAY:-0}" == "1" ]]; then
  ray start --head --node-ip-address 0.0.0.0 --num-gpus "$GRPO_RAY_NUM_GPUS"
fi

GRPO_COMMAND=(
  python3 -m logtir_train_ppo_ray
  --ref.num_nodes 1 \
  --ref.num_gpus_per_node "$GRPO_REF_GPUS_PER_NODE" \
  --actor.num_nodes 1 \
  --actor.num_gpus_per_node "$GRPO_ACTOR_GPUS_PER_NODE" \
  --vllm.num_engines "$GRPO_VLLM_ENGINES" \
  --vllm.tensor_parallel_size "$GRPO_VLLM_TP" \
  --actor.model_name_or_path "$GRPO_ACTOR_MODEL" \
  "${AGENT_ARGS[@]}" \
  --ckpt.output_dir "$GRPO_OUTPUT_DIR" \
  --ckpt.path "$GRPO_CKPT_DIR" \
  --ckpt.save_hf \
  "${EVAL_ARGS[@]}" \
  --train.batch_size "$GRPO_TRAIN_BATCH_SIZE" \
  --rollout.batch_size "$GRPO_ROLLOUT_BATCH_SIZE" \
  --rollout.n_samples_per_prompt "$GRPO_N_SAMPLES_PER_PROMPT" \
  --train.max_epochs "$GRPO_MAX_EPOCHS" \
  --data.max_len "$GRPO_MAX_LEN" \
  --data.max_samples "$GRPO_MAX_SAMPLES" \
  --ds.zero_stage 3 \
  --ds.param_dtype bf16 \
  --actor.adam.lr "$GRPO_ACTOR_LR" \
  --algo.kl.init_coef "$GRPO_INIT_KL_COEF" \
  "${KL_ARGS[@]}" \
  --algo.advantage.estimator group_norm \
  --data.prompt_dataset "$GRPO_PROMPT_DATA" \
  --data.input_key prompt \
  --data.label_key label \
  --actor.gradient_checkpointing_enable
)
if [[ -n "$GRPO_MAX_NEW_TOKENS" ]]; then
  GRPO_COMMAND+=(--rollout.max_new_tokens "$GRPO_MAX_NEW_TOKENS")
fi
if [[ "$GRPO_PACKING_SAMPLES" == "1" ]]; then
  GRPO_COMMAND+=(--ds.packing_samples)
fi
if [[ "$GRPO_VLLM_ENGINES" != "0" ]]; then
  GRPO_COMMAND+=(--vllm.sync_backend nccl)
fi
if [[ "$GRPO_COLOCATE_ALL" == "1" ]]; then
  GRPO_COMMAND+=(--train.colocate_all)
else
  GRPO_COMMAND+=(--train.colocate_actor_ref)
fi
if [[ "$GRPO_VLLM_ENFORCE_EAGER" == "1" ]]; then
  GRPO_COMMAND+=(--vllm.enforce_eager)
fi
if [[ "$GRPO_VLLM_ENABLE_SLEEP" == "1" ]]; then
  GRPO_COMMAND+=(--vllm.enable_sleep)
fi
if [[ -n "$GRPO_VLLM_GPU_MEMORY_UTILIZATION" ]]; then
  GRPO_COMMAND+=(--vllm.gpu_memory_utilization "$GRPO_VLLM_GPU_MEMORY_UTILIZATION")
fi
if [[ "$GRPO_DYNAMIC_BATCH_ENABLE" == "1" ]]; then
  GRPO_COMMAND+=(--train.dynamic_batch_enable)
fi
if [[ "$GRPO_ADAM_OFFLOAD" == "1" ]]; then
  GRPO_COMMAND+=(--ds.adam_offload)
fi
if [[ "${LOGTIR_WANDB:-0}" == "1" ]]; then
  GRPO_COMMAND+=("${WANDB_ARGS[@]}")
fi

if [[ "$GRPO_USE_RAY_JOB" == "1" ]]; then
  GRPO_COMMAND=(
    ray job submit --address="$RAY_ADDRESS"
    --runtime-env-json="{\"working_dir\":\"$REPO_ROOT\"}"
    -- "${GRPO_COMMAND[@]}"
  )
fi

"${GRPO_COMMAND[@]}"
