# GRPO Training Issues

This note records the launch-time issues hit while starting the GRPO stage and the fixes applied.

## Context

- Target stage: GRPO training with OpenRLHF and Ray.
- Run script: `scripts/train_grpo_openrlhf.sh`
- Active config: `configs/openrlhf_day3.env`
- Training command uses the interpreter found through:

```bash
PATH=/Work21/2024/luyuheng/miniconda3/envs/logtir/bin:$PATH bash scripts/train_grpo_openrlhf.sh
```

The effective training Python is:

```text
/Work21/2024/luyuheng/miniconda3/envs/logtir/bin/python -> Python 3.10.18
```

## 1. Wandb Initialization Timeout

### Symptom

Previous GRPO launches died during Ray actor startup with:

```text
wandb.errors.errors.CommError: Run initialization has timed out after 90.0 sec
ray.exceptions.ActorDiedError: The actor died because of an error raised in its creation task
```

The failure happened inside `PPOTrainer.__init__`, so a transient wandb network problem could kill the whole training launch.

### Cause

The server could not reliably reach `api.wandb.ai` during `wandb.init`. OpenRLHF initializes wandb inside a Ray actor, so a wandb timeout becomes an actor creation failure instead of a recoverable logging warning.

### Fix

Added a wandb preflight step to `scripts/train_grpo_openrlhf.sh` when `LOGTIR_WANDB=1`.

The script now:

- resolves `WANDB_API_KEY` before launch,
- tries a short GraphQL request to `https://api.wandb.ai/graphql`,
- records the preflight result in the run log,
- continues when `WANDB_MODE=offline`,
- disables wandb for the run if preflight fails while not in offline mode,
- supports `LOGTIR_WANDB_PREFLIGHT_STRICT=1` for fail-fast behavior.

The rule was also added to `AGENTS.md`: every wandb-enabled training run should test wandb connectivity first.

Current GRPO config keeps:

```bash
LOGTIR_WANDB=1
WANDB_MODE=offline
```

This preserves local wandb logs without letting online wandb timeout kill Ray actors.

## 2. Python Environment Confusion

### Symptom

During preflight testing, this command reported Python 3.8.8:

```bash
/Work21/2024/luyuheng/miniconda3/bin/conda run -n logtir python -V
```

But the project is expected to use Python 3.10 for training.

### Cause

`conda run -n logtir` did not match the interpreter actually used by the launch script. The training script runs after prepending:

```bash
/Work21/2024/luyuheng/miniconda3/envs/logtir/bin
```

to `PATH`, and that direct interpreter is Python 3.10.18.

### Fix

For training-related checks, prefer the exact interpreter path used by the launch script:

```bash
/Work21/2024/luyuheng/miniconda3/envs/logtir/bin/python -V
```

Do not infer the training runtime from plain `conda run -n logtir` unless it has been verified to point to the same environment.

## 3. DeepSpeed Batch Size Assertion

### Symptom

After wandb offline mode was working, the next GRPO launch failed in `ReferenceModelActor.init_model_from_pretrained`:

```text
AssertionError: Check batch related parameters.
train_batch_size is not equal to micro_batch_per_gpu * gradient_acc_step * world_size
32 != 1 * 10 * 3
```

### Cause

The configured global train batch size was:

```bash
GRPO_TRAIN_BATCH_SIZE=32
```

The actor/reference layout uses 3 GPUs:

```bash
GRPO_ACTOR_GPUS_PER_NODE=3
GRPO_REF_GPUS_PER_NODE=3
```

OpenRLHF/DeepSpeed computed `micro_batch_per_gpu=1`, `gradient_accumulation_steps=10`, and `world_size=3`, which implies an effective train batch size of `1 * 10 * 3 = 30`, not 32.

### Fix

Changed GRPO train batch size from 32 to 30 in:

- `configs/openrlhf_day3.env`
- `configs/openrlhf_day3.env.example`
- fallback default in `scripts/train_grpo_openrlhf.sh`

The active launch now includes:

```text
--train.batch_size 30
```

and OpenRLHF prints:

```text
train=namespace(... batch_size=30 ...)
```

This resolved the DeepSpeed assertion.

## 4. Training Status After Fixes

The restarted run used:

```text
launcher log: logs/grpo_launcher_setsid_20260427_022508.log
run dir: logs/run_20260427_022515_grpo
```

Observed startup milestones:

- wandb preflight timed out and was recorded,
- training continued because `WANDB_MODE=offline`,
- Ray local instance started,
- rollout/model/reference/trainer actors initialized,
- data preprocessing completed for 2000 train examples and 100 eval examples,
- wandb offline run was created locally,
- training reached `PPOTrainer.fit` and `PolicyModelActor.execute_batch`.

The earlier wandb timeout and DeepSpeed batch assertion did not recur.

## 5. Proxy Variable Drift

### Symptom

After changing the proxy, the active shell had:

```bash
http_proxy=http://172.18.181.89:7890
https_proxy=http://172.18.181.89:7890
all_proxy=socks5h://127.0.0.1:7891
```

`curl -I --connect-timeout 8 https://api.wandb.ai/graphql` timed out through
the inherited `172.18.181.89:7890` HTTP proxy.

### Cause

The local `mihomo` process was still alive, but it was listening on loopback:

```text
127.0.0.1:7890
127.0.0.1:7891
```

The HTTP(S) proxy variables pointed at a stale non-loopback address.

### Fix

For GRPO launches from this server, prefer explicit loopback proxy variables:

```bash
http_proxy=http://127.0.0.1:7890
https_proxy=http://127.0.0.1:7890
HTTP_PROXY=http://127.0.0.1:7890
HTTPS_PROXY=http://127.0.0.1:7890
all_proxy=socks5h://127.0.0.1:7891
ALL_PROXY=socks5h://127.0.0.1:7891
```

With `-x http://127.0.0.1:7890`, wandb returned HTTP 405 for a HEAD request,
which is expected for the GraphQL endpoint and confirms proxy connectivity.

## 6. Packed Experience Too Small For 3 Actor Ranks

### Symptom

Run `logs/run_20260427_022515_grpo` reached the first rollout and then exited:

```text
ValueError: Insufficient batch size for async_run_method_batch: total_length=1, effective_actors=3
```

### Cause

The run used:

```bash
GRPO_ACTOR_GPUS_PER_NODE=3
GRPO_DYNAMIC_BATCH_ENABLE=1
GRPO_PACKING_SAMPLES=1
```

OpenRLHF's dynamic batch path forces packing. In this rollout, the packed
experience list collapsed to length 1, but the Ray dispatcher had 3 effective
actor ranks and requires at least one experience chunk per rank.

### Fix

Changed the active GRPO config to keep the 3-rank actor/reference layout but
disable packing and dynamic batch:

```bash
GRPO_DYNAMIC_BATCH_ENABLE=0
GRPO_PACKING_SAMPLES=0
```

This keeps `GRPO_N_SAMPLES_PER_PROMPT=4` for group-normalized GRPO while
avoiding the packed-experience dispatcher failure. If this run later hits a
memory limit, reduce `GRPO_ROLLOUT_BATCH_SIZE` before reducing `G`.

## 7. 3-Actor First-Update Stall

### Symptom

Run `logs/run_20260427_032837_grpo` successfully passed the earlier failures:

- wandb preflight returned HTTP 200 through the loopback proxy,
- Ray and vLLM started,
- train/eval datasets loaded,
- the first rollout generated all 16 prompt batches,
- sandbox SQL reward workers were launched.

After that, training stopped making observable progress during the first actor
forward/update:

```text
PolicyModelActor pid=74383 forward: ... 20/22
```

`train.stdout.log` stopped updating at `2026-04-27 03:32:20`,
`train.stderr.log` stopped updating at `2026-04-27 03:34:15`, and
`metrics.jsonl` stayed at 0 bytes. Two actor GPUs stayed at 100% utilization
while the third actor GPU was idle, and no `Global step` was emitted.

### Cause

No traceback or Ray worker error was written before termination. The most likely
issue is a first-update hang in the 3-rank actor/reference ZeRO3 path after
rollout collection. This is separate from the earlier packed-experience
dispatcher exception because packing and dynamic batch were already disabled.

### Fix

Changed the active GRPO config to a conservative stability layout:

```bash
GRPO_GPUS_PER_NODE=1
GRPO_RAY_NUM_GPUS=2
GRPO_ACTOR_GPUS_PER_NODE=1
GRPO_REF_GPUS_PER_NODE=1
GRPO_TRAIN_BATCH_SIZE=8
GRPO_ROLLOUT_BATCH_SIZE=8
GRPO_DYNAMIC_BATCH_ENABLE=0
GRPO_PACKING_SAMPLES=0
```

This colocates actor/reference on one GPU, keeps one GPU for vLLM, preserves
`GRPO_N_SAMPLES_PER_PROMPT=4`, and prioritizes reaching stable `Global step`
metrics before scaling throughput back up.

## 8. Multi-Turn Agent Scored Cumulative Text

### Symptom

Run `logs/run_20260427_034852_grpo` reached stable optimizer steps, but the
first metrics stayed at:

```text
exec_match_rate = 0.0
no_error_rate = 0.0
format_match_rate = 0.46875
```

The printed OpenRLHF samples showed `Execution feedback` reporting:

```text
format: Response must contain only <thought> and <action> tags
```

even when a visible generated answer contained `<thought>...</thought>` and
`<action>...</action>`.

### Cause

The SFT stage stores prompt and target response separately, so it does not use
the multi-turn agent scoring path. GRPO multi-turn scoring does. In practice,
the text passed into `AgentInstance.step()` can include the current observation
or cumulative transcript around the generated answer. The previous
implementation sent that raw text directly to `score_response`, whose strict
parser expects only the two response tags.

That made format reward noisy and prevented the execution reward from measuring
the generated SQL cleanly.

### Fix

Added response extraction in `openrlhf_agent.py` before scoring:

- strip an `observation_text` / `observation` / `prompt` prefix when present,
- if a cumulative transcript remains, score the latest complete
  `<thought>...</thought><action>...</action>` pair,
- keep `score_response` strict so standalone reward validation is unchanged.

Added regression tests in `tests/test_openrlhf_agent.py` for:

- `observation + response` passed as `action_text`,
- second-turn cumulative observation plus final response.

Verified with:

```bash
/Work21/2024/luyuheng/miniconda3/envs/logtir/bin/python -m pytest \
  tests/test_openrlhf_agent.py tests/test_openrlhf_reward.py
```

## 9. Sandbox Timeout Noise Under Rollout Concurrency

### Symptom

After the multi-turn response extraction fix, run `logs/run_20260427_041357_grpo`
no longer showed prompt-prefix format failures and `format_match_rate` improved
from `0.46875` to `0.75`. However `no_error_rate` and `exec_match_rate` stayed
at `0.0`.

The printed sample showed a simple query receiving a sandbox timeout:

```sql
SELECT title FROM Papers WHERE title LIKE "%Database%"
```

with:

```text
TimeoutExpired: query exceeded 3.0s
```

Running the same query directly through `sandbox.execute_sql(..., timeout_s=3.0)`
completed in under 0.2 seconds.

### Cause

The query and DB path are valid. The likely issue is rollout-time concurrency:
`GRPO_ROLLOUT_BATCH_SIZE=8` with `GRPO_N_SAMPLES_PER_PROMPT=4` can create up to
32 generated samples per step, each with up to two turns and sandbox subprocess
execution. The parent-side 3 second timeout includes subprocess scheduling and
startup time, so concurrent reward execution can produce false timeout rewards.

### Attempted Fix

Kept the SQL sandbox timeout at 3 seconds and tried reducing rollout
concurrency:

```bash
GRPO_ROLLOUT_BATCH_SIZE=4
GRPO_N_SAMPLES_PER_PROMPT=4
```

This kept group-normalized GRPO at `G=4` while reducing simultaneous sandbox
subprocess pressure, but run `logs/run_20260427_042932_grpo` failed during
multi-turn vLLM generation:

```text
ValueError: Token id 151890 is out of vocabulary
```

The failure happened when the multi-turn agent fed generated/action tokens back
into vLLM for the second turn. The earlier rollout batch 8 runs did not hit this
vLLM token validation failure.

### Current Fix

Restored the stable rollout batch size and gave the parent-side sandbox wrapper
more scheduling headroom:

```bash
GRPO_ROLLOUT_BATCH_SIZE=8
LOGTIR_AGENT_TIMEOUT=5.0
```

This avoids the batch-4 vLLM OOV path while reducing false timeout rewards from
subprocess scheduling pressure. Longer-term, the better fix is a bounded
sandbox worker pool or batched execution path rather than relying on more
parent-side timeout slack.

### Follow-up Status

Run `logs/run_20260427_043906_grpo` was restarted from the SFT checkpoint with
the response extraction fix and the timeout/batch settings above. OpenRLHF's
`Sample:` log still prints the full multi-turn transcript, including prompt and
execution feedback, but reward scoring now extracts the latest valid
`<thought>/<action>` response before calling `score_response`.

The first two global steps completed and entered the third rollout:

```text
step 1: format_match_rate=0.71875, no_error_rate=0.5625, exec_match_rate=0.5, self_correction_rate=0.21875
step 2: format_match_rate=0.875, no_error_rate=0.6875, exec_match_rate=0.46875, self_correction_rate=0.15625
```

This confirms the prompt/transcript scoring bug is fixed in the GRPO agent path.
Remaining work is throughput and timeout-noise tuning rather than basic reward
connectivity.

## 10. Formal 4-GPU GRPO Run

### Status

Run `logs/run_20260427_054006_grpo` was launched as the formal 4-GPU GRPO run
from the SFT checkpoint:

```text
--actor.model_name_or_path checkpoints/qwen2.5-coder-3b-logtir-sft
--actor.num_gpus_per_node 3
--ref.num_gpus_per_node 3
--vllm.num_engines 1
--vllm.tensor_parallel_size 1
--actor.entropy_coef 0
```

The layout is:

- GPUs 0-2: colocated actor/reference ranks.
- GPU 3: vLLM rollout engine.

`--actor.entropy_coef 0` is intentional: it enables entropy logging without
adding an entropy loss term to the PPO objective.

### Logging Check

The first global step completed and `logs/latest/metrics.jsonl` contains the
fields needed for post-run diagnosis:

```text
step=1
rollout/reward_mean=0.6604166626930237
rollout/response_length_mean=153.7916717529297
entropy_loss=0.36055440259364047
exec_match_rate=0.5000000111758709
no_error_rate=0.541666679084301
format_match_rate=0.7500000111758709
```

The run then completed a second global step and continued into the third
rollout:

```text
step=2
rollout/reward_mean=0.43958330154418945
rollout/response_length_mean=177.0416717529297
entropy_loss=0.32936348532814974
exec_match_rate=0.2916666716337204
no_error_rate=0.4583333395421505
format_match_rate=0.9166666716337204
```

The offline wandb run for this launch is stored locally under:

```text
wandb/offline-run-20260427_054257-sjussly9
```

Wandb online preflight timed out, but training continued because
`WANDB_MODE=offline`.

### Launch Note

An earlier 4-GPU attempt wrote only a locale warning to
`logs/grpo_launcher_4gpu_20260427_053119.log` and did not create a run
directory because it was started as a plain background job from a short-lived
shell. Use `setsid` or another detached launcher for long training runs.

## Operational Notes

- For long domestic-server GRPO runs, keep `WANDB_MODE=offline` unless preflight confirms a stable online wandb connection.
- Export loopback proxy variables before training if the shell contains stale proxy addresses.
- For 3 actor/reference GPUs with micro batch size 1, prefer train batch sizes divisible by 3, such as 30 or 60.
- With 3 actor ranks, keep `GRPO_PACKING_SAMPLES=0` and `GRPO_DYNAMIC_BATCH_ENABLE=0` unless OpenRLHF's dispatcher is patched to tolerate short packed batches.
- The current formal layout is 3 actor/reference GPUs plus 1 vLLM GPU; use the 1 actor/reference + 1 vLLM layout only for short stability checks.
- For multi-turn GRPO, verify early samples do not get format failures from prompt or transcript text being included in the scored response.
- If valid simple SQL receives `TimeoutExpired` during GRPO but not direct sandbox calls, reduce rollout concurrency before raising `LOGTIR_AGENT_TIMEOUT`.
- When verifying Python, use the exact interpreter path from the training `PATH`, not a separate `conda run` invocation.
- Before changing reward or rollout logic, continue to run the Spider gold-SQL evaluator and focused reward/rollout tests.
