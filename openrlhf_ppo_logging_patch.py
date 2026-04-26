from __future__ import annotations

from typing import Any


def apply_patch() -> None:
    from openrlhf.trainer.ppo_trainer import BasePPOTrainer
    from openrlhf.utils.logging_utils import WandbLogger

    if getattr(BasePPOTrainer, "_logtir_logging_patch_applied", False):
        return

    original_rollout_stats = BasePPOTrainer._compute_rollout_stats
    original_train_step = BasePPOTrainer.train_step
    original_wandb_init = WandbLogger.__init__
    original_log_eval = WandbLogger.log_eval

    def _mean_info(experiences: list[Any], key: str) -> float | None:
        import torch

        values = [exp.info[key].float().flatten() for exp in experiences if key in exp.info]
        if not values:
            return None
        return torch.cat(values).mean().item()

    def _compute_rollout_stats(self: Any, experiences: list[Any]) -> dict[str, float]:
        stats = original_rollout_stats(self, experiences)
        stats["reward_mean"] = stats.get("rollout/reward_mean", 0.0)
        stats["reward_std"] = stats.get("rollout/reward_std", 0.0)
        stats["avg_response_length"] = stats.get("rollout/response_length_mean", 0.0)

        info_keys: set[str] = set()
        for exp in experiences:
            info_keys.update(exp.info.keys())

        skipped = {
            "reward",
            "score",
            "response_clip_ratio",
            "kl",
            "logprobs_diff",
            "group_reward_std",
            "return",
        }
        for key in sorted(info_keys - skipped):
            value = _mean_info(experiences, key)
            if value is not None:
                stats[key] = value
        return stats

    def train_step(self: Any, rollout_samples: Any, global_step: int) -> tuple[dict[str, Any], int]:
        status, next_step = original_train_step(self, rollout_samples, global_step)
        if "kl" in status and "kl_div" not in status:
            status["kl_div"] = status["kl"]
        return status, next_step

    def wandb_init(self: Any, args: Any) -> None:
        original_wandb_init(self, args)
        # OpenRLHF binds eval/* to eval/epoch but logs eval/global_step.
        # Bind eval curves to the step that is actually emitted.
        self.handle.define_metric("eval/global_step")
        self.handle.define_metric("eval/*", step_metric="eval/global_step", step_sync=True)

    def log_eval(self: Any, global_step: int, logs_dict: dict[str, Any]) -> None:
        logs_dict = dict(logs_dict)
        logs_dict["global_step"] = global_step
        logs_dict["epoch"] = global_step
        original_log_eval(self, global_step, logs_dict)

    BasePPOTrainer._compute_rollout_stats = _compute_rollout_stats
    BasePPOTrainer.train_step = train_step
    WandbLogger.__init__ = wandb_init
    WandbLogger.log_eval = log_eval
    BasePPOTrainer._logtir_logging_patch_applied = True
