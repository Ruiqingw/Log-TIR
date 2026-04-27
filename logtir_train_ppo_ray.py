from __future__ import annotations

import faulthandler
import os
import runpy
import signal
import sys

from openrlhf_ppo_logging_patch import apply_patch


def _patch_ray_init() -> None:
    import ray

    original_init = ray.init

    def init_with_logtir_defaults(*args, **kwargs):
        kwargs.setdefault("include_dashboard", False)
        kwargs.setdefault("address", "local")
        if "num_cpus" not in kwargs and os.environ.get("LOGTIR_RAY_NUM_CPUS"):
            kwargs["num_cpus"] = int(os.environ["LOGTIR_RAY_NUM_CPUS"])
        if "num_gpus" not in kwargs and os.environ.get("LOGTIR_RAY_NUM_GPUS"):
            kwargs["num_gpus"] = int(os.environ["LOGTIR_RAY_NUM_GPUS"])
        if (
            "_temp_dir" not in kwargs
            and "address" not in kwargs
            and os.environ.get("RAY_TMPDIR")
        ):
            kwargs["_temp_dir"] = os.environ["RAY_TMPDIR"]
        return original_init(*args, **kwargs)

    ray.init = init_with_logtir_defaults


if __name__ == "__main__":
    faulthandler.enable(file=sys.stderr, all_threads=True)
    faulthandler.register(signal.SIGUSR1, file=sys.stderr, all_threads=True)
    apply_patch()
    _patch_ray_init()
    runpy.run_module("openrlhf.cli.train_ppo_ray", run_name="__main__")
