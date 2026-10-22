"""
R6 co-design retraining launcher (thin wrapper around train_all_models_v18).

Training hyper-parameters are NOT changed; this only
  * uses DummyVecEnv (same 8 envs, same seeds, same rollout size 8x4096) so a
    job occupies one core instead of 9 processes (shared-server friendly),
  * disables the tqdm/rich progress bar and logs SB3 metrics to CSV
    (<log_root>/sb3/progress.csv) instead of tensorboard,
  * adds ShieldInterventionLogger -> <log_root>/shield_interventions.csv
    (per-chunk shield intervention / banking / downgrade counts, so co-design
    activity is verifiable),
  * writes <model_root>/run_config.json with the exact training config.

Usage (from new/):
  ABR_STABLE_VBR=1 python src/training/train_codesign_r6.py \
      --models proposed_cps --seeds 0 --num-envs 8 \
      --trace-dir data/standardized/train_traces_5g_v18 \
      --test-trace-dir data/standardized/train_traces_5g_v18 \
      --tag master_v18_5gtrain
"""
import inspect
import json
import os
import platform
import subprocess
import sys
import time
from pathlib import Path

os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")

sys.path.append(str(Path(__file__).parent.parent.parent))

from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import BaseCallback, CallbackList
from stable_baselines3.common.vec_env import DummyVecEnv

import src.training.train_all_models_v18 as tm
from src.training.constrained_abr_v14 import LagrangianRewardWrapperV14


class ShieldInterventionLogger(BaseCallback):
    def __init__(self, log_dir: str, log_freq: int = 10_000):
        super().__init__()
        self.path = Path(log_dir) / "shield_interventions.csv"
        self.log_freq = log_freq
        self.reset_window()
        self.cum_steps = self.cum_int = 0
        self.t0 = time.time()

    def reset_window(self):
        self.w_steps = self.w_int = self.w_bank = self.w_down = self.w_emerg = 0
        self.w_rebuf = 0.0
        self.w_q = []

    def _on_training_start(self):
        self.path.parent.mkdir(parents=True, exist_ok=True)
        with open(self.path, "w") as f:
            f.write("timesteps,elapsed_s,fps,window_chunks,interv_rate,bits_saved_rate,"
                    "downgrade_rate,emergency_rate,cum_interventions,cum_interv_rate,"
                    "lambda_rebuf,lambda_smooth\n")

    def _on_step(self):
        for info in self.locals.get("infos", []):
            self.w_steps += 1
            iv = int(info.get("shield_intervened", 0))
            self.w_int += iv
            self.cum_int += iv
            self.w_bank += int(info.get("banked_bits", 0.0) > 0)
            self.w_down += int(info.get("safety_downgrade", 0))
            self.w_emerg += int(info.get("emergency", 0))
        self.cum_steps += len(self.locals.get("infos", []))
        if self.w_steps >= self.log_freq:
            info0 = self.locals["infos"][0]
            el = time.time() - self.t0
            with open(self.path, "a") as f:
                f.write(f"{self.num_timesteps},{el:.0f},{self.num_timesteps/max(el,1e-9):.1f},"
                        f"{self.w_steps},{self.w_int/self.w_steps:.4f},{self.w_bank/self.w_steps:.4f},"
                        f"{self.w_down/self.w_steps:.4f},{self.w_emerg/self.w_steps:.4f},"
                        f"{self.cum_int},{self.cum_int/max(self.cum_steps,1):.4f},"
                        f"{info0.get('lambda_rebuf','')},{info0.get('lambda_smooth','')}\n")
            self.reset_window()
        return True


def _git_rev():
    try:
        return subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip()
    except Exception:
        return "unknown"


_ORIG_CBL = tm.CallbackList
_ORIG_LEARN = PPO.learn


def _patched_cbl(cbs):
    log_dir = None
    for cb in cbs:
        if hasattr(cb, "log_path") and getattr(cb, "log_path", None):
            log_dir = str(Path(cb.log_path).parent)
    for cb in cbs:
        if type(cb).__name__ == "ConstraintDiagnosticsLogger":
            log_dir = getattr(cb, "log_dir", log_dir)
    cbs = list(cbs) + [ShieldInterventionLogger(str(log_dir))]
    return _ORIG_CBL(cbs)


def _patched_learn(self, *a, **kw):
    kw["progress_bar"] = False
    # SB3 metrics -> <log_root>/sb3/progress.csv (+stdout); no tensorboard dependency.
    from stable_baselines3.common.logger import configure
    d = Path(self.tensorboard_log or ".") / "sb3"
    self.set_logger(configure(str(d), ["stdout", "csv"]))
    return _ORIG_LEARN(self, *a, **kw)


def write_run_config(args):
    sig = inspect.signature(LagrangianRewardWrapperV14.__init__)
    lag = {k: (v.default if v.default is not inspect._empty else None)
           for k, v in sig.parameters.items() if k not in ("self", "env")}
    lag["rebuf_target"] = tm.REBUF_TARGET
    lag["smooth_target"] = tm.SMOOTH_TARGET
    try:
        from src.environment import abr_multi_env_v18 as e18
        env_consts = {k: getattr(e18.ABREnv, k) for k in dir(e18.ABREnv)
                      if k.isupper() and isinstance(getattr(e18.ABREnv, k), (int, float, str, list, tuple))}
    except Exception as exc:  # pragma: no cover
        env_consts = {"error": repr(exc)}
    cps = tm.make_cps_cfg()
    for mk in args.models:
        spec = tm.MODEL_SPECS[mk]
        for s in args.seeds:
            root = tm.PATHS["models"] / args.tag / spec["folder"] / f"seed_{s}"
            root.mkdir(parents=True, exist_ok=True)
            cfg = {
                "model_key": mk, "seed": s, "tag": args.tag, "git_rev": _git_rev(),
                "host": platform.node(), "started": time.strftime("%Y-%m-%d %H:%M:%S %z"),
                "ABR_STABLE_VBR": os.environ.get("ABR_STABLE_VBR"),
                "train_trace_dir": args.trace_dir, "eval_trace_dir": args.test_trace_dir,
                "train_videos": tm.Config.TRAIN_VIDEOS, "in_training_eval_videos": tm.Config.TEST_VIDEOS,
                "max_chunks": tm.Config.MAX_CHUNKS, "num_envs": args.num_envs, "vec_env": "DummyVecEnv",
                "timesteps": int(spec["timesteps"] * args.timesteps_scale),
                "spec": spec,
                "ppo": {"lr": "linear 3e-4 -> 1e-5", "n_steps": tm.Config.N_STEPS,
                        "batch_size": tm.Config.BATCH_SIZE, "n_epochs": tm.Config.N_EPOCHS,
                        "gamma": tm.Config.GAMMA, "gae_lambda": tm.Config.GAE_LAMBDA,
                        "clip_range": tm.Config.CLIP_RANGE, "vf_coef": tm.Config.VF_COEF,
                        "max_grad_norm": tm.Config.MAX_GRAD_NORM, "ent_coef": spec["ent_coef"],
                        "net_arch": "pi/vf [256,256] tanh"},
                "lagrangian_v14": lag if spec.get("use_lagrangian") else None,
                "cps_train_cfg": (repr(cps) if spec.get("cps_train") else None),
                "env_constants_v18": env_consts,
            }
            with open(root / "run_config.json", "w") as f:
                json.dump(cfg, f, indent=2, default=str)


def main():
    tm.SubprocVecEnv = DummyVecEnv
    tm.CallbackList = _patched_cbl
    PPO.learn = _patched_learn
    import argparse
    p = argparse.ArgumentParser()
    p.add_argument("--models", type=tm.parse_model_list, required=True)
    p.add_argument("--seeds", type=tm.parse_seed_list, default=[0])
    p.add_argument("--timesteps-scale", type=float, default=1.0)
    p.add_argument("--num-envs", type=int, default=8)
    p.add_argument("--torch-threads", type=int, default=1)
    p.add_argument("--trace-dir", required=True)
    p.add_argument("--test-trace-dir", required=True)
    p.add_argument("--tag", required=True)
    args = p.parse_args()
    import torch
    torch.set_num_threads(max(1, args.torch_threads))
    write_run_config(args)
    for k in args.models:
        for s in args.seeds:
            tm.train_one_model(k, seed=s, timesteps_scale=args.timesteps_scale,
                               num_envs=args.num_envs, train_dir=args.trace_dir,
                               test_dir=args.test_trace_dir, model_tag=args.tag)


if __name__ == "__main__":
    main()
