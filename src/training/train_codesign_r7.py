"""
R7 co-design retraining launcher: identical to the r6 launcher (train_codesign_r6.py:
DummyVecEnv, CSV logging, shield-intervention logging, run_config.json) EXCEPT for the
pre-registered dual-ascent changes in results/r7_codesign/PREREG.md:

  * rebuffer-ratio target          0.05        -> 0.01
  * lambda_r range                 [10, 400]   -> [0.5, 400]
  * lambda_s range                 [0.3, 5.0]  -> [0.1, 50]
  * dual step eta_r                40          -> 12
  * dual step eta_s                0.05        -> 0.003
  (smoothness target 3.5, inits 60 / 0.7, asymmetric step multipliers, 60 warm-up episodes,
   100-episode window, reward form, PPO, envs, shield config: all unchanged.)

Diagnostics only (do not change training): the wrapper also exposes the dual's own 100-episode
window means in info (dual_mean_rebuf, dual_mean_smooth, dual_n_updates), and DualAllEnvLogger
writes <log_root>/dual_all_envs.csv with the mean/min/max of lambda and of those window means
over ALL 8 envs plus the fraction of envs sitting at a bound. (The legacy dual_variables.csv
only samples env 0 mid-episode, which is too noisy to judge constraint satisfaction.)

Usage (from new/):
  ABR_STABLE_VBR=1 python src/training/train_codesign_r7.py --models proposed_cps --seeds 0 \
      --num-envs 8 --trace-dir data/standardized/train_traces_5g_v18 \
      --test-trace-dir data/standardized/train_traces_5g_v18 --tag master_v18_5gtrain_r7
"""
import json
import os
import sys
import time
from pathlib import Path

os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
sys.path.append(str(Path(__file__).parent.parent.parent))

import numpy as np
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.vec_env import DummyVecEnv

import src.training.train_all_models_v18 as tm
import src.training.train_codesign_r6 as r6
from src.training.constrained_abr_v14 import LagrangianRewardWrapperV14

R7_DUAL = dict(
    rebuf_target=0.01,
    smooth_target=3.5,
    dual_lr_rebuf=12.0,
    dual_lr_smooth=0.003,
    lambda_rebuf_init=60.0,
    lambda_smooth_init=0.7,
    lambda_rebuf_range=(0.5, 400.0),
    lambda_smooth_range=(0.1, 50.0),
    buffer_dev_weight=0.03,
    lyapunov_weight=0.4,
    warmup_episodes=60,
)


class LagrangianRewardWrapperR7(LagrangianRewardWrapperV14):
    """V14 wrapper with the r7 dual settings; adds read-only dual diagnostics to info."""

    def __init__(self, env, rebuf_target=None, smooth_target=None, **_ignored):
        kw = dict(R7_DUAL)
        super().__init__(env, **kw)

    def step(self, action):
        obs, reward, terminated, truncated, info = super().step(action)
        w = min(100, len(self._recent_rebuf))
        if w > 0:
            info["dual_mean_rebuf"] = float(np.mean(self._recent_rebuf[-w:]))
            info["dual_mean_smooth"] = float(np.mean(self._recent_smooth[-w:]))
        info["dual_n_updates"] = max(0, self._episode_count - self.warmup_episodes + 1)
        return obs, reward, terminated, truncated, info


class DualAllEnvLogger(BaseCallback):
    COLS = ("lambda_rebuf", "lambda_smooth", "dual_mean_rebuf", "dual_mean_smooth")

    def __init__(self, log_dir, log_freq=10_000):
        super().__init__()
        self.path = Path(log_dir) / "dual_all_envs.csv"
        self.log_freq = log_freq
        self._next = log_freq

    def _on_training_start(self):
        self.path.parent.mkdir(parents=True, exist_ok=True)
        hdr = ["timesteps", "n_envs", "min_dual_updates"]
        for c in self.COLS:
            hdr += [f"{c}_mean", f"{c}_min", f"{c}_max"]
        hdr += ["frac_lr_at_lo", "frac_lr_at_hi", "frac_ls_at_lo", "frac_ls_at_hi"]
        with open(self.path, "w") as f:
            f.write(",".join(hdr) + "\n")

    def _on_step(self):
        if self.num_timesteps < self._next:
            return True
        self._next += self.log_freq
        infos = self.locals.get("infos", [])
        if not infos or "lambda_rebuf" not in infos[0]:
            return True
        row = [self.num_timesteps, len(infos), min(int(i.get("dual_n_updates", 0)) for i in infos)]
        for c in self.COLS:
            v = np.array([float(i.get(c, np.nan)) for i in infos])
            row += [np.nanmean(v), np.nanmin(v), np.nanmax(v)] if np.isfinite(v).any() else [np.nan] * 3
        lr = np.array([i["lambda_rebuf"] for i in infos]); ls = np.array([i["lambda_smooth"] for i in infos])
        (rlo, rhi), (slo, shi) = R7_DUAL["lambda_rebuf_range"], R7_DUAL["lambda_smooth_range"]
        row += [np.mean(np.isclose(lr, rlo)), np.mean(np.isclose(lr, rhi)),
                np.mean(np.isclose(ls, slo)), np.mean(np.isclose(ls, shi))]
        with open(self.path, "a") as f:
            f.write(",".join(f"{x:.6g}" if isinstance(x, float) else str(x) for x in row) + "\n")
        return True


def _patched_cbl(cbs):
    log_dir = None
    for cb in cbs:
        if type(cb).__name__ == "ConstraintDiagnosticsLogger":
            log_dir = str(cb.log_dir)
    if log_dir is None:
        for cb in cbs:
            if getattr(cb, "log_path", None):
                log_dir = str(Path(cb.log_path).parent)
    cbs = list(cbs) + [r6.ShieldInterventionLogger(str(log_dir)), DualAllEnvLogger(str(log_dir))]
    return r6._ORIG_CBL(cbs)


def write_run_config(args):
    cps = tm.make_cps_cfg()
    for mk in args.models:
        spec = tm.MODEL_SPECS[mk]
        for s in args.seeds:
            root = tm.PATHS["models"] / args.tag / spec["folder"] / f"seed_{s}"
            root.mkdir(parents=True, exist_ok=True)
            cfg = {
                "model_key": mk, "seed": s, "tag": args.tag, "git_rev": r6._git_rev(),
                "launcher": "train_codesign_r7.py", "prereg": "results/r7_codesign/PREREG.md",
                "host": __import__("platform").node(), "started": time.strftime("%Y-%m-%d %H:%M:%S %z"),
                "ABR_STABLE_VBR": os.environ.get("ABR_STABLE_VBR"),
                "train_trace_dir": args.trace_dir, "eval_trace_dir": args.test_trace_dir,
                "train_videos": tm.Config.TRAIN_VIDEOS, "in_training_eval_videos": tm.Config.TEST_VIDEOS,
                "max_chunks": tm.Config.MAX_CHUNKS, "num_envs": args.num_envs, "vec_env": "DummyVecEnv",
                "timesteps": int(spec["timesteps"] * args.timesteps_scale), "spec": spec,
                "ppo": {"lr": "linear 3e-4 -> 1e-5", "n_steps": tm.Config.N_STEPS,
                        "batch_size": tm.Config.BATCH_SIZE, "n_epochs": tm.Config.N_EPOCHS,
                        "gamma": tm.Config.GAMMA, "gae_lambda": tm.Config.GAE_LAMBDA,
                        "clip_range": tm.Config.CLIP_RANGE, "vf_coef": tm.Config.VF_COEF,
                        "max_grad_norm": tm.Config.MAX_GRAD_NORM, "ent_coef": spec["ent_coef"],
                        "net_arch": "pi/vf [256,256] tanh"},
                "lagrangian_r7": R7_DUAL if spec.get("use_lagrangian") else None,
                "cps_train_cfg": (repr(cps) if spec.get("cps_train") else None),
            }
            with open(root / "run_config.json", "w") as f:
                json.dump(cfg, f, indent=2, default=str)


def main():
    tm.SubprocVecEnv = DummyVecEnv
    tm.CallbackList = _patched_cbl
    tm.LagrangianRewardWrapperV14 = LagrangianRewardWrapperR7
    tm.REBUF_TARGET = R7_DUAL["rebuf_target"]       # used by ConstraintDiagnosticsLogger
    tm.SMOOTH_TARGET = R7_DUAL["smooth_target"]
    PPO.learn = r6._patched_learn
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
