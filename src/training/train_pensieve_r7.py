"""
r7 Pensieve retraining launcher (pre-registered: results/r7_pensieve/PREREG.md).

Thin wrapper around train_all_models_v18.train_one_model, like train_codesign_r7.py (DummyVecEnv,
CSV logging, run_config.json). Adds ONE model spec, `pensieve_r7`, which differs from the legacy
`pensieve` spec only in:
  * use_future=True so the next-chunk sizes are in the observation (Pensieve observes them), and
  * the Pensieve-faithful mask (src/environment/pensieve_state.py) instead of obs[15:]=0;
    evaluation uses the same mask (eval_certified_shield_v18.py --blind --blind-mode pensieve).
Trace pools / videos / ABR_STABLE_VBR are set on the command line, identical to r7 co-design.
Everything else (plain v18 reward with REBUF_PENALTY_BASE=100, no Lyapunov, no Lagrangian, no
shield, PPO hyper-parameters, ent_coef 0.03, 1M steps, 8 envs x 4096) is unchanged.

Usage (from new/):
  ABR_STABLE_VBR=1 python src/training/train_pensieve_r7.py --seeds 0 --num-envs 8 \
      --trace-dir data/standardized/train_traces_5g_v18 \
      --test-trace-dir data/standardized/train_traces_5g_v18 --tag master_v18_5g_pensieve_r7
"""
import json
import os
import platform
import sys
import time
from pathlib import Path

os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
sys.path.append(str(Path(__file__).parent.parent.parent))

from gymnasium import ObservationWrapper
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv

import src.training.train_all_models_v18 as tm
import src.training.train_codesign_r6 as r6
from src.environment.pensieve_state import PENSIEVE_MASK, pensieve_obs

tm.MODEL_SPECS["pensieve_r7"] = {
    "folder": "pensieve_r7", "use_lyapunov": False, "use_future": True,
    "blind_features": True, "use_lagrangian": False, "shield": "off",
    "penalty": False, "hysteresis": False, "ent_coef": 0.03, "timesteps": 1_000_000,
}


class PensieveStateWrapper(ObservationWrapper):
    def __init__(self, env):
        super().__init__(env)
        self.observation_space = env.observation_space

    def observation(self, obs):
        return pensieve_obs(obs)


def write_run_config(args):
    spec = tm.MODEL_SPECS["pensieve_r7"]
    for s in args.seeds:
        root = tm.PATHS["models"] / args.tag / spec["folder"] / f"seed_{s}"
        root.mkdir(parents=True, exist_ok=True)
        cfg = {
            "model_key": "pensieve_r7", "seed": s, "tag": args.tag, "git_rev": r6._git_rev(),
            "launcher": "train_pensieve_r7.py", "prereg": "results/r7_pensieve/PREREG.md",
            "host": platform.node(), "started": time.strftime("%Y-%m-%d %H:%M:%S %z"),
            "ABR_STABLE_VBR": os.environ.get("ABR_STABLE_VBR"),
            "train_trace_dir": args.trace_dir, "eval_trace_dir": args.test_trace_dir,
            "train_videos": tm.Config.TRAIN_VIDEOS, "in_training_eval_videos": tm.Config.TEST_VIDEOS,
            "max_chunks": tm.Config.MAX_CHUNKS, "num_envs": args.num_envs, "vec_env": "DummyVecEnv",
            "timesteps": int(spec["timesteps"] * args.timesteps_scale), "spec": spec,
            "obs_mask_zeroed": [int(i) for i in PENSIEVE_MASK],
            "reward": "v18 env reward: (VMAF - 100*rebuf - |dVMAF| - 0.03*max(0,B_target-B))/100, no Lyapunov",
            "ppo": {"lr": "linear 3e-4 -> 1e-5", "n_steps": tm.Config.N_STEPS,
                    "batch_size": tm.Config.BATCH_SIZE, "n_epochs": tm.Config.N_EPOCHS,
                    "gamma": tm.Config.GAMMA, "gae_lambda": tm.Config.GAE_LAMBDA,
                    "clip_range": tm.Config.CLIP_RANGE, "vf_coef": tm.Config.VF_COEF,
                    "max_grad_norm": tm.Config.MAX_GRAD_NORM, "ent_coef": spec["ent_coef"],
                    "net_arch": "pi/vf [256,256] tanh"},
        }
        with open(root / "run_config.json", "w") as f:
            json.dump(cfg, f, indent=2, default=str)


def main():
    tm.SubprocVecEnv = DummyVecEnv
    tm.ContentBlindWrapper = PensieveStateWrapper
    PPO.learn = r6._patched_learn
    import argparse
    p = argparse.ArgumentParser()
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
    for s in args.seeds:
        tm.train_one_model("pensieve_r7", seed=s, timesteps_scale=args.timesteps_scale,
                           num_envs=args.num_envs, train_dir=args.trace_dir,
                           test_dir=args.test_trace_dir, model_tag=args.tag)


if __name__ == "__main__":
    main()
