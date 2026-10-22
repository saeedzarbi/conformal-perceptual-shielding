"""
Video-level (cluster-aware) statistics for the CPS paired evaluations (r6).

Episodes are clustered by title (12 videos): episodes of the same title share the
VMAF ladder and VBR profile, so per-episode tests treat ~204 correlated draws as
independent. This script re-analyses existing ``episodes.csv`` files with the
TITLE as the unit of analysis:

  * per-video mean paired difference (treat - ref) for each metric,
  * Wilcoxon signed-rank on the N=12 per-video means (exact, two-sided),
  * exact cluster sign-flip permutation test (2^12 flips of whole videos) on the
    mean of per-video means,
  * cluster bootstrap 95% CI (resampling videos),
  * video-level TOST of the VMAF difference within +/- epsilon (t, df=N-1),
  * count of videos where the difference has the reported sign.

The episode -> video map is read from the ``video`` column when present (r6 eval
writes it); otherwise it is reconstructed by replaying ``env.reset(seed=1000+ep)``
exactly as the evaluation does (video draw is the first draw of the seeded
generator, independent of the trace pool and of PYTHONHASHSEED).

Usage (from new/):
  python src/evaluation/video_level_stats.py \
      --runs results/v18_certified/greedy_5g results/v18_certified/bba_5g \
      --pairs certified:safety,certified:raw --out results/v18_certified/video_level
"""

from __future__ import annotations

import argparse
import itertools
import json
import random
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.append(str(Path(__file__).resolve().parents[2]))

METRICS = ("bitrate_mean_kbps", "rebuffer_total", "vmaf_mean", "stallfree")


def episode_video_map(n_episodes: int, trace_dir: str | None = None) -> pd.Series:
    from configs.paths import get_paths
    from src.evaluation.eval_certified_shield_v18 import build_base_env
    P = get_paths()
    td = trace_dir or str(Path(P["data_dir"]) / "standardized" / "test_traces_5g_v18")
    env = build_base_env(td, 12.0, False)
    vids = []
    for ep in range(n_episodes):
        random.seed(1000 + ep)
        env.reset(seed=1000 + ep)
        vids.append(env.current_video_name)
    return pd.Series(vids, index=range(n_episodes), name="video")


def _wilcoxon(x):
    from scipy.stats import wilcoxon
    x = np.asarray(x, float)
    if np.allclose(x, 0):
        return float("nan"), 1.0
    try:
        r = wilcoxon(x, alternative="two-sided", method="exact")
    except TypeError:
        r = wilcoxon(x, alternative="two-sided")
    return float(r.statistic), float(r.pvalue)


def cluster_signflip_p(vm: np.ndarray) -> float:
    """Exact two-sided sign-flip test on the mean of per-video mean differences."""
    vm = np.asarray(vm, float)
    n = len(vm)
    obs = abs(vm.mean())
    signs = np.array(list(itertools.product([-1.0, 1.0], repeat=n)))
    perm = np.abs((signs * vm).mean(axis=1))
    return float((perm >= obs - 1e-12).mean())


def cluster_bootstrap_ci(vm, n_boot=20000, seed=0):
    vm = np.asarray(vm, float)
    rng = np.random.default_rng(seed)
    means = vm[rng.integers(0, len(vm), size=(n_boot, len(vm)))].mean(axis=1)
    return float(np.percentile(means, 2.5)), float(np.percentile(means, 97.5))


def tost(vm, margin):
    from scipy.stats import t
    vm = np.asarray(vm, float)
    n = len(vm)
    md, sd = float(vm.mean()), float(vm.std(ddof=1))
    if sd == 0:
        return 0.0 if abs(md) < margin else 1.0
    se = sd / np.sqrt(n)
    p_low = 1 - t.cdf((md + margin) / se, df=n - 1)
    p_high = t.cdf((md - margin) / se, df=n - 1)
    return float(max(p_low, p_high))


def analyse(run_dir: Path, pairs, epsilon, vmap_cache):
    df = pd.read_csv(run_dir / "episodes.csv")
    n_ep = int(df["episode"].max()) + 1
    if "video" not in df.columns or df["video"].isna().all():
        if n_ep not in vmap_cache:
            vmap_cache[n_ep] = episode_video_map(n_ep)
        df["video"] = df["episode"].map(vmap_cache[n_ep])
    out = []
    for treat, ref in pairs:
        if treat not in set(df.arm) or ref not in set(df.arm):
            continue
        a = df[df.arm == treat].set_index("episode")
        b = df[df.arm == ref].set_index("episode")
        for m in METRICS:
            d = (a[m] - b[m]).rename("d").to_frame()
            d["video"] = a["video"]
            ref_mean = b[m].groupby(a["video"]).mean()
            per_vid = d.groupby("video")["d"].mean()
            vm = per_vid.values
            _, p_w = _wilcoxon(vm)
            p_perm = cluster_signflip_p(vm)
            lo, hi = cluster_bootstrap_ci(vm)
            ep_mean = float(d["d"].mean())
            row = {
                "run": run_dir.name, "treat": treat, "ref": ref, "metric": m,
                "n_videos": int(len(vm)), "n_episodes": int(len(d)),
                "episode_mean_diff": ep_mean,
                "video_mean_diff": float(vm.mean()),
                "video_ci95_lo": lo, "video_ci95_hi": hi,
                "video_wilcoxon_p": p_w, "video_signflip_p": p_perm,
                "videos_negative": int((vm < 0).sum()), "videos_positive": int((vm > 0).sum()),
                "videos_zero": int((vm == 0).sum()),
            }
            if m in ("bitrate_mean_kbps", "rebuffer_total"):
                # relative change, title-weighted (mean of per-video % changes where ref>0)
                rel = 100.0 * per_vid / ref_mean.replace(0, np.nan)
                row["video_mean_pct_change"] = float(np.nanmean(rel.values))
                row["episode_pct_change"] = float(100.0 * ep_mean / max(b[m].mean(), 1e-12))
            if m == "vmaf_mean":
                row["video_tost_p_equiv"] = tost(vm, epsilon)
            out.append(row)
            if m in ("bitrate_mean_kbps", "rebuffer_total", "vmaf_mean"):
                for v, val in per_vid.items():
                    out.append({"run": run_dir.name, "treat": treat, "ref": ref,
                                "metric": m + "@video", "video": v,
                                "video_mean_diff": float(val),
                                "n_episodes": int((d["video"] == v).sum())})
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs", nargs="+", required=True)
    ap.add_argument("--pairs", default="certified:safety,certified:raw")
    ap.add_argument("--epsilon", type=float, default=1.0)
    ap.add_argument("--out", default="results/v18_certified/video_level")
    args = ap.parse_args()
    pairs = [tuple(p.split(":")) for p in args.pairs.split(",") if p]
    cache: dict = {}
    rows = []
    for r in args.runs:
        rows += analyse(Path(r), pairs, args.epsilon, cache)
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    full = pd.DataFrame(rows)
    full.to_csv(out / "video_level_stats_long.csv", index=False)
    main_tbl = full[~full["metric"].str.contains("@")].copy()
    main_tbl.to_csv(out / "video_level_stats.csv", index=False)
    cols = ["run", "treat", "ref", "metric", "episode_mean_diff", "video_mean_diff",
            "video_ci95_lo", "video_ci95_hi", "video_wilcoxon_p", "video_signflip_p",
            "videos_negative", "videos_positive"]
    extra = [c for c in ("episode_pct_change", "video_mean_pct_change", "video_tost_p_equiv")
             if c in main_tbl.columns]
    with pd.option_context("display.width", 250, "display.max_columns", 30,
                           "display.float_format", "{:.4g}".format):
        print(main_tbl[cols + extra].to_string(index=False))
    print(f"\nwrote {out}/video_level_stats.csv and video_level_stats_long.csv")


if __name__ == "__main__":
    main()
