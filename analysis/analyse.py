"""review8 analysis: arm means (abs), post-warm-up coverage, smoothness, title-level contrasts
vs Safety (and Safety vs baselines) with Holm-Bonferroni per table family."""
import sys, glob
from pathlib import Path
import numpy as np, pandas as pd
sys.path.insert(0, ".")
from src.evaluation.video_level_stats import _wilcoxon, cluster_bootstrap_ci, tost

def holm(p):
    p = np.asarray(p, float); m = len(p); o = np.argsort(p); adj = np.empty(m); run = 0
    for i, j in enumerate(o):
        run = max(run, min(1.0, (m - i) * p[j])); adj[j] = run
    return adj

def means(run_dirs):
    rows = []
    for d in run_dirs:
        e = pd.read_csv(Path(d) / "episodes.csv")
        for arm, g in e.groupby("arm", sort=False):
            rows.append(dict(run=Path(d).name, arm=arm, reb=g.rebuffer_total.mean(), vmaf=g.vmaf_mean.mean(),
                br=g.bitrate_mean_kbps.mean(), smooth=g.smooth_kbps.mean(), stallfree=g.stallfree.mean(),
                cov=(g.cov_hits.sum() / g.cov_n.sum()) if g.cov_n.sum() else np.nan,
                cov_pw=(g.cov_pw_hits.sum() / g.cov_pw_n.sum()) if g.cov_pw_n.sum() else np.nan))
    return pd.DataFrame(rows)

def contrasts(run_dirs, pairs):
    rows = []
    for d in run_dirs:
        e = pd.read_csv(Path(d) / "episodes.csv")
        for t, r in pairs:
            if t not in set(e.arm) or r not in set(e.arm): continue
            a = e[e.arm == t].set_index("episode"); b = e[e.arm == r].set_index("episode")
            row = dict(run=Path(d).name, treat=t, ref=r)
            for m, k in [("bitrate_mean_kbps", "br"), ("rebuffer_total", "reb"), ("vmaf_mean", "vmaf"), ("smooth_kbps", "smooth")]:
                dd = (a[m] - b[m]).groupby(a.video).mean(); refm = b[m].groupby(a.video).mean()
                row[k + "_diff"] = dd.mean(); row[k + "_p"] = _wilcoxon(dd.values)[1]
                if m == "bitrate_mean_kbps": row["br_pct"] = float(np.nanmean(100 * dd / refm.replace(0, np.nan)))
                if m == "vmaf_mean":
                    row["vmaf_lo"], row["vmaf_hi"] = cluster_bootstrap_ci(dd.values); row["tost_p"] = tost(dd.values, 1.0)
            rows.append(row)
    return pd.DataFrame(rows)

if __name__ == "__main__":
    R = "analysis"
    runs = sorted(glob.glob(f"{R}/*_b30_warm_a05"))
    M = means(runs); M.to_csv(f"{R}/arm_means.csv", index=False)
    pairs = [(x, "safety") for x in ["certified", "certified_a", "certified_cf", "certified_a_cf"]] + \
            [("safety", x) for x in ["raw", "fixed_safety", "tfixed_safety", "aci01_safety", "aci05_safety"]]
    C = contrasts(runs, pairs)
    C["pool"] = C.run.str.split("_").str[0]
    # Holm families: per pool (= one table), per metric, over all host x contrast rows
    for k in ["br", "reb", "vmaf", "smooth"]:
        C[k + "_p_holm"] = C.groupby("pool")[k + "_p"].transform(lambda s: holm(s.values))
    C["tost_p_holm"] = C.groupby("pool")["tost_p"].transform(lambda s: holm(s.values))
    C.to_csv(f"{R}/contrasts_holm.csv", index=False)
    pd.set_option("display.width", 250); pd.set_option("display.max_rows", 500)
    print(M.round(3).to_string())
