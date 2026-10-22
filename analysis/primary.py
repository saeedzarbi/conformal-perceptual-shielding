"""review8b: primary family (graphical gatekeeping) + mixed-effects complement. See PREREG_stats.md."""
import sys, json, warnings
import numpy as np, pandas as pd
sys.path.insert(0, "."); sys.path.insert(0, "analysis")
from src.evaluation.video_level_stats import _wilcoxon, cluster_bootstrap_ci, tost
import statsmodels.formula.api as smf
warnings.filterwarnings("ignore")
R = "analysis"; HOSTS = ["greedy", "bba", "bola", "mpc"]
POOLS = {"synthetic": ["syn"], "measured": ["irish", "lumos"]}
ARMS = {"CPS-S": "certified", "CPS-A": "certified_a"}

def episode_diffs(arm, datasets):
    out = []
    for ds in datasets:
        per = []
        for h in HOSTS:
            e = pd.read_csv(f"{R}/{ds}_{h}_b30_warm_a05/episodes.csv")
            a = e[e.arm == arm].set_index("episode"); b = e[e.arm == "safety"].set_index("episode")
            assert (a.video == b.video).all() and (a.trace_idx == b.trace_idx).all()
            per.append(pd.DataFrame(dict(video=a.video, trace=ds + "_" + a.trace_idx.astype(str), host=h,
                br_t=a.bitrate_mean_kbps, br_s=b.bitrate_mean_kbps, reb=a.rebuffer_total - b.rebuffer_total,
                vmaf=a.vmaf_mean - b.vmaf_mean)))
        d = pd.concat(per); g = d.groupby(level=0)
        ep = pd.DataFrame(dict(video=g.video.first(), trace=g.trace.first(), br_t=g.br_t.mean(), br_s=g.br_s.mean(),
                               reb=g.reb.mean(), vmaf=g.vmaf.mean())); ep["ds"] = ds
        ep["br_pct"] = 100 * (ep.br_t - ep.br_s) / ep.br_s
        out.append(ep.reset_index(drop=True))
    return pd.concat(out, ignore_index=True)

def title_level(ep):
    t = ep.groupby("video")
    br = 100 * (t.br_t.mean() - t.br_s.mean()) / t.br_s.mean()
    return br.values, t.reb.mean().values, t.vmaf.mean().values

def graphical(p, w, G, alpha=0.05):
    """Bretz et al. (2009) sequentially rejective graph; returns adjusted p-values."""
    p = np.asarray(p, float); n = len(p); w = np.array(w, float); G = np.array(G, float)
    act = np.ones(n, bool); adj = np.ones(n); pmax = 0.0
    while act.any():
        ratio = np.where(act & (w > 0), p / np.where(w > 0, w, 1), np.inf)
        j = int(np.argmin(ratio)); pmax = max(pmax, min(1.0, ratio[j])); adj[j] = pmax
        act[j] = False
        w2 = w.copy(); G2 = G.copy()
        for l in range(n):
            if not act[l]: w2[l] = 0; continue
            w2[l] = w[l] + w[j] * G[j, l]
            for k in range(n):
                if not act[k] or k == l: G2[l, k] = 0; continue
                den = 1 - G[l, j] * G[j, l]
                G2[l, k] = (G[l, k] + G[l, j] * G[j, k]) / den if den > 0 else 0
        w, G = w2, G2
        if not np.isfinite(ratio[j]): adj[j] = 1.0
    return adj

def mixed(ep, col, measured):
    d = ep.copy(); d["g"] = 1
    f = f"{col} ~ 1" + (" + C(ds, Sum)" if measured else "")
    m = smf.mixedlm(f, d, groups="g", vc_formula={"title": "0 + C(video)", "trace": "0 + C(trace)"}).fit(reml=True)
    est = m.params["Intercept"]; se = m.bse["Intercept"]
    return dict(est=est, se=se, lo95=est - 1.96 * se, hi95=est + 1.96 * se, lo90=est - 1.645 * se,
                hi90=est + 1.645 * se, p=m.pvalues["Intercept"], n=len(d))

rows = []
for an, arm in ARMS.items():
    for pn, dss in POOLS.items():
        ep = episode_diffs(arm, dss); br, reb, vm = title_level(ep)
        r = dict(arm=an, pool=pn)
        for k, x in [("br", br), ("reb", reb), ("vmaf", vm)]:
            r[k] = x.mean(); r[k + "_lo"], r[k + "_hi"] = cluster_bootstrap_ci(x)
            sd = x.std(ddof=1); r[k + "_dz"] = x.mean() / sd if sd > 0 else np.nan
        r["br_p"] = _wilcoxon(br)[1]; r["reb_p"] = _wilcoxon(reb)[1]; r["tost_p"] = tost(vm, 1.0)
        for k, c in [("br", "br_pct"), ("reb", "reb"), ("vmaf", "vmaf")]:
            mm = mixed(ep, c, pn == "measured")
            for q, v in mm.items(): r[f"mm_{k}_{q}"] = v
        r["mm_tost_eq"] = bool(r["mm_vmaf_lo90"] > -1 and r["mm_vmaf_hi90"] < 1)
        rows.append(r)
D = pd.DataFrame(rows)
# graph: nodes ordered chain-major [H1,H2,H3] x 4 chains
n = 12; w = np.zeros(n); G = np.zeros((n, n)); P = []
for c in range(4):
    w[3 * c] = 0.25; G[3 * c, 3 * c + 1] = 1; G[3 * c + 1, 3 * c + 2] = 1
    for o in range(4):
        if o != c: G[3 * c + 2, 3 * o] = 1 / 3
    P += [D.br_p[c], D.reb_p[c], D.tost_p[c]]
A = graphical(P, w, G)
D["br_padj"], D["reb_padj"], D["tost_padj"] = A[0::3], A[1::3], A[2::3]
D.to_csv(f"{R}/primary.csv", index=False)
pd.set_option("display.width", 250)
print(D.T.to_string())
