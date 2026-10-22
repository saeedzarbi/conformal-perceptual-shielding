"""Select tuned fixed-rho on calibration pools only (PREREG.md rule)."""
import json, os
os.environ["ABR_STABLE_VBR"] = "1"
import numpy as np
from src.training.certified_perceptual_shield import residual_ratios_from_traces
G = [0.15, 0.25, 0.35, 0.50, 0.65, 0.80]; ALPHA = 0.05
out = {}
for n, d in [("syn", "train_traces_5g_v18"), ("irish", "train_traces_real5g_irish"),
             ("lumos", "train_traces_real5g_lumos"), ("bb", "train_traces_v18")]:
    r = np.asarray(residual_ratios_from_traces("data/standardized/" + d, k_predict=5,
                   chunk_duration=4.0, max_chunks=48, max_ratios=None))
    cov = {g: float((r >= g).mean()) for g in G}
    ok = [g for g in G if cov[g] >= 1 - ALPHA]
    rho = max(ok) if ok else max(G, key=lambda g: (cov[g], -g))
    out[n] = {"calib_pool": d, "n_ratios": int(r.size), "coverage": cov, "rho_star": rho,
              "attains_target": bool(ok), "q05": float(np.quantile(r, 0.05))}
json.dump(out, open("analysis/rho_selection.json", "w"), indent=2)
print(json.dumps({k: (v["rho_star"], v["attains_target"]) for k, v in out.items()}))
