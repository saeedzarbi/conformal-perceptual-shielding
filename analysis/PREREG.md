# review8 pre-registration (written and committed BEFORE any review8 test-pool run)

Date: 2026-10-20 (Asia/Tehran). Code: eval_certified_shield_v18.py with arms
tfixed_safety, aci01_safety, aci05_safety; ABR_STABLE_VBR=1 (--stable-vbr).

## Tuned fixed-rho (tfixed_safety)
* Grid G = {0.15, 0.25, 0.35, 0.50, 0.65, 0.80}.
* Data: ONLY the calibration (train) pool of each dataset (train_traces_5g_v18,
  train_traces_real5g_irish, train_traces_real5g_lumos, train_traces_v18 for broadband);
  residual ratios actual/harmonic-mean(5) prediction, same predictor as the shield
  (residual_ratios_from_traces, all ratios, no window cap).
* Rule: rho* = the LARGEST rho in G whose calibration coverage P(ratio >= rho) >= 1 - alpha
  (alpha = 0.05, main config). If no rho in G attains it, rho* = the rho in G with the
  highest calibration coverage (i.e. the smallest, 0.15). No test-pool data is used.
* rho* is fixed per dataset before evaluation; same rho* for all hosts.

## ACI (aci01_safety, aci05_safety)
* Gibbs & Candes (2021) adaptive conformal inference replacing the split-conformal
  quantile inside the Safety shield: alpha_{t+1} = alpha_t + gamma (alpha - err_t),
  err_t = 1{realized < bound_t}; bound = pred * order statistic floor(alpha_t (n+1)) of the
  same 200-ratio window; alpha_t <= 0 -> trivial bound (min throughput); same warm start,
  pre-calibration, fallback (n < 20) as Safety. gamma in {0.01, 0.05}; alpha_t carried
  across episodes under warm start.

## Runs (main configuration: B_max=30 s, alpha=0.05, warm start + held-out pre-calibration,
## eps=1, 204 paired episodes)
* Pools: synthetic 5G, Irish 5G, Lumos5G, broadband (FCC+Norway); hosts greedy, BBA, BOLA,
  RobustMPC.
* Arms: raw, fixed_safety (rho=0.8), tfixed_safety (rho*), aci01_safety, aci05_safety, safety,
  certified (CPS-S), certified_a (CPS-A), certified_cf, certified_a_cf.

## Analysis
* Title-level (N=12) paired contrasts vs Safety: exact Wilcoxon (bitrate %, rebuffer s),
  cluster-bootstrap CI and TOST +/-1 VMAF, as in r7.
* Holm-Bonferroni within each table family (family = all p-values of one metric type in one
  table; stated in each caption). Raw p and Holm-adjusted p both kept in CSV.
* Coverage reported inclusive and post-warm-up (in-episode chunk index k >= W_calib = 20).
* Smoothness = mean |bitrate_k - bitrate_{k-1}| (kb/s) per episode.
* All outcomes are reported whatever their sign.
