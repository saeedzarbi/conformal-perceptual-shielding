# cf_budget: counterfactual-reference perceptual budget (CPS-S-CF / CPS-A-CF)

Definition, eps, and evaluation plan: PREREG.md (commit b7750d6, made before any evaluation).
Code: src/evaluation/eval_certified_shield_v18.py (arms certified_cf, certified_a_cf;
CounterfactualReferenceEnv). certified_perceptual_shield.py now also forwards eps_eff in info
(no behaviour change).  Runs: src/training/run_cf_budget.sh (exact commands in commands.txt).
Tables: summarize_cf_budget.py writes cf_compare.csv, cf_checks.csv, codesign_pooled.csv and tables.md.

## Pathwise checks (PREREG "expected property")
In all 84 CF arm-runs (42 runs x 2 CF arms, 204 episodes each), we found:
* 0 chunks with V(p_t) < V(r_t) - eps_eff;
* 0 chunks where the played rebuffering exceeds the reference rebuffering;
* the reference trajectory equals the Safety arm exactly (max |dVMAF| = max |dRebuffer| = 0).

## Primary endpoint: primary PPO, CPS-S-CF vs Safety, main config (B30 warm, alpha 0.05)
Pre-registered success criterion: title-level TOST (+/-1 VMAF) p < 0.05.

| pool | variant | bitrate | stall diff (s) | dVMAF [95% CI] | TOST p | titles < -1 |
|---|---|---|---|---|---|---|
| irish | CPS-S (current) | -36.4% | -0.96 (p=0.016) | -1.78 [-3.93, 0.00] | 0.765 | 5 |
| irish | **CPS-S-CF** | -6.6% (p=0.004) | -0.54 (p=0.031) | +0.34 [0.02, 0.94] | **0.023 (pass)** | 0 |
| lumos | CPS-S (current) | -34.5% | -0.05 (p=0.125) | -1.86 [-4.10, 0.00] | 0.774 | 5 |
| lumos | **CPS-S-CF** | -6.3% (p=0.004) | -0.00 (p=0.125) | +0.34 [0.01, 0.96] | **0.026 (pass)** | 0 |
| synthetic | CPS-S (current) | -39.8% | -0.13 (p=0.125) | -0.92 [-2.81, 0.84] | 0.466 | 3 |
| synthetic | **CPS-S-CF** | -9.0% (p=0.004) | -0.00 (p=0.5) | +0.77 [0.18, 1.64] | **0.294 (FAIL)** | 0 |

The synthetic failure is on the UPPER side.  CPS-S-CF gives higher VMAF than Safety (lower CI
bound +0.18), and two-sided TOST rejects equivalence when the CI exceeds +1.  This is possible
because the ladders are non-monotone: the knee can select a cheaper rung with higher VMAF, e.g.
750 vs 1200 kbps.  A one-sided non-inferiority reading would pass, but that was NOT the
pre-registered test.

## Honest summary
* The drift is removed for PPO.  The VMAF loss vs Safety turns into a small gain, and no title is
  below -1 (previously 5 to 7 titles, down to -11).  The price is that most of the earlier PPO
  "bandwidth saving" disappears: -6 to -11% instead of -34 to -43%.  The earlier saving was
  largely the policy drifting to lower rungs, not perceptual banking.
* Paper config (B12 reset), PPO CPS-S-CF: irish TOST p=0.040, lumos 0.0495 (both pass);
  synthetic 0.38 (fail, again upper side: +0.87 [0.24, 1.72]).  CPS-A-CF on PPO: irish/lumos
  pass at both configs (main 0.022/0.026; paper 0.040/0.048).  It fails on synthetic main
  (0.056; dVMAF +0.18 [-0.55, 1.19], 2 titles < -1) and on synthetic paper (0.21).
* Non-PPO hosts on the real traces: CPS-S-CF is about the same as CPS-S (bitrate -6.8 to -7.1%,
  dVMAF within +/-0.07, TOST p < 1e-6).  Title-level bitrate significance becomes p <= 0.03 for
  greedy and MPC.  The stall diff is slightly larger and now often significant (e.g. irish greedy
  B30: -0.22 s, p=0.031).
* Non-PPO hosts on synthetic: CPS-S-CF saves more than CPS-S (B30: -4.5 to -5.3% vs -2.9 to -3.4%)
  with a non-negative dVMAF (+0.11 to +0.42).  CPS-A-CF is WORSE than CPS-A on VMAF there.
  B30: dVMAF -0.31 to -0.57 and 3 to 4 titles below -1, vs +0.40 to +0.60 for CPS-A.  It saves more
  (-26.5 to -27.5% vs -21.2 to -22.8%) and stalls a little less.  TOST still passes but greedy is
  marginal (0.045).
* Co-design r6 (synthetic, seed-pooled, N=12):
  - cocps: CPS-S-CF dVMAF vs Safety +1.66 (B30) / +1.49 (B12); TOST fails (0.80 / 0.76) on the
    upper side; no title below -1; bitrate saving -219 / -203 kbps (vs -402 / -359 for CPS-S).
  - coreg: CPS-S-CF +0.10 / +0.08, TOST p < 1e-8; CPS-A-CF -0.36 / -0.17, TOST p < 1e-3.
  - Per seed: cocps s0 and s2 lose half or more of the bitrate saving (about -12% vs -32%).  cocps s1
    is essentially unchanged; its PPO request differs from the shadow on only 0.8% of chunks.
* Coverage is unchanged (the conformal estimator is exogenous): 0.95 warm / 0.83 to 0.92 reset.

## Deviations from PREREG
None in method, eps, arms, hosts, pools, configs, or statistics.  Before the full grid we ran 3
smoke runs (5 episodes: PPO, BBA, MPC on irish B30) to check the code runs and that the
reference equals Safety.  These were not used to tune anything.

## Deployment note (for the paper)
The reference trajectory needs a shadow copy of the host and of the buffer recursion.  The
shadow's download times are computed from the measured throughput, i.e. throughput is assumed
exogenous (independent of the requested rung), the same assumption the simulator makes.
