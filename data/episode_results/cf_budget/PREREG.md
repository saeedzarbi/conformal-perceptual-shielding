# Pre-registration: counterfactual-reference perceptual budget (CPS-CF)

Written and committed BEFORE any evaluation of the variant was run (see git history: this file
is committed before the implementation and every result under results/cf_budget/).
Approved by S. Zarbi on 2026-10-19 as "method improvement #1".

## Problem being addressed
In CPS-S / CPS-A the perceptual budget is checked against the host's CURRENT request
a_t = pi(o_t), where o_t is observed on the shielded trajectory. For a stateful learned host
(PPO), banking lowers the buffer/last-rung features it observes, and its requests drift down
(primary PPO, Irish B30: share of 1850-kbps requests 86% -> 35%). The per-chunk bound
V(played) >= V(a_t) - eps holds, but the episode VMAF falls about 1.8 below Safety, outside +/-1.

## Definition (exact; this is what the code implements)
Notation: rungs j in {0..M-1}, V_t(j) the per-title VMAF of rung j, D_t(j | L) = min(S_t(j) / L, 60 s)
the download time of chunk t at rung j under throughput L, B_t the buffer before chunk t,
L_t the conformal lower bound, delta = 0.5 s the safety margin, B_min = 0.3 s.
Safety-only projection:  P(j; B, L) = 0 if B <= B_min, else max{ i <= j : D_t(i|L) <= max(B - delta, 0.1) } (0 if empty).
CPS knee for reference rung r with budget e:  K(r, e) = min{ j <= r : V_t(r) - V_t(j) <= e }.

Two coupled trajectories start from the same initial state (same title, trace, initial buffer, seed,
and conformal-estimator state):

1. Reference (counterfactual Safety-only) trajectory, state x~_t with buffer B~_t and observation o~_t:
       r_t = P( pi(o~_t); B~_t, L_t ),     x~_{t+1} = f(x~_t, r_t, c_t).
2. Played trajectory, state x_t with buffer B_t:
       e_t   = eps                                  (CPS-S-CF)
             = eps_eff,t from the CPS-A rule        (CPS-A-CF; buffer ramp on B_t, 6-chunk look-ahead
                                                     at rung r_t under the dip-forecast plan rate)
       p_t   = P( K(r_t, e_t); B_t, L_t )           (played rung)
       x_{t+1} = f(x_t, p_t, c_t).

c_t is the exogenous trace throughput. L_t is the SAME in both trajectories because the conformal
estimator only consumes realised trace throughput, which is exogenous in the simulator. The host's
request on the played trajectory, pi(o_t), is NOT used (logged only). In code, CPS-CF is the
unchanged CPS operator (certified_safe_action) applied to r_t instead of pi(o_t), and r_t comes from
a shadow env + shadow policy copy wrapped by the Safety-only shield.

Paper form (CPS-S-CF):  p_t = P( min{ j : V_t(r_t) - V_t(j) <= eps }, B_t, L_t ),
with r_t the Safety-only rung of the counterfactual unbanked trajectory.

## Expected property (to be CHECKED, not assumed)
Chunk sizes are increasing in the rung, and buffer and rebuffering dynamics are monotone. So
p_t <= r_t gives B_t >= B~_t for all t by induction. Then K(r_t, e_t) is feasible whenever r_t was
feasible on B~_t, which gives, path by path and chunk by chunk:
  (i)  V_t(p_t) >= V_t(r_t) - e_t,   (ii) rebuffer_t(played) <= rebuffer_t(reference).
The reference trajectory must coincide with the Safety arm of the same process.
Checks logged per run: number of chunks violating (i) by more than 1e-9, number violating (ii),
and max |episode VMAF / rebuffer of reference - Safety arm|.

## Evaluation plan (fixed; no sweeps)
* eps = 1.0 (CPS-A: eps_risk 4.0, risk_buffer 8 s, look-ahead 6, horizon quantile 0.2, unchanged).
* Arms in ONE process: raw, safety, certified (CPS-S), certified_a (CPS-A), certified_cf (CPS-S-CF),
  certified_a_cf (CPS-A-CF), fixed_safety, fixed_certified, fixed_certified_a (rho = 0.80).
* 204 paired episodes, ABR_STABLE_VBR=1, --stable-vbr, 12 evaluation titles.
* Hosts:
  - PRIMARY: PPO proposed_v14 best_model (results/models/master_v18_5g/proposed_v14/seed_0/best_model).
  - Sanity: greedy, BBA, BOLA, RobustMPC.
  - Co-design r6: master_v18_5gtrain_r6/{proposed_cps_v18, proposed_v14}/seed_{0,1,2}/best_model
    (synthetic 5G only, as in results/r6_codesign).
* Trace pools / configs:
  - synthetic: test_traces_5g_v18 (pre-calibration pool train_traces_5g_v18)
  - irish, lumos: test_traces_real5g_{irish,lumos} (4-s means; pre-cal train_traces_real5g_<ds>)
  - main config: B_max 30 s, --persist-calib + --precalib-dir, alpha 0.05
  - paper config: B_max 12 s, per-episode reset, alpha 0.10
* Statistics: video_level_stats.py (N = 12 titles; exact Wilcoxon; cluster bootstrap CI; TOST
  margin +/-1 VMAF) for the pairs certified_cf:safety, certified_a_cf:safety, certified:safety,
  certified_a:safety, certified_cf:certified, certified_cf:raw, certified_a_cf:raw.
* PRIMARY endpoint: primary PPO, CPS-S-CF vs Safety, main config, each of the 3 pools.
  Success iff the title-level TOST p < 0.05 (VMAF within +/-1). Reported together with the
  bitrate saving (title Wilcoxon p) and the stall difference.
* Secondary endpoints: the same for CPS-A-CF; the same on the paper config; non-PPO hosts
  (non-regression check vs CPS-S/CPS-A); co-design checkpoints per seed and seed-pooled.
* Every result is reported whatever its direction. Any deviation from this plan is listed in
  results/cf_budget/README.md under "Deviations".
* Before the full runs, implementation smoke tests (<= 5 episodes) only check that the code runs and
  that the reference trajectory equals the Safety arm. Their outcomes are not used to change the method.
