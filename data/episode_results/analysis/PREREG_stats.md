# review8b statistical analysis plan (primary family + gatekeeping + mixed model)

Written and committed 2026-10-20 (Asia/Tehran) BEFORE the tests below were computed.
Disclosure: the episode-level review8 data and the per-host / per-pool contrasts (Holm over
36-test families) had already been inspected. This plan is therefore NOT a pre-registration
in the strict sense. The primary family is fixed by the paper's headline claims (the r7
abstract and contributions: CPS-S / CPS-A vs Safety, main configuration, synthetic and
measured 5G). It is not chosen by p-values. It is stated as such in the paper.

## Primary family (confirmatory), main config (B30, warm start + precalib, alpha=0.05, eps=1)
Chains c in {CPS-S, CPS-A} x {synthetic 5G, measured 5G (Irish + Lumos5G pooled)} = 4 chains.
Unit: title (N=12). The per-title value averages the paired episode differences (arm minus Safety)
over the 4 heuristic hosts (greedy, BBA, BOLA, RobustMPC) and, for measured 5G, over both datasets.
Each chain has 3 ordered hypotheses:
 H1 bitrate: relative bitrate change (100 * mean diff / mean Safety, per title) != 0, exact Wilcoxon two-sided.
 H2 stall: rebuffering difference (s/episode) != 0, exact Wilcoxon two-sided.
 H3 equivalence: VMAF difference within +/-1 point, TOST (paired t).
Error control: graphical gatekeeping procedure (Bretz et al. 2009), FWER 0.05.
 Initial weights 1/4 on each H1. H1_c -> H2_c (weight 1), H2_c -> H3_c (weight 1),
 H3_c -> H1 of the other chains (1/3 each). This is Holm across chains with fixed
 sequence within each chain. Adjusted p-values are reported.
 A hypothesis that is not reached or not rejected is reported as not confirmed.

## Complementary mixed model (per primary chain, not used for error control)
Unit = trace x title episode (204 synthetic; 2 x 204 measured), outcome = episode diff averaged over hosts.
Model: diff ~ 1 [+ dataset, sum-coded, measured pool], random intercepts for title and trace
(crossed, statsmodels MixedLM variance components, REML). Report the intercept estimate, its 95% Wald CI and p.
For the VMAF equivalence: the 90% CI is checked against +/-1 (TOST equivalent).

## Secondary / exploratory (everything else)
New baselines (tuned fixed-rho, ACI), per-host and per-pool rows, CF variants, broadband, PPO,
Pensieve, co-design, reset ablation, sensitivity: raw (unadjusted) p, effect size, 95%
title-cluster bootstrap CI. Labelled exploratory; no multiplicity claim.
