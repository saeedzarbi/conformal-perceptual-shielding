# Processed per-episode results (CPS Paper 1)

Episode-level evaluation exports for *Conformal Perceptual Shielding for Adaptive Bitrate Streaming*.
Each run directory contains `episodes.csv` and usually `summary.json` from `eval_certified_shield_v18.py`.

## Layout
- `analysis/` — main synthetic-5G host/settings packs used in the paper tables
- `cf_budget/` — counterfactual budget variants
- `real5g/`, `real5g_eps05/` — Irish and Lumos5G measured-5G packs
- `v18_certified/`, `ablation_eps_alpha/`, `r7_verify/` — supporting evaluations

Trained model checkpoints are not in this repository (large binaries; kept privately by the authors).
