# Conformal Perceptual Shielding for Adaptive Bitrate Streaming

Code for the paper *Conformal Perceptual Shielding for Adaptive Bitrate Streaming*
(S. Zarbi, L. Farzinvash, P. Salehpour). The Conformal Perceptual Shield (CPS) is a
run-time layer that sits on top of any adaptive bitrate (ABR) controller. It removes
bitrate rungs whose extra cost is not justified by perceptual quality (per-chunk VMAF),
and uses an online conformal bound on throughput to spend a stall budget safely
(CPS-S: static budget, CPS-A: adaptive budget). This repository contains the chunk-level
streaming simulator, the CPS and baseline shields, the ABR hosts (greedy, BBA, BOLA,
RobustMPC, PPO/Pensieve), the synthetic 5G trace generator, the evaluation driver, and
the statistics and figure scripts.

## Layout
```
configs/            paths and video catalogue
src/environment/    simulator (abr_multi_env_v18.py and the classes it extends)
src/training/       CPS (certified_perceptual_shield.py), safety shield, PPO / co-design / Pensieve training
src/baselines/      BBA, MPC, VMAF-MPC, genie
src/evaluation/     evaluation driver (eval_certified_shield_v18.py), coverage-drift and per-chunk ablations
data/               trace generator and preparation scripts, per-chunk VMAF tables, SI/TI features
analysis/   statistics (primary.py, analyse.py, rho_selection.py), table/figure scripts, analysis plan (PREREG*.md)
outputs/            created by the scripts (figures, tables); not tracked
```

## Install
```
python -m venv .venv && . .venv/bin/activate
pip install torch --index-url https://download.pytorch.org/whl/cpu   # or a CUDA build
pip install -r requirements.txt
```
Python 3.10+ is recommended. Run all commands from the repository root.

## Traces
The synthetic 5G test set (`data/standardized/test_traces_5g_v18`, 50 traces, seed 42) is included.
To regenerate it and the disjoint training set (80 traces, seed 7000):
```
python data/prepare_traces_v18.py --5g-only
```
The broadband (FCC / Norway) split is built by the same script without `--5g-only` once the
original traces are placed in `data/standardized/train_traces` and `data/standardized/test_traces`.
The measured 5G datasets are not redistributed; download and convert them with
```
python data/prepare_traces_real5g.py --raw-dir data/raw_real5g
```
The manifests in `data/standardized/real5g_*_manifest.json` list the expected windows.

## Evaluation
Quick check (a few episodes, non-learned host):
```
python src/evaluation/eval_certified_shield_v18.py --policy greedy \
    --trace-dir data/standardized/test_traces_5g_v18 --episodes 3 \
    --arms raw,safety,certified --out outputs/smoke
```
Use `--policy {greedy,bba,bola,mpc,ppo}` (`--ckpt` for PPO), `--episodes`, `--alpha`, `--epsilon`
and the other options listed by `--help`. Each run writes `episodes.csv` and `summary.json`.
PPO hosts are trained with `src/training/train_all_models_v18.py`, `train_codesign_r7.py`
and `train_pensieve_r7.py`; trained checkpoints are not included.

## Statistics and figures
With evaluation outputs under `analysis/` (one directory per pool/host/setting, e.g. `--out analysis/syn_greedy_b30_warm_a05`, each containing
`episodes.csv`), run from the repository root:
```
python analysis/rho_selection.py
python analysis/analyse.py
python analysis/primary.py
python analysis/mk5.py
python analysis/figs_fonts.py
python analysis/figs_cvd_ladder.py
python src/evaluation/coverage_under_drift.py
python src/evaluation/ablation_perchunk_ladder.py
```
Tables and figures are written to `outputs/tables` and `outputs/figures`. The analysis plan
is in `analysis/PREREG.md` and `analysis/PREREG_stats.md`.

## Data sources and licences
- FCC Measuring Broadband America traces (Federal Communications Commission, 2016), public data.
- Norway/Nornet 3G/HSDPA traces (Kvalbein et al., NorNet), public data.
- Irish 5G dataset: D. Raca et al., "Beyond Throughput, The Next Generation: A 5G Dataset with
  Channel and Context Metrics", ACM MMSys 2020, https://github.com/uccmisl/5Gdataset (GPL-3.0;
  not redistributed here, fetched by the conversion script).
- Lumos5G: A. Narayanan et al., "Lumos5G: Mapping and Predicting Commercial mmWave 5G
  Throughput", ACM IMC 2020, https://lumos5g.umn.edu/ (CC BY 4.0; fetched by the conversion script).
- Video titles: Blender open movies (Big Buck Bunny, Sintel, Tears of Steel; CC BY) and the
  Xiph.org Derf test media collection; quality scores computed with VMAF.
See `data/standardized/REAL5G_ATTRIBUTION.md` for details.

## Citation
```
@article{zarbi2026cps,
  title   = {Conformal Perceptual Shielding for Adaptive Bitrate Streaming},
  author  = {Zarbi, Saeed and Farzinvash, Leili and Salehpour, Pedram},
  journal = {Computer Networks},
  note    = {Submitted},
  year    = {2026}
}
```

## Licence
Code: MIT (see `LICENSE`). Third-party datasets keep their own licences.
