"""
Real 5G throughput traces for the CPS evaluation (r6 reviewer follow-up).

Replaces / complements the synthetic 5G generator (data/standardized/*_5g_v18) with two
PUBLIC measured 5G datasets, converted to the repo trace format
({"throughput_kbps": [...]} sampled at 1 s, read by the simulator as trace[int(k*T_c)]).

Datasets (downloaded by this script; raw files are NOT committed):

  irish  : D. Raca, D. Leahy, C. J. Sreenan, J. J. Quinlan, "Beyond Throughput, The Next
           Generation: A 5G Dataset with Channel and Context Metrics", ACM MMSys 2020,
           doi:10.1145/3339825.3394938.  https://github.com/uccmisl/5Gdataset
           (commit 306d92eb, file 5G-production-dataset.zip).  License: GPL-3.0.
           Commercial NSA 5G (major Irish operator, Cork), G-NetTrack Pro logs, 1 s.
           We use ONLY the "Download" (file-download, i.e. capacity-limited) sessions;
           the Netflix/Amazon sessions are application-limited (player on/off) and would
           understate link capacity.  DL_bitrate is in kbps (no conversion).
  lumos  : A. Narayanan et al., "Lumos5G: Mapping and Predicting Commercial mmWave 5G
           Throughput", ACM IMC 2020, doi:10.1145/3419394.3423629.
           https://lumos5g.umn.edu/ (Lumos5G-v1.0, Google-Drive id below).
           License: CC BY 4.0.  Verizon mmWave 5G, Minneapolis "Loop", iPerf3 per-second
           TCP downlink; Throughput is in Mbps (x1000 -> kbps).

Preprocessing (no throughput scaling):
  * irish : rows grouped by their 1-s timestamp (duplicate seconds averaged), re-indexed
            on a regular 1 s grid; missing seconds (logger gaps, max 3 s) forward-filled.
            Rows with State == "I" (idle between downloads; DL_bitrate is always 0) are
            treated as missing and forward-filled.  Rows with State == "D" are kept as
            measured, including near-zero seconds (real outages).
            5G filter: a second is 5G iff every row in it has NetworkMode == "5G";
            only CONTIGUOUS 5G segments are used (no splicing across LTE/HSPA).
  * lumos : per run_num, seq_num is already a gap-free 1 s index.  mmWave coverage is
            intermittent: only 1 of 118 runs contains a contiguous NR-only stretch of
            >=192 s, so a strict NR-only filter is infeasible without splicing.  We
            instead keep windows in which the UE is NR-connected (nrStatus == CONNECTED)
            for >= LUMOS_MIN_NR_FRAC of the seconds, and keep the LTE-fallback seconds
            exactly as measured (they are the real mmWave "dips").
  * windows: non-overlapping WINDOW_S = 200 s windows (an episode reads samples
            0,4,...,188 s = 48 chunks x 4 s); a leftover tail < 200 s is dropped.
  * split  : by SOURCE session (irish: CSV file; lumos: run_num), stratified by mobility,
            seeded; ~CALIB_FRACTION of sessions -> calibration / held-out pool
            (train_traces_real5g_<ds>, used only for --precalib-dir), the rest -> test pool
            (test_traces_real5g_<ds>).  No session contributes to both pools.
  * two variants are written for every pool:
      <pool>_traces_real5g_<ds>     (PRIMARY) each 4-s block [4k, 4k+4) replaced by its mean,
                                    i.e. the average throughput available during chunk slot k.
                                    The simulator reads ONE sample per 4-s chunk
                                    (trace[int(4k)]) and applies it to the whole download, so
                                    a single 1-s outage sample would otherwise stall a full
                                    chunk (capped at 60 s).  Block-averaging is a time
                                    average, not a scaling: the mean throughput is unchanged.
      <pool>_traces_real5g_<ds>_1s  (sensitivity) raw 1-s samples, point-sampled as measured.
  * the simulator itself clips throughput to [10, 20000] kbps (MIN/MAX_NETWORK_THROUGHPUT);
    we do NOT clip in the files, but report the fraction of samples above 20 Mbps.

Usage (from new/):  python data/prepare_traces_real5g.py [--raw-dir data/raw_real5g]
"""
from __future__ import annotations

import argparse
import hashlib
import json
import random
import sys
import urllib.request
import zipfile
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent            # new/data
STD = ROOT / "standardized"

IRISH_URL = ("https://github.com/uccmisl/5Gdataset/raw/"
             "306d92eba81cb4f1195d8b8eba625666a02a10ec/5G-production-dataset.zip")
IRISH_SHA256 = "abc729d696b5e0ba34a9c6ed35a851f2868d2b5071be5efb8a49691c23bb1a9b"
IRISH_LICENSE_URL = ("https://raw.githubusercontent.com/uccmisl/5Gdataset/"
                     "306d92eba81cb4f1195d8b8eba625666a02a10ec/LICENSE")
LUMOS_URL = "https://drive.google.com/uc?export=download&id=1spFuxOe5VFyWOHdeJsBeHPHbHiGidw22"
LUMOS_SHA256 = "1ea8770b3090735f987d4ef7b248188de7c667bf6be5716e4891daf9e8d331e8"

WINDOW_S = 200
CALIB_FRACTION = 0.40
LUMOS_MIN_NR_FRAC = 0.80
SPLIT_SEED = 2026
SIM_MAX_KBPS = 20000.0


def _fetch(url, dest: Path, sha):
    if not dest.exists():
        dest.parent.mkdir(parents=True, exist_ok=True)
        print(f"[real5g] downloading {url}")
        urllib.request.urlretrieve(url, dest)
    h = hashlib.sha256(dest.read_bytes()).hexdigest()
    if sha and h != sha:
        raise SystemExit(f"sha256 mismatch for {dest}: {h}")
    return dest


def _runs(mask):
    """(start, length) of contiguous True runs."""
    out, s = [], None
    for i, m in enumerate(list(mask) + [False]):
        if m and s is None:
            s = i
        elif not m and s is not None:
            out.append((s, i - s))
            s = None
    return out


def irish_sessions(raw: Path):
    z = _fetch(IRISH_URL, raw / "irish_5G-production-dataset.zip", IRISH_SHA256)
    ext = raw / "irish"
    if not ext.exists():
        zipfile.ZipFile(z).extractall(ext)
    files = sorted(p for p in ext.glob("5G-production-dataset/Download/*/*.csv"))
    out = []
    for f in files:
        mob = f.parent.name.lower()                       # static / driving
        d = pd.read_csv(f)
        d["t"] = pd.to_datetime(d["Timestamp"], format="%Y.%m.%d_%H.%M.%S")
        d["DL_bitrate"] = pd.to_numeric(d["DL_bitrate"], errors="coerce")
        # State "I" (idle: no download running between files) always logs DL_bitrate=0;
        # that is not a capacity measurement, so treat it as missing (forward-filled).
        n_idle = int((d["State"] == "I").sum())
        d.loc[d["State"] == "I", "DL_bitrate"] = np.nan
        g = d.groupby("t").agg(dl=("DL_bitrate", "mean"),
                               is5g=("NetworkMode", lambda x: float((x == "5G").all())))
        grid = pd.date_range(g.index[0], g.index[-1], freq="1s")
        n_missing = int(len(grid) - len(g) + g["dl"].isna().sum())
        g = g.reindex(grid).ffill().bfill()   # bfill only for a leading idle second
        tp = g["dl"].to_numpy(float)
        windows = []
        for s, L in _runs(g["is5g"].to_numpy() == 1.0):
            for w0 in range(s, s + L - WINDOW_S + 1, WINDOW_S):
                windows.append({"start_s": int(w0), "tp": tp[w0:w0 + WINDOW_S],
                                "nr_frac": 1.0})
        out.append({"dataset": "irish", "session": f.stem, "mobility": mob,
                    "src": str(f.relative_to(ext)), "seconds": len(grid),
                    "missing_seconds_ffilled": n_missing, "idle_rows_dropped": n_idle,
                    "frac_5g": float(g["is5g"].mean()), "windows": windows})
    return out


def lumos_sessions(raw: Path):
    z = _fetch(LUMOS_URL, raw / "lumos5g-v1.0.zip", LUMOS_SHA256)
    ext = raw / "lumos"
    if not ext.exists():
        zipfile.ZipFile(z).extractall(ext)
    d = pd.read_csv(ext / "Lumos5G-v1.0" / "Lumos5G-v1.0.csv")
    out = []
    for rn, g in d.groupby("run_num"):
        g = g.sort_values("seq_num")
        assert (np.diff(g["seq_num"].to_numpy()) == 1).all(), f"gap in run {rn}"
        tp = g["Throughput"].to_numpy(float) * 1000.0     # Mbps -> kbps
        nr = (g["nrStatus"] == "CONNECTED").to_numpy(float)
        windows = []
        for w0 in range(0, len(tp) - WINDOW_S + 1, WINDOW_S):
            f = float(nr[w0:w0 + WINDOW_S].mean())
            if f >= LUMOS_MIN_NR_FRAC:
                windows.append({"start_s": int(w0), "tp": tp[w0:w0 + WINDOW_S], "nr_frac": f})
        out.append({"dataset": "lumos", "session": f"run{int(rn):03d}",
                    "mobility": str(g["mobility_mode"].iloc[0]),
                    "direction": str(g["trajectory_direction"].iloc[0]),
                    "src": f"Lumos5G-v1.0.csv:run_num={int(rn)}", "seconds": len(g),
                    "frac_5g": float(nr.mean()), "windows": windows})
    return out


CHUNK_S = 4


def block_mean(tp):
    tp = np.asarray(tp, float)
    n = len(tp) // CHUNK_S * CHUNK_S
    b = tp[:n].reshape(-1, CHUNK_S).mean(axis=1)
    return np.repeat(b, CHUNK_S)


def split_and_write(sessions, ds, variant=""):
    rng = random.Random(SPLIT_SEED)
    sessions = [s for s in sessions if s["windows"]]
    calib, test = [], []
    for mob in sorted({s["mobility"] for s in sessions}):
        grp = sorted([s for s in sessions if s["mobility"] == mob], key=lambda s: s["session"])
        rng.shuffle(grp)
        k = max(1, int(round(CALIB_FRACTION * len(grp))))
        if len(grp) - k < 1:
            k = len(grp) - 1
        calib += grp[:k]
        test += grp[k:]
    manifest = {"dataset": ds, "variant": variant or "4s_block_mean", "window_s": WINDOW_S, "calib_fraction": CALIB_FRACTION,
                "split_seed": SPLIT_SEED, "pools": {}}
    for pool, ss in (("train", calib), ("test", test)):
        d = STD / f"{pool}_traces_real5g_{ds}{variant}"
        d.mkdir(parents=True, exist_ok=True)
        for old in d.glob("*.json"):
            old.unlink()
        allv, rows = [], []
        for s in ss:
            for w in s["windows"]:
                name = f"{ds}_{s['mobility']}_{s['session']}_t{w['start_s']:05d}.json"
                raw_tp = np.asarray(w["tp"], float)
                tp = [float(x) for x in (raw_tp if variant == "_1s" else block_mean(raw_tp))]
                (d / name).write_text(json.dumps({"throughput_kbps": tp}))
                allv.append(np.asarray(tp))
                rows.append({"file": name, "session": s["session"], "mobility": s["mobility"],
                             "start_s": w["start_s"], "nr_frac": w["nr_frac"],
                             "mean_kbps": float(np.mean(tp))})
        v = np.concatenate(allv) if allv else np.zeros(1)
        # what the simulator actually reads: samples at 0,4,...,188 s, clipped
        sim = np.concatenate([np.clip(a[0:192:4], 10.0, SIM_MAX_KBPS) for a in allv]) \
            if allv else np.zeros(1)
        manifest["pools"][pool] = {
            "dir": str(d.relative_to(ROOT.parent)),
            "sessions": [{k: s[k] for k in s if k != "windows"} | {"n_windows": len(s["windows"])}
                         for s in ss],
            "n_sessions": len(ss), "n_traces": len(rows),
            "kbps_raw": {"mean": float(v.mean()), "median": float(np.median(v)),
                         "p10": float(np.percentile(v, 10)), "p5": float(np.percentile(v, 5)),
                         "max": float(v.max())},
            "frac_samples_above_20mbps": float((v > SIM_MAX_KBPS).mean()),
            "frac_samples_below_6mbps": float((v < 6000).mean()),
            "frac_samples_below_1_2mbps": float((v < 1200).mean()),
            "sim_view_kbps": {"mean": float(sim.mean()), "median": float(np.median(sim)),
                              "frac_below_6000": float((sim < 6000).mean()),
                              "frac_below_2850": float((sim < 2850).mean())},
            "traces": rows,
        }
    (STD / f"real5g_{ds}{variant}_manifest.json").write_text(json.dumps(manifest, indent=2))
    for pool in ("train", "test"):
        p = manifest["pools"][pool]
        print(f"[real5g] {ds:5s}{variant:3s} {pool:5s}: {p['n_sessions']:3d} sessions, {p['n_traces']:3d} "
              f"traces, median {p['kbps_raw']['median']/1000:.1f} Mbps, "
              f">20Mbps {100*p['frac_samples_above_20mbps']:.1f}%, "
              f"<6Mbps {100*p['frac_samples_below_6mbps']:.1f}%")
    return manifest


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--raw-dir", default=str(ROOT / "raw_real5g"))
    ap.add_argument("--datasets", default="irish,lumos")
    a = ap.parse_args()
    raw = Path(a.raw_dir)
    for ds in a.datasets.split(","):
        sess = irish_sessions(raw) if ds == "irish" else lumos_sessions(raw)
        split_and_write(sess, ds, "")
        split_and_write(sess, ds, "_1s")


if __name__ == "__main__":
    sys.exit(main())
