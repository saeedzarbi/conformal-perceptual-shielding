# Real 5G traces: attribution and licenses

The directories `{train,test}_traces_real5g_{irish,lumos}[_1s]/` are DERIVED from:

* Irish 5G dataset: D. Raca, D. Leahy, C. J. Sreenan, J. J. Quinlan, "Beyond Throughput,
  The Next Generation: A 5G Dataset with Channel and Context Metrics", ACM MMSys 2020,
  doi:10.1145/3339825.3394938. https://github.com/uccmisl/5Gdataset (commit 306d92eb).
  Licensed GPL-3.0 (see real5g_irish_LICENSE_GPL-3.0.txt). The derived traces are
  redistributed under the same license.
* Lumos5G-v1.0: A. Narayanan et al., "Lumos5G: Mapping and Predicting Commercial mmWave
  5G Throughput", ACM IMC 2020, doi:10.1145/3419394.3423629. https://lumos5g.umn.edu/
  Licensed CC BY 4.0 (see real5g_lumos_LICENSE_CC-BY-4.0.txt). Changes: unit conversion
  (Mbps to kbps), 200-s windowing, NR-dominant window filter, 4-s block averaging
  (primary variant).

Regenerate the traces from the original downloads with `python data/prepare_traces_real5g.py`.
The raw downloads (`data/raw_real5g/`) are not committed.
