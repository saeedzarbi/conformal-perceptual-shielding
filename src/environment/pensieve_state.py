"""Pensieve-faithful observation mask (r7_pensieve, see results/r7_pensieve/PREREG.md).

The v5..v18 observation (35 dims) is
    0-11 throughput history | 12 buffer | 13 buffer trend | 14 last bitrate |
    15-16 SI/TI | 17-22 VMAF per rung | 23-28 next-chunk sizes | 29-34 download-time/buffer ratios.

Pensieve (Mao et al., SIGCOMM 2017) observes past throughput / download times, buffer, last bitrate,
chunks left and the NEXT CHUNK SIZES of every rung; it is content-(perception-)blind but not
size-blind. The legacy ContentBlindWrapper zeroed obs[15:], i.e. also the next-chunk sizes, and the
legacy pensieve spec trained with use_future=False (sizes = 0). Here we zero only the perceptual
features (SI/TI, VMAF ladder: 15-22) and the engineered dl-ratio features of the proposed model
(29-34), and keep the next-chunk sizes (23-28). The SAME function is used in training and at eval.
"""
import numpy as np

PENSIEVE_MASK = np.r_[15:23, 29:35]


def pensieve_obs(obs):
    o = np.array(obs, dtype=np.float32, copy=True)
    o[PENSIEVE_MASK] = 0.0
    return o
