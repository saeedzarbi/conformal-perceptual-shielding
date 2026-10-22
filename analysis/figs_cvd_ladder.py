"""Colour-blind-safe regeneration of Fig. 2 (fig_ladder_perchunk): same data/layout as
src/paper/overleaf_upload/scripts/regen_figures.py::ladder() (no panel tags, as printed);
Okabe-Ito colours plus distinct markers/linestyles per video."""
from pathlib import Path as _P; import os as _os; _ROOT=_P(__file__).resolve().parents[1]; _os.makedirs(_ROOT/'outputs/tables',exist_ok=True); _os.makedirs(_ROOT/'outputs/figures',exist_ok=True)
import sys, numpy as np, matplotlib as mpl; mpl.use("Agg"); import matplotlib.pyplot as plt
sys.path.insert(0,str(_ROOT)+'/analysis'); import analyze_vmaf_ladder_perchunk as L
mpl.rcParams.update({"font.family":"serif","font.serif":["Times New Roman","Liberation Serif","STIXGeneral"],"mathtext.fontset":"stix",
 "font.size":8,"axes.labelsize":8,"axes.titlesize":8,"xtick.labelsize":7.5,"ytick.labelsize":7.5,"legend.fontsize":7.5,
 "axes.linewidth":0.6,"xtick.major.width":0.6,"ytick.major.width":0.6,"lines.linewidth":1.2,"grid.linewidth":0.4,
 "pdf.fonttype":42,"ps.fonttype":42,"legend.handlelength":2.2,"legend.borderpad":0.3,"legend.labelspacing":0.3})
OUT=sys.argv[1] if len(sys.argv)>1 else str(_ROOT)+'/outputs/figures/fig_ladder_perchunk.pdf'
lad,pooled=L.build_ladders(); B=L.BITRATES
fig,axes=plt.subplots(2,1,figsize=(3.5,4.1),layout="constrained"); ax=axes[0]
wide=lad.pivot_table(index=["video","chunk"],columns="bitrate_kbps",values="vmaf"); pw=pooled.pivot_table(index="video",columns="bitrate_kbps",values="frame_mean")
x=np.arange(len(B)); sty=[('#0072B2','o','-'),('#E69F00','s','--'),('#009E73','^','-.')]
for i,v in enumerate(L.VIDEOS):
    c,mk,ls=sty[i]; blk=wide.loc[v]
    ax.fill_between(x,blk[B].min(axis=0).to_numpy(),blk[B].max(axis=0).to_numpy(),alpha=0.20,color=c,linewidth=0)
    ax.plot(x,pw.loc[v,B].to_numpy(),marker=mk,markersize=3,ls=ls,color=c,label=L.VIDEO_LABEL[v])
ax.set_xticks(x); ax.set_xticklabels([str(b) for b in B]); ax.set_xlabel("Representation bitrate (kb/s)"); ax.set_ylabel("VMAF"); ax.grid(alpha=0.3); ax.legend(loc="lower right",framealpha=0.92)
ax=axes[1]; gaps,labels=[],[]
for lo,hi in zip(B[:-1],B[1:]): gaps.append((wide[hi]-wide[lo]).to_numpy()); labels.append(f"{lo}\u2192{hi}")
pos=list(range(len(gaps)))
parts=ax.boxplot(gaps,positions=pos,widths=0.6,showfliers=True,patch_artist=True,medianprops={"color":"black","lw":0.9},boxprops=dict(lw=0.6),whiskerprops=dict(lw=0.6),capprops=dict(lw=0.6),flierprops=dict(markersize=2,markeredgewidth=0.4))
for p in parts["boxes"]: p.set_facecolor("#9ecae1"); p.set_alpha(0.8)
ax.axhline(0.0,color="crimson",linewidth=0.9,linestyle="--"); ax.set_xticks(pos); ax.set_xticklabels(labels,rotation=20,ha="right")
ax.set_xlabel("Adjacent representation pair (kb/s)"); ax.set_ylabel("Per-chunk VMAF gap"); ax.grid(alpha=0.3)
fig.savefig(OUT,bbox_inches="tight",pad_inches=0.02); print("wrote",OUT)
