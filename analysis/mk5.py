"""review8b: primary-family table (graphical gatekeeping + mixed model)."""
from pathlib import Path as _P; import os as _os; _ROOT=_P(__file__).resolve().parents[1]; _os.makedirs(_ROOT/'outputs/tables',exist_ok=True); _os.makedirs(_ROOT/'outputs/figures',exist_ok=True)
import pandas as pd
R=str(_ROOT)+'/analysis/'; T=str(_ROOT)+'/outputs/tables/'
D=pd.read_csv(R+'primary.csv')
def p(x): return '$<$0.001' if x<0.001 else f'{x:.3f}'
def pa(x,reached=True):
    s=p(x); return f'\\textbf{{{s}}}' if x<0.05 else s
L=[r'\begin{tabular*}{\textwidth}{@{\extracolsep{\fill}}l l c c c c c c@{}}',r'\toprule',
 r'\textbf{Chain} & \textbf{Hypothesis} & \textbf{Effect [95\% CI]} & \textbf{$d_z$} & \textbf{$p$} & \textbf{$p_{\mathrm{adj}}$} & \textbf{Mixed model [95\% CI]} & \textbf{$p_{\mathrm{MM}}$} \\',r'\midrule']
names={'synthetic':'synth.\\ 5G','measured':'measured 5G'}
for i,r in D.iterrows():
    ch=f"\\mbox{{{r.arm}}}, {names[r.pool]}"
    L.append(f"{ch} & H1 $\\Delta$bitrate (\\%) & {r.br:+.1f} [{r.br_lo:+.1f}, {r.br_hi:+.1f}] & {r.br_dz:.2f} & {p(r.br_p)} & {pa(r.br_padj)} & {r.mm_br_est:+.1f} [{r.mm_br_lo95:+.1f}, {r.mm_br_hi95:+.1f}] & {p(r.mm_br_p)} \\\\")
    L.append(f" & H2 $\\Delta$reb.\\ (s) & {r.reb:+.2f} [{r.reb_lo:+.2f}, {r.reb_hi:+.2f}] & {r.reb_dz:.2f} & {p(r.reb_p)} & {pa(r.reb_padj)} & {r.mm_reb_est:+.2f} [{r.mm_reb_lo95:+.2f}, {r.mm_reb_hi95:+.2f}] & {p(r.mm_reb_p)} \\\\")
    L.append(f" & H3 $\\Delta$VMAF within $\\pm1$ & {r.vmaf:+.2f} [{r.vmaf_lo:+.2f}, {r.vmaf_hi:+.2f}] & {r.vmaf_dz:.2f} & {p(r.tost_p)} & {pa(r.tost_padj)} & {r.mm_vmaf_est:+.2f} [{r.mm_vmaf_lo90:+.2f}, {r.mm_vmaf_hi90:+.2f}]$^{{\\dagger}}$ & {'equiv.' if r.mm_tost_eq else 'not equiv.'} \\\\")
    L.append(r'\midrule' if i<3 else r'\bottomrule')
L.append(r'\end{tabular*}')
import re
def cols(lines,keep):
    out=[]
    for l in lines:
        if '&' in l and l.rstrip().endswith('\\\\'):
            c=l.rstrip()[:-2].split(' & '); out.append(' & '.join(c[i] for i in keep)+' \\\\')
        else: out.append(l)
    return out
M=cols(L,[0,1,2,4,5]); M[0]=r'\begin{tabular*}{\linewidth}{@{\extracolsep{\fill}}l l c c c@{}}'
A=cols(L,[0,1,3,6,7]); A[0]=r'\begin{tabular*}{\linewidth}{@{\extracolsep{\fill}}l l c c c@{}}'
open(T+'table_r7_primary_mm.tex','w').write('% review8b: primary family, d_z and mixed model; source analysis/primary.csv\n'+'\n'.join(A)+'\n')
L=M
open(T+'table_r7_primary.tex','w').write('% review8b: primary family; source analysis/primary.csv (primary.py, PREREG_stats.md)\n'+'\n'.join(L)+'\n')
