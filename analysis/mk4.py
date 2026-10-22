"""review8: Holm-adjusted versions of PPO and co-design tables (same sources as mk2.py)."""
from pathlib import Path as _P; import os as _os; _ROOT=_P(__file__).resolve().parents[1]; _os.makedirs(_ROOT/'outputs/tables',exist_ok=True); _os.makedirs(_ROOT/'outputs/figures',exist_ok=True)
import pandas as pd, sys
sys.path.insert(0,str(_ROOT)+'/analysis'); from analyse import holm
exec(open(str(_ROOT)+'/analysis/mk2.py').read().split('# fix minus')[0])
rows=[]
for pool in ['syn','irish','lumos']:
    for var in ['CPS-S','CPS-S-CF','CPS-A-CF']:
        rows.append(cf[(cf.run==f'{pool}_prop_b30_warm_a05')&(cf.variant==var)].iloc[0].copy())
D=pd.DataFrame(rows).reset_index(drop=True)
pass  # review8b: raw p (exploratory)
hdr=[r'\toprule',r'\textbf{Pool} & \textbf{Variant} & \textbf{$\Delta$Bitrate} & \textbf{$\Delta$Reb.\ (s)} & \textbf{$\Delta$VMAF} & \textbf{TOST $p$} \\',r'\midrule']
L=[r'\begin{tabular*}{\linewidth}{@{\extracolsep{\fill}}l l cccc@{}}']+hdr; i=0
for pool,name in [('syn','Synth.\\ 5G'),('irish','Irish 5G'),('lumos','Lumos5G')]:
    for j,lab in enumerate(['\\mbox{CPS-S}','\\mbox{CPS-S-CF}','\\mbox{CPS-A-CF}']):
        r=D.iloc[i]; i+=1
        L.append(f"{name if j==0 else ''} & {lab} & {cell(num(r.bitrate_pct,'{:+.1f}')+'\\%',pp(r.bitrate_p))} & {cell(num(r.stall_diff_s),pp(r.stall_p))} & {ci(r.dVMAF,r.dVMAF_lo,r.dVMAF_hi)} & {pt(r.TOST_p)}\\\\")
    L.append(r'\midrule' if pool!='lumos' else r'\bottomrule')
L.append(r'\end{tabular*}'); open(T+'table_r7_ppo.tex','w').write('% review8: primary PPO host, main config, vs Safety; raw p, exploratory; source results/cf_budget/cf_compare.csv (pooled-episode % as in r7)\n'+'\n'.join(L)+'\n')
print(D[['run','variant','bitrate_pct','bitrate_p','stall_p','TOST_p']])
d=pd.read_csv(R+'r7_codesign/codesign_compare/codesign_title_level.csv'); d=d[d.seed=='pooled']
rows=[]
for cfg,name in [('b30_warm_a05','Synth.\\ (main)'),('b12_reset','Synth.\\ ($B_{\\max}{=}12$)'),('irish_b30_warm_a05','Irish 5G'),('lumos_b30_warm_a05','Lumos5G')]:
    for arm,lab in [('certified','\\mbox{CPS-S}'),('certified_a','\\mbox{CPS-A}')]:
        q=d[(d.config==cfg)&(d.treat=='cocps:'+arm)&(d.ref=='coreg:'+arm)]
        b=q[q.metric=='bitrate_mean_kbps'].iloc[0]; rb=q[q.metric=='rebuffer_total'].iloc[0]; vm=q[q.metric=='vmaf_mean'].iloc[0]
        rows.append(dict(cfg=cfg,name=name,lab=lab,bpc=b.video_mean_pct_change,bp=b.wilcoxon_p,rd=rb.video_mean_diff,rp=rb.wilcoxon_p,vd=vm.video_mean_diff,lo=vm.ci95_lo,hi=vm.ci95_hi))
D=pd.DataFrame(rows); pass
L=[r'\begin{tabular*}{\linewidth}{@{\extracolsep{\fill}}l l ccc@{}}',r'\toprule',r'\textbf{Pool} & \textbf{Arm} & \textbf{$\Delta$Bitrate} & \textbf{$\Delta$Reb.\ (s)} & \textbf{$\Delta$VMAF} \\',r'\midrule']
for k,r in D.iterrows():
    L.append(f"{r['name'] if k%2==0 else ''} & {r.lab} & {cell(num(r.bpc,'{:+.1f}')+'\\%',pp(r.bp))} & {cell(num(r.rd),pp(r.rp))} & {ci(r.vd,r.lo,r.hi)}\\\\")
    if k%2==1: L.append(r'\midrule' if k<7 else r'\bottomrule')
L.append(r'\end{tabular*}'); open(T+'table_r7_codesign.tex','w').write('% review8: co-design, raw p, exploratory; source results/r7_codesign/codesign_compare\n'+'\n'.join(L)+'\n')
print(D)
