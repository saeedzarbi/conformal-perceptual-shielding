from pathlib import Path as _P; import os as _os; _ROOT=_P(__file__).resolve().parents[1]; _os.makedirs(_ROOT/'outputs/tables',exist_ok=True); _os.makedirs(_ROOT/'outputs/figures',exist_ok=True)
import pandas as pd, re
R=str(_ROOT)+'/results/'; T=str(_ROOT)+'/outputs/tables/'
def num(x,f='{:+.2f}'): return f.format(x).replace('-','$-$').replace('+','$+$')
def pp(x): return f'$p{{=}}{x:.3f}$' if x>=0.001 else '$p{<}0.001$'
def pt(x): return f'{x:.3f}' if x>=0.001 else r'$<$0.001'
def cell(a,b): return f'\\makecell[c]{{{a}\\\\[-1pt]\\scriptsize {b}}}'
def ci(m,lo,hi): return f'\\makecell[c]{{{num(m)}\\\\[-1pt]\\scriptsize\\ensuremath{{[{lo:.2f},\\,{hi:.2f}]}}}}'
cf=pd.read_csv(R+'cf_budget/cf_compare.csv')
# fix minus in contrasts table
p=T+'table_r7_contrasts.tex'; s=open(p).read()
s=re.sub(r'\{-(\d)',r'{$-$\1',s); s=re.sub(r'\{\+(\d)',r'{$+$\1',s); open(p,'w').write(s)
hdr=[r'\toprule',r'\textbf{Pool} & \textbf{Variant} & \textbf{$\Delta$Bitrate} & \textbf{$\Delta$Reb.\ (s)} & \textbf{$\Delta$VMAF} & \textbf{TOST $p$} \\',r'\midrule']
def rowcf(run,var,lab,first):
    r=cf[(cf.run==run)&(cf.variant==var)].iloc[0]
    return f"{first} & {lab} & {cell(num(r.bitrate_pct,'{:+.1f}')+'\\%',pp(r.bitrate_p))} & {cell(num(r.stall_diff_s),pp(r.stall_p))} & {ci(r.dVMAF,r.dVMAF_lo,r.dVMAF_hi)} & {pt(r.TOST_p)}\\\\"
# PPO primary host table
L=[r'\begin{tabular*}{\linewidth}{@{\extracolsep{\fill}}l l cccc@{}}']+hdr
for pool,name in [('syn','Synth.\\ 5G'),('irish','Irish 5G'),('lumos','Lumos5G')]:
    for i,(var,lab) in enumerate([('CPS-S','\\mbox{CPS-S}'),('CPS-S-CF','\\mbox{CPS-S-CF}'),('CPS-A-CF','\\mbox{CPS-A-CF}')]):
        L.append(rowcf(f'{pool}_prop_b30_warm_a05',var,lab,name if i==0 else ''))
    L.append(r'\midrule' if pool!='lumos' else r'\bottomrule')
L.append(r'\end{tabular*}'); open(T+'table_r7_ppo.tex','w').write('% r7: primary PPO host, main config, vs Safety; source results/cf_budget/cf_compare.csv\n'+'\n'.join(L)+'\n')
# real 5G non-PPO table (CF variants and CPS-S), main config
L=[r'\begin{tabular*}{\linewidth}{@{\extracolsep{\fill}}l l cccc@{}}',r'\toprule',r'\textbf{Pool / host} & \textbf{Variant} & \textbf{$\Delta$Bitrate} & \textbf{$\Delta$Reb.\ (s)} & \textbf{$\Delta$VMAF} & \textbf{TOST $p$} \\',r'\midrule']
for pool,name in [('irish','Irish'),('lumos','Lumos')]:
    for h,hn in [('greedy','Greedy'),('bba','BBA'),('bola','BOLA'),('mpc','RobustMPC')]:
        for i,(var,lab) in enumerate([('CPS-S-CF','S-CF'),('CPS-A-CF','A-CF')]):
            L.append(rowcf(f'{pool}_{h}_b30_warm_a05',var,lab,f'{name} {hn}' if i==0 else ''))
    L.append(r'\midrule' if pool!='lumos' else r'\bottomrule')
L.append(r'\end{tabular*}'); open(T+'table_r7_real5g.tex','w').write('% r7: real 5G, main config, vs Safety; source results/cf_budget/cf_compare.csv\n'+'\n'.join(L)+'\n')
# Pensieve r7 table: arm means (seed pooled)
a=pd.read_csv(R+'r7_pensieve/arm_means.csv'); a=a[a.host=='r7_pooled']
L=[r'\begin{tabular*}{\linewidth}{@{\extracolsep{\fill}}l l cccc@{}}',r'\toprule',r'\textbf{Pool} & \textbf{Arm} & \textbf{Reb.\ (s)} & \textbf{VMAF} & \textbf{Bitrate} & \textbf{Stall-free} \\',r'\midrule']
for cfg,name in [('syn_b30_warm_a05','Synth.\\ (main)'),('syn_b12_reset','Synth.\\ ($B_{\\max}{=}12$)'),('irish_b30_warm_a05','Irish 5G'),('lumos_b30_warm_a05','Lumos5G')]:
    for i,(arm,lab) in enumerate([('raw','\\textsc{Raw}'),('safety','\\textsc{Safety}'),('certified_cf','\\mbox{CPS-S-CF}'),('certified_a_cf','\\mbox{CPS-A-CF}')]):
        r=a[(a.config==cfg)&(a.arm==arm)].iloc[0]
        L.append(f"{name if i==0 else ''} & {lab} & {r.rebuffer_total:.2f} & {r.vmaf_mean:.2f} & {r.bitrate_mean_kbps:.0f} & {r.stallfree:.3f}\\\\")
    L.append(r'\midrule' if cfg!='lumos_b30_warm_a05' else r'\bottomrule')
L.append(r'\end{tabular*}'); open(T+'table_r7_pensieve.tex','w').write('% r7: retrained Pensieve (3 seeds pooled, 612 episodes per arm); source results/r7_pensieve/arm_means.csv\n'+'\n'.join(L)+'\n')
# Co-design r7 table: cocps vs coreg, seed-pooled title level
d=pd.read_csv(R+'r7_codesign/codesign_compare/codesign_title_level.csv'); d=d[d.seed=='pooled']
L=[r'\begin{tabular*}{\linewidth}{@{\extracolsep{\fill}}l l ccc@{}}',r'\toprule',r'\textbf{Pool} & \textbf{Arm} & \textbf{$\Delta$Bitrate} & \textbf{$\Delta$Reb.\ (s)} & \textbf{$\Delta$VMAF} \\',r'\midrule']
for cfg,name in [('b30_warm_a05','Synth.\\ (main)'),('b12_reset','Synth.\\ ($B_{\\max}{=}12$)'),('irish_b30_warm_a05','Irish 5G'),('lumos_b30_warm_a05','Lumos5G')]:
    for i,(arm,lab) in enumerate([('certified','\\mbox{CPS-S}'),('certified_a','\\mbox{CPS-A}')]):
        q=d[(d.config==cfg)&(d.treat=='cocps:'+arm)&(d.ref=='coreg:'+arm)]
        b=q[q.metric=='bitrate_mean_kbps'].iloc[0]; rb=q[q.metric=='rebuffer_total'].iloc[0]; vm=q[q.metric=='vmaf_mean'].iloc[0]
        L.append(f"{name if i==0 else ''} & {lab} & {cell(num(b.video_mean_pct_change,'{:+.1f}')+'\\%',pp(b.wilcoxon_p))} & {cell(num(rb.video_mean_diff),pp(rb.wilcoxon_p))} & {ci(vm.video_mean_diff,vm.ci95_lo,vm.ci95_hi)}\\\\")
    L.append(r'\midrule' if cfg!='lumos_b30_warm_a05' else r'\bottomrule')
L.append(r'\end{tabular*}'); open(T+'table_r7_codesign.tex','w').write('% r7: co-design (3 seeds per policy), co-trained minus shield-at-eval-only, same certified arm, title level N=12; source results/r7_codesign/codesign_compare\n'+'\n'.join(L)+'\n')
# broadband greedy table from r6 bb_greedy_b12_reset
am=pd.read_csv(R+'r6/arm_means.csv'); am=am[am.run=='bb_greedy_b12_reset']
L=[r'\begin{tabular*}{\linewidth}{@{\extracolsep{\fill}}l cccc@{}}',r'\toprule',r'\textbf{Arm} & \textbf{Reb.\ (s)} & \textbf{VMAF} & \textbf{Bitrate} & \textbf{Cover.} \\',r'\midrule']
for arm,lab in [('raw','\\textsc{Raw}'),('fixed_safety','Fixed-$\\rho$'),('safety','\\textsc{Safety}'),('certified','\\mbox{CPS-S}')]:
    r=am[am.arm==arm].iloc[0]; cov='---' if pd.isna(r['cov']) else f"{r['cov']:.3f}"
    L.append(f"{lab} & {r.reb:.2f} & {r.vmaf:.2f} & {r.br:.0f} & {cov}\\\\")
L+= [r'\bottomrule',r'\end{tabular*}']; open(T+'table_r7_broadband.tex','w').write('% r7: broadband greedy, B12 per-episode reset, alpha 0.10; source results/r6/arm_means.csv (bb_greedy_b12_reset)\n'+'\n'.join(L)+'\n')
