"""Font-consistent regeneration of Fig. 3 (fig_r7_main), Fig. 4 (coverage drift), Fig. 6 (ablation) from mk3.py at natural printed width.
Same data/code path as mk3.py; tables are written to a scratch dir only."""
from pathlib import Path as _P; import os as _os; _ROOT=_P(__file__).resolve().parents[1]; _os.makedirs(_ROOT/'outputs/tables',exist_ok=True); _os.makedirs(_ROOT/'outputs/figures',exist_ok=True)
import pandas as pd, numpy as np, json, glob, matplotlib; matplotlib.use('Agg'); import matplotlib.pyplot as plt
plt.rcParams.update({'font.family':'serif','font.serif':['Times New Roman','Liberation Serif','Nimbus Roman'],'mathtext.fontset':'stix','pdf.fonttype':42,'ps.fonttype':42,'font.size':8,'axes.labelsize':8,'xtick.labelsize':7,'ytick.labelsize':7,'legend.fontsize':7})
import sys; sys.path.insert(0,str(_ROOT)); sys.path.insert(0,str(_ROOT)+'/analysis')
from analyse import holm
R=str(_ROOT)+'/results/'; T=str(_ROOT)+'/outputs/tables/'; import os; os.makedirs(T,exist_ok=True); import shutil; None; F=str(_ROOT)+'/outputs/figures/'
M=pd.read_csv(str(_ROOT)+'/analysis/'+'arm_means.csv'); C=pd.read_csv(str(_ROOT)+'/analysis/'+'contrasts_holm.csv'); RHO=json.load(open(str(_ROOT)+'/analysis/'+'rho_selection.json'))
def num(x,f='{:+.2f}'): return f.format(x).replace('-','$-$').replace('+','$+$')
def pp(x): return f'$p{{=}}{x:.3f}$' if x>=0.001 else '$p{<}0.001$'
def pt(x): return f'{x:.3f}' if x>=0.001 else r'$<$0.001'
def cell(a,b): return f'\\makecell[c]{{{a}\\\\[-1pt]\\scriptsize {b}}}'
def ci(m,lo,hi): return f'\\makecell[c]{{{num(m)}\\\\[-1pt]\\scriptsize\\ensuremath{{[{lo:.2f},\\,{hi:.2f}]}}}}'
H=[('greedy','Greedy'),('bba','BBA'),('bola','BOLA'),('mpc','RobustMPC')]
ARMS=[('raw','\\textsc{Raw}'),('fixed_safety','Fixed-$\\rho$ (0.8)'),('tfixed_safety','Fixed-$\\rho^\\ast$'),('aci01_safety','ACI ($\\gamma$ 0.01)'),
      ('aci05_safety','ACI ($\\gamma$ 0.05)'),('safety','\\textsc{Safety}'),('certified','\\mbox{CPS-S}'),('certified_a','\\mbox{CPS-A}'),
      ('certified_cf','\\mbox{CPS-S-CF}'),('certified_a_cf','\\mbox{CPS-A-CF}')]
def m(pool,h,arm): return M[(M.run==f'{pool}_{h}_b30_warm_a05')&(M.arm==arm)].iloc[0]
def abs_table(pools, fn, note, hosts=H):
    if len(pools)==1:
        L=[r'\begin{tabular*}{\linewidth}{@{\extracolsep{\fill}}l l ccccc@{}}',r'\toprule',
           r'\textbf{Host} & \textbf{Arm} & \textbf{Reb.\ (s)} & \textbf{VMAF} & \textbf{Bitrate} & \textbf{Smooth.} & \textbf{Stall-free} \\',r'\midrule']
    else:
        L=[r'\begin{tabular*}{\linewidth}{@{\extracolsep{\fill}}l l cccc cccc@{}}',r'\toprule',
           r' & & \multicolumn{4}{c}{\textbf{Irish 5G}} & \multicolumn{4}{c}{\textbf{Lumos5G}}\\ \cmidrule(lr){3-6}\cmidrule(lr){7-10}',
           r'\textbf{Host} & \textbf{Arm} & \textbf{Reb.} & \textbf{VMAF} & \textbf{Bitr.} & \textbf{Sm.} & \textbf{Reb.} & \textbf{VMAF} & \textbf{Bitr.} & \textbf{Sm.} \\',r'\midrule']
    for h,n in hosts:
        for i,(a,lab) in enumerate(ARMS):
            cells=[]
            for p in pools:
                r=m(p,h,a); cells+= [f'{r.reb:.2f}',f'{r.vmaf:.2f}',f'{r.br:.0f}',f'{r.smooth:.0f}']+([f'{r.stallfree:.3f}'] if len(pools)==1 else [])
            L.append(f"{n if i==0 else ''} & {lab} & "+' & '.join(cells)+r'\\')
        L.append(r'\midrule' if h!=hosts[-1][0] else r'\bottomrule')
    L.append(r'\end{tabular*}'); open(T+fn,'w').write(f'% review8: {note}; source analysis/arm_means.csv\n'+'\n'.join(L)+'\n')
abs_table(['syn'],'table_r7_main.tex','synthetic 5G main config, all arms')
abs_table(['irish','lumos'],'table_r7_real5g_abs.tex','measured 5G main config, all arms')
abs_table(['bb'],'table_r7_broadband.tex','broadband main config (B30 warm a05), all arms')
VAR=[('certified','\\mbox{CPS-S}'),('certified_a','\\mbox{CPS-A}'),('certified_cf','\\mbox{CPS-S-CF}'),('certified_a_cf','\\mbox{CPS-A-CF}')]
def crow(pool,h,t):
    return C[(C.run==f'{pool}_{h}_b30_warm_a05')&(C.treat==t)&(C.ref=='safety')].iloc[0]
def contr(pools,fn,note):
    L=[r'\begin{tabular*}{\linewidth}{@{\extracolsep{\fill}}l l cccc@{}}',r'\toprule',
       r'\textbf{Host} & \textbf{Variant} & \textbf{$\Delta$Bitrate} & \textbf{$\Delta$Reb.\ (s)} & \textbf{$\Delta$VMAF} & \textbf{TOST $p$} \\',r'\midrule']
    for p in pools:
        pn={'syn':'','irish':'Irish ','lumos':'Lumos '}[p]
        for h,n in H:
            for i,(t,lab) in enumerate(VAR):
                o=crow(p,h,t)
                L.append(f"{pn+n if i==0 else ''} & {lab} & {cell(num(o.br_pct,'{:+.1f}')+'\\%',pp(o.br_p))} & {cell(num(o.reb_diff),pp(o.reb_p))} & {ci(o.vmaf_diff,o.vmaf_lo,o.vmaf_hi)} & {pt(o.tost_p)}\\\\")
            L.append(r'\midrule' if not (p==pools[-1] and h=='mpc') else r'\bottomrule')
    L.append(r'\end{tabular*}'); open(T+fn,'w').write(f'% review8: {note}; unadjusted p (exploratory); source analysis/contrasts_holm.csv\n'+'\n'.join(L)+'\n')
contr(['syn'],'table_r7_contrasts.tex','synthetic contrasts vs Safety'); contr(['irish','lumos'],'table_r7_real5g.tex','measured 5G contrasts vs Safety')
# baselines: Safety minus baseline, reb diff with Holm marker
BL=[('fixed_safety','Fixed-$\\rho$ (0.8)'),('tfixed_safety','Fixed-$\\rho^\\ast$'),('aci01_safety','ACI 0.01'),('aci05_safety','ACI 0.05')]
L=[r'\begin{tabular*}{\linewidth}{@{\extracolsep{\fill}}l l cccc@{}}',r'\toprule',r'\textbf{Pool} & \textbf{Bound} & \textbf{Greedy} & \textbf{BBA} & \textbf{BOLA} & \textbf{RobustMPC} \\',r'\midrule']
for p,pn in [('syn','Synth.\\ 5G'),('irish','Irish 5G'),('lumos','Lumos5G'),('bb','Broadband')]:
    for i,(b,lab) in enumerate(BL):
        cs=[]
        for h,_ in H:
            o=C[(C.run==f'{p}_{h}_b30_warm_a05')&(C.treat=='safety')&(C.ref==b)].iloc[0]
            star='$^{\\ast}$' if o.reb_p<0.05 else ''
            cs.append(cell(num(o.reb_diff)+star, num(o.vmaf_diff)))
        lab2=lab+(f' ({RHO[p]["rho_star"]:.2f})' if b=='tfixed_safety' else '')
        L.append(f"{pn if i==0 else ''} & {lab2} & "+' & '.join(cs)+r'\\')
    L.append(r'\midrule' if p!='bb' else r'\bottomrule')
L.append(r'\end{tabular*}'); open(T+'table_r7_baselines.tex','w').write('% review8: Safety minus baseline (title level); top: Reb s (* raw p<0.05), bottom: VMAF; source analysis/contrasts_holm.csv\n'+'\n'.join(L)+'\n')
# coverage table: inclusive / post-warm-up, main + reset
L=[r'\begin{tabular}{@{}l ccccc c@{}}',r'\toprule',
   r'\textbf{Pool} & \textbf{Fixed-$\rho$} & \textbf{Fixed-$\rho^\ast$} & \textbf{ACI 0.01} & \textbf{ACI 0.05} & \textbf{Conformal} & \textbf{Conformal, reset} \\',
   r' & \multicolumn{5}{c}{main configuration, target 0.95} & target 0.90 \\',r'\midrule']
rc={}
for p in ['syn','irish','lumos','bb']:
    e=pd.read_csv(R+f'analysis/reset_cov/{p}_greedy_b12_reset/episodes.csv'); g=e[e.arm=='safety']
    rc[p]=(g.cov_hits.sum()/g.cov_n.sum(), g.cov_pw_hits.sum()/g.cov_pw_n.sum())
for p,pn in [('syn','Synth.\\ 5G'),('irish','Irish 5G'),('lumos','Lumos5G'),('bb','Broadband')]:
    cs=[]
    for a in ['fixed_safety','tfixed_safety','aci01_safety','aci05_safety','safety']:
        r=m(p,'greedy',a); cs.append(f'{r["cov"]:.3f} / {r.cov_pw:.3f}')
    cs.append(f'{rc[p][0]:.3f} / {rc[p][1]:.3f}')
    L.append(pn+' & '+' & '.join(cs)+r'\\')
L+=[r'\bottomrule',r'\end{tabular}']; open(T+'table_r7_coverage.tex','w').write('% review8: coverage inclusive / post-warm-up (k>=20), greedy host (coverage of fixed/conformal bounds is host independent up to ACI feedback); source analysis\n'+'\n'.join(L)+'\n')
# sensitivity tables (warm B30)
def sm(d):
    e=pd.read_csv(str(_ROOT)+'/analysis/'+'sens/'+d+'/episodes.csv'); g=e.groupby('arm')
    mm=g[['rebuffer_total','vmaf_mean','bitrate_mean_kbps','interv_rate']].mean(); s=g[['cov_hits','cov_n','cov_pw_hits','cov_pw_n']].sum()
    mm['cov']=s.cov_hits/s.cov_n; mm['cov_pw']=s.cov_pw_hits/s.cov_pw_n; return mm
L=['% review8: greedy, synthetic 5G, main config (B30, warm, alpha 0.05); source analysis/sens',r'\begin{tabularx}{\linewidth}{@{}l *{4}{>{\centering\arraybackslash}X}@{}}',r'\toprule',
   r'$\varepsilon$ (pts) & Bitrate saved (\%) & $\Delta$Reb.\ (\%) & $\Delta$VMAF (pts) & Coverage \\',r'\midrule']
sens={}
for e_ in ['0.5','1.0','2.0','4.0']:
    q=sm('eps_'+e_); s,c=q.loc['safety'],q.loc['certified']
    bw=100*(1-c.bitrate_mean_kbps/s.bitrate_mean_kbps); rb=100*(c.rebuffer_total/s.rebuffer_total-1); dv=c.vmaf_mean-s.vmaf_mean; sens[e_]=(bw,rb,dv)
    L.append(f'{e_} & {bw:.1f} & {rb:.1f} & {dv:+.2f} & {s["cov"]:.3f} \\\\')
L+=[r'\bottomrule',r'\end{tabularx}']; open(T+'table_ablation_eps.tex','w').write('\n'.join(L)+'\n')
L=['% review8: greedy, synthetic 5G, B30 warm start; source analysis/sens',r'\begin{tabularx}{\linewidth}{@{}l *{6}{>{\centering\arraybackslash}X}@{}}',r'\toprule',
   r'$\alpha$ & Target & Cover. & Cover.\ ($k{\ge}20$) & Interv.\ (\%) & Reb.\ (s) & Bitrate saved (\%) \\',r'\midrule']
alp={}
for a_ in ['0.02','0.05','0.10','0.20']:
    q=sm('alpha_'+a_); s,c=q.loc['safety'],q.loc['certified']; bw=100*(1-c.bitrate_mean_kbps/s.bitrate_mean_kbps); alp[a_]=(s['cov'],s.cov_pw,s.rebuffer_total,c.rebuffer_total,bw)
    L.append(f'{a_} & {1-float(a_):.2f} & {s["cov"]:.3f} & {s.cov_pw:.3f} & {100*c.interv_rate:.1f} & {c.rebuffer_total:.2f} & {bw:.1f} \\\\')
L+=[r'\bottomrule',r'\end{tabularx}']; open(T+'table_ablation_alpha.tex','w').write('\n'.join(L)+'\n')
fig,ax=plt.subplots(1,2,figsize=(3.46,1.65))
ax[0].plot([0.5,1,2,4],[sens[k][0] for k in sens],'o-',color='#1f4e79',ms=3.5,lw=1.2); ax[0].set_xticks([1,2,3,4]); ax[0].set_xlabel(r'$\varepsilon$ (VMAF points)',fontsize=8); ax[0].set_ylabel('Bitrate saved (%)',fontsize=8)
al=[0.02,0.05,0.10,0.20]; ax[1].plot([1-a for a in al],[alp[k][0] for k in alp],'o-',color='#1f4e79',ms=3.5,lw=1.2,label='Inclusive'); ax[1].plot([1-a for a in al],[alp[k][1] for k in alp],'s--',color='#c44e52',ms=3.5,lw=1.2,label=r'$k\geq20$')
ax[1].plot([0.78,1],[0.78,1],':',color='grey'); ax[1].set_xlabel(r'Target $1-\alpha$',fontsize=8); ax[1].set_ylabel('Coverage',fontsize=8); ax[1].legend(fontsize=7,frameon=False)
for a in ax: a.tick_params(labelsize=7)
for a,l in zip(ax,['(a)','(b)']): a.set_title(l,fontsize=8,pad=2)  # panel labels (advisor)
fig.tight_layout(); fig.savefig(F+'fig_cps_ablation.pdf')
# drift fig (warm start)
ch=pd.read_csv(str(_ROOT)+'/analysis/'+'coverage_drift_warm/chunks.csv'); ch=ch[ch.rolling_coverage.notna()]; y=ch.groupby('chunk').rolling_coverage.mean()
fig,ax=plt.subplots(figsize=(3.46,1.9)); ax.axvspan(28,38,color='#f4cccc',alpha=0.75,label='Injected shock')
ax.plot(y.index,y.values,color='#1f4e79',lw=1.2,label='Rolling mean (20 chunks)'); ax.axhline(0.95,color='#888888',ls='--',lw=0.9,label=r'Target $1-\alpha=0.95$')
ax.set_xlabel('Chunk index'); ax.set_ylabel('Conformal coverage'); ax.set_ylim(0,1.02); ax.set_xlim(left=0); ax.legend(loc='lower right',fontsize=7,framealpha=0.92); fig.tight_layout(); fig.savefig(F+'fig_cps_coverage_drift.pdf')
print('drift min rolling', y.min(), 'during shock min', y.loc[28:37].min())
# main fig
var=[('certified','CPS-S'),('certified_a','CPS-A'),('certified_cf','CPS-S-CF'),('certified_a_cf','CPS-A-CF')]
fig,ax=plt.subplots(1,3,figsize=(7.0,2.2)); x=np.arange(4); w=0.2; col=['#0072B2','#E69F00','#56B4E9','#D55E00']; hat=[None,None,'/////','/////']; plt.rcParams['hatch.linewidth']=0.5; plt.rcParams['hatch.color']='white'  # Okabe-Ito; CF variants hatched (colour-blind/greyscale safe)
for i,(t,k) in enumerate(var):
    br=[-crow('syn',h,t).br_pct for h,_ in H]; rb=[-100*crow('syn',h,t).reb_diff/m('syn',h,'safety').reb for h,_ in H]; vm=[crow('syn',h,t).vmaf_diff for h,_ in H]
    ax[0].bar(x+(i-1.5)*w,br,w,label=k,color=col[i],hatch=hat[i],lw=0); ax[1].bar(x+(i-1.5)*w,rb,w,color=col[i],hatch=hat[i],lw=0); ax[2].bar(x+(i-1.5)*w,vm,w,color=col[i],hatch=hat[i],lw=0)
for a,t in zip(ax,['Bitrate saved (%)','Rebuffering reduced (%)','$\\Delta$VMAF (points)']):
    a.set_xticks(x); a.set_xticklabels([n for _,n in H],fontsize=7,rotation=20); a.set_ylabel(t,fontsize=7); a.tick_params(labelsize=7); a.axhline(0,color='k',lw=0.6)
ax[2].axhline(1,ls='--',color='grey',lw=0.6); ax[2].axhline(-1,ls='--',color='grey',lw=0.6)
h_,l_=ax[0].get_legend_handles_labels(); fig.legend(h_,l_,fontsize=7,frameon=False,ncol=4,loc='upper center'); fig.tight_layout(rect=(0,0,1,0.9)); fig.savefig(F+'fig_r7_main.pdf')
print(sens, alp)

# review8 layout: natural-width tabular for single-column (resizebox) appendix tables
import re as _re
for _fn in ['table_r7_main.tex','table_r7_contrasts.tex','table_r7_real5g.tex','table_r7_real5g_abs.tex','table_r7_baselines.tex','table_r7_broadband.tex']:
    if not _os.path.exists(T+_fn): continue
    _t=open(T+_fn).read()
    _t=_re.sub(r'\\begin\{tabular\*\}\{\\linewidth\}\{@\{\\extracolsep\{\\fill\}\}', r'\\begin{tabular}{@{}', _t).replace(r'\end{tabular*}', r'\end{tabular}')
    open(T+_fn,'w').write(_t)
