"""Rebuild a shareable dataset overview from the completed local dataset."""
import argparse
import json
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.ticker import FuncFormatter


def plot(root):
    root=Path(root)
    f=pd.read_parquet(root/'features.parquet')
    s=json.loads((root/'summary.json').read_text())
    audits=pd.DataFrame(s['daily_audits']);audits=audits[audits.target].copy()
    audits['month']=pd.to_datetime(audits.day).dt.to_period('M')
    monthly=audits.groupby('month').eligible.sum()
    horizons=['5','15','30','60','90','120','close'];names=['5 min','15 min','30 min','60 min','90 min','120 min','Close']
    thresholds=[5,10,15,20,30,50]
    p=pd.DataFrame(s['prevalence']).set_index(['horizon','threshold_pct'])
    counts=np.array([[p.loc[(h,t),'positive'] for h in horizons] for t in thresholds],dtype=float)
    rate=counts/len(f)*100
    plt.rcParams.update({'font.family':'DejaVu Sans','font.size':11,'axes.spines.top':False,'axes.spines.right':False,'axes.titleweight':'bold','axes.labelcolor':'#40516a','text.color':'#172b46','xtick.color':'#40516a','ytick.color':'#40516a','savefig.facecolor':'#f6f8fc'})
    fig=plt.figure(figsize=(17,12),facecolor='#f6f8fc')
    grid=fig.add_gridspec(2,2,left=.075,right=.955,bottom=.14,top=.78,hspace=.45,wspace=.30,height_ratios=[.8,1.15])
    fig.text(.075,.94,'PRE-MARKET RUNNER DATASET',fontsize=26,weight='bold')
    fig.text(.075,.905,'29 Oct 2024 – 28 Oct 2026  •  Features at 09:45 US Eastern  •  Historical observations',fontsize=12,color='#536781')
    for x,value,label in [(.075,f'{len(f):,}','eligible stock-days'),(.35,f'{f.ticker.nunique():,}','distinct tickers'),(.60,f"{s['target_days_completed']:,}",'trading sessions'),(.81,'42','runner definitions')]:
        fig.text(x,.845,value,fontsize=26,weight='bold',color='#126b88')
        fig.text(x,.815,label,fontsize=11,color='#536781')
    ax=fig.add_subplot(grid[0,:]);ax.set_facecolor('white')
    x=np.arange(len(monthly));colors=['#89b8ca' if i in [0,len(x)-1] else '#197d9c' for i in x]
    ax.bar(x,monthly.values,color=colors,width=.72,zorder=3)
    ax.set_title('Qualifying stock-days by month',loc='left',pad=14)
    ax.set_ylabel('Stock-days');ax.set_xticks(x,[p.strftime('%b\n%Y') for p in monthly.index],fontsize=9)
    ax.yaxis.set_major_formatter(FuncFormatter(lambda v,p:f'{v:,.0f}'));ax.grid(axis='y',alpha=.16,zorder=0)
    ax.text(1,1.04,'First and last months cover partial calendar months',transform=ax.transAxes,ha='right',fontsize=10,color='#536781')
    ax=fig.add_subplot(grid[1,0]);ax.set_facecolor('white')
    im=ax.imshow(rate,cmap='YlGnBu',vmin=0,vmax=55,aspect='auto')
    for i in range(len(thresholds)):
        for j in range(len(horizons)):
            ax.text(j,i,f'{int(counts[i,j]):,}\n{rate[i,j]:.1f}%',ha='center',va='center',fontsize=9,color='white' if rate[i,j]>30 else '#172b46')
    ax.set_xticks(range(7),names,fontsize=10);ax.set_yticks(range(6),[f'+{t}%' for t in thresholds]);ax.set_ylabel('Upside threshold');ax.set_xlabel('Horizon after 09:45')
    ax.set_title('Observed runners: count and % of all eligible rows',loc='left',pad=16,fontsize=12)
    cb=fig.colorbar(im,ax=ax,orientation='horizontal',pad=.20,fraction=.045,aspect=35)
    cb.set_label('Observed hit rate (%) — lower bound when bars are missing',fontsize=9)
    ax=fig.add_subplot(grid[1,1]);ax.set_facecolor('white')
    pos=np.array([p.loc[(h,10),'positive'] for h in horizons]);neg=np.array([p.loc[(h,10),'negative'] for h in horizons]);unknown=np.array([p.loc[(h,10),'unknown'] for h in horizons])
    y=np.arange(7)
    for vals,left,col,label in [(pos,np.zeros(7),'#168a86','Observed +10% hit'),(neg,pos,'#5e7fa6','Confirmed negative'),(unknown,pos+neg,'#e6b55b','Unknown')]:
        ax.barh(y,vals/len(f)*100,left=left/len(f)*100,color=col,label=label,height=.65)
        for i,v in enumerate(vals):
            if v/len(f)>.065:ax.text((left[i]+v/2)/len(f)*100,i,f'{v:,}',ha='center',va='center',fontsize=9,color='#172b46' if col=='#e6b55b' else 'white')
    ax.set_yticks(y,names);ax.invert_yaxis();ax.set_xlim(0,100);ax.set_xlabel('Share of all eligible stock-days (%)');ax.set_title('Label coverage at the +10% threshold',loc='left',pad=36,fontsize=12)
    ax.legend(loc='lower left',bbox_to_anchor=(0,1.01),ncol=3,frameon=False,fontsize=8.5,borderaxespad=0)
    fig.text(.075,.075,'Eligibility: pre-market price gain ≥10% AND pre-market volume ≥5× the previous 10-session average. Natural class distribution retained.',fontsize=10,color='#536781')
    fig.text(.075,.052,'Entry proxy: 09:45 minute open. A runner reaches the threshold at any point within its horizon; this is not the end-of-window return.',fontsize=10,color='#536781')
    fig.text(.075,.029,'Unknown = insufficient minute-bar coverage to confirm a negative (or missing entry). Observed hits remain positive. No model or execution costs included.',fontsize=10,color='#536781')
    out=root/'charts';out.mkdir(exist_ok=True)
    for ext in ('png','pdf','svg'):fig.savefig(out/f'dataset_overview.{ext}',dpi=180)
    plt.close(fig)
    print(out/'dataset_overview.png')

if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('root',nargs='?',default='data/master');plot(parser.parse_args().root)
