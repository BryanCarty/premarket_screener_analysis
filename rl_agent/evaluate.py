import numpy as np
import pandas as pd
from modeling.core import wilson
from .agent import WAIT,ENTER,EXIT,COST


def replay(agent,data,ix,margin=0.,policy_name=None):
    ix=np.asarray(ix,dtype=int);rows=[];trace=[]
    # Evaluate grouped states to avoid one estimator call per single stock/day.
    groups={(-1,0):ix};terminal={}
    while groups:
        (entry,t),members=groups.popitem()
        available=np.isfinite(data['price'][members,t])
        if t==0:available[:]=True # 09:45 execution price is hidden at initial decision.
        bad=members[~available]
        for i in bad:terminal[int(i)]={'status':'unresolved','entered':entry>=0,'entry_index':entry,'exit_index':-1,'net_pct':np.nan,'gross_pct':np.nan,'exit_reason':'missing_current_landmark'}
        members=members[available]
        if not len(members):continue
        action=agent.actions(data,members,t,entry,margin)
        for act in [WAIT,ENTER,EXIT]:
            ids=members[action==act]
            if not len(ids):continue
            newentry=entry
            if act==ENTER:newentry=0 if t==0 else t+1
            end=(act==EXIT) or (t==6 and entry>=0)
            fill_index=(0 if t==0 else t+1) if act==ENTER else t+1
            # A wait with no position at final decision needs no future execution price.
            needed=(act in [ENTER,EXIT]) or entry>=0
            valid=np.isfinite(data['price'][ids,fill_index]) if needed else np.ones(len(ids),dtype=bool)
            for i in ids:
                trace.append({'episode':int(i),'stage':t,'entry_before':entry,'action':int(act),'fill_index':fill_index if act in [ENTER,EXIT] else -1})
            for i in ids[~valid]:
                terminal[int(i)]={'status':'unresolved','entered':entry>=0 or act==ENTER,'entry_index':newentry,'exit_index':-1,'net_pct':np.nan,'gross_pct':np.nan,'exit_reason':'missing_execution_or_mark'}
            ids=ids[valid]
            if end:
                gross=100*(data['price'][ids,t+1]/data['price'][ids,newentry]-1)
                for i,g in zip(ids,gross):terminal[int(i)]={'status':'closed','entered':True,'entry_index':newentry,'exit_index':t+1,'net_pct':float(g-COST),'gross_pct':float(g),'exit_reason':'agent_exit' if act==EXIT else 'forced_close'}
            elif t==6:
                for i in ids:terminal[int(i)]={'status':'no_trade','entered':False,'entry_index':-1,'exit_index':-1,'net_pct':0.,'gross_pct':0.,'exit_reason':'never_entered'}
            elif len(ids):
                key=(newentry,t+1)
                groups[key]=np.r_[groups.get(key,np.array([],dtype=int)),ids]
    for i in ix:
        info=terminal[int(i)];r={'episode':int(i),**data['frame'].iloc[i].to_dict(),**info}
        e=info['entry_index'];x=info['exit_index'];r['entry_minute_after_945']=float(data['times'][i,e]) if e>=0 else np.nan;r['exit_minute_after_945']=float(data['times'][i,x]) if x>=0 else np.nan
        r['full_endpoint_coverage']=bool(data['complete'][i]);r['policy']=policy_name or f'{agent.mode}_{agent.seed}';rows.append(r)
    return pd.DataFrame(rows),pd.DataFrame(trace)


def baseline(data,ix,name):
    rows=[]
    for i in ix:
        r={'episode':int(i),**data['frame'].iloc[i].to_dict(),'policy':name,'full_endpoint_coverage':bool(data['complete'][i]),'entry_index':-1,'exit_index':-1,'entry_minute_after_945':np.nan,'exit_minute_after_945':np.nan}
        if name=='cash':r.update(status='no_trade',entered=False,net_pct=0.,gross_pct=0.,exit_reason='cash')
        else:
            end=4 if name=='buy945_exit60' else 7
            p=data['price'][i,[0,end]]
            if not np.isfinite(p).all():r.update(status='unresolved',entered=True,net_pct=np.nan,gross_pct=np.nan,exit_reason='missing_price')
            else:
                g=float(100*(p[1]/p[0]-1));r.update(status='closed',entered=True,net_pct=g-COST,gross_pct=g,exit_reason='fixed_exit',entry_index=0,exit_index=end,entry_minute_after_945=0.,exit_minute_after_945=float(data['times'][i,end]))
        rows.append(r)
    return pd.DataFrame(rows)


def summarize(d):
    closed=d[d.status.eq('closed')];net=closed.net_pct.to_numpy(float);unresolved=int(d.status.eq('unresolved').sum());wins=int((net>0).sum());n=len(net)
    # Unresolved flat episodes might enter later: all unresolved episodes join the conservative denominator.
    low,high=wilson(wins,n+unresolved)
    return {'episodes':len(d),'closed_trades':n,'wins':wins,'losses':int((net<=0).sum()),'unresolved_episodes':unresolved,'unresolved_with_entry':int((d.status.eq('unresolved')&d.entered).sum()),'no_trade':int(d.status.eq('no_trade').sum()),'signal_days':int(closed.date.nunique()),
            'win_precision_known':wins/n if n else np.nan,'win_precision_conservative':wins/(n+unresolved) if n+unresolved else np.nan,'precision_wilson_low':low,'precision_wilson_high':high,
            'mean_trade_net_pct':float(np.mean(net)) if n else np.nan,'median_trade_net_pct':float(np.median(net)) if n else np.nan,'worst_trade_net_pct':float(np.min(net)) if n else np.nan,
            'known_net_sum_per_opportunity_pct':float(np.sum(net)/len(d)),
            'gross5_realized_count':int((closed.gross_pct>=5-1e-8).sum()),
            'mean_trade_net_100bps_pct':float(np.mean(closed.gross_pct-1)) if n else np.nan,'mean_trade_net_200bps_pct':float(np.mean(closed.gross_pct-2)) if n else np.nan}


def bootstrap_days(d,seed=731):
    known=d[d.status.ne('unresolved')].copy();known['wins']=(known.status.eq('closed')&known.net_pct.gt(0)).astype(int);known['trades']=known.status.eq('closed').astype(int)
    g=known.groupby('date')[['net_pct','wins','trades']].sum().to_numpy()
    if not len(g):return {}
    rng=np.random.default_rng(seed);s=g[rng.integers(0,len(g),size=(2000,len(g)))].sum(axis=1);valid=s[:,2]>0
    if not valid.any():return {}
    p=np.quantile(s[valid,1]/s[valid,2],[.025,.975]);m=np.quantile(s[valid,0]/s[valid,2],[.025,.975])
    return {'day_bootstrap_known_precision_low':float(p[0]),'day_bootstrap_known_precision_high':float(p[1]),'day_bootstrap_known_mean_low':float(m[0]),'day_bootstrap_known_mean_high':float(m[1])}
