"""Train batch-RL agents, choose on earlier data, freeze, then evaluate later dates."""
import os
os.environ.setdefault('OMP_NUM_THREADS','2')
os.environ.setdefault('OPENBLAS_NUM_THREADS','2')
import argparse
import json
import time
from pathlib import Path
import joblib
import numpy as np
import pandas as pd
import sklearn
from threadpoolctl import threadpool_limits
from .data import load,DYNAMIC
from .agent import Agent
from .evaluate import replay,baseline,summarize,bootstrap_days

MODES=['return','loss_averse','precision']
SEEDS=[731,732]
MARGINS=[0.,.1,.25,.5]

def write_json(obj,path):
    path=Path(path);tmp=path.with_suffix('.tmp');tmp.write_text(json.dumps(obj,indent=2,default=lambda o:o.item() if isinstance(o,np.generic) else str(o)));tmp.replace(path)

def rank(metrics):
    support=metrics['closed_trades']>=25 and metrics['signal_days']>=10
    profitable=metrics['known_net_sum_per_opportunity_pct']>0
    bound=metrics['precision_wilson_low'];bound=-1 if not np.isfinite(bound) else bound
    return support and profitable,bound,metrics['known_net_sum_per_opportunity_pct']

def train(args):
    out=Path(args.output);out.mkdir(parents=True,exist_ok=True);(out/'agents').mkdir(exist_ok=True)
    if (out/'test_metrics.csv').exists():raise RuntimeError('Completed test results exist; use a new output directory and disclose holdout reuse.')
    data=load(Path(args.source));f=data['frame'];trainix=np.flatnonzero(f.date.isin(data['splits']['train']).to_numpy()&data['complete']);validix=np.flatnonzero(f.date.isin(data['splits']['validation']));thresholdix=np.flatnonzero(f.date.isin(data['splits']['threshold']));testix=np.flatnonzero(f.date.isin(data['splits']['test']))
    manifest={'algorithm':'Finite-horizon fitted Q-learning with Extra Trees action values; exact backward Bellman backups, gamma=1',
              'source_sha256':data['source_sha256'],'sklearn':sklearn.__version__,'static_features':data['features'],'dynamic_features':DYNAMIC,
              'split_dates':{k:list(v) for k,v in data['splits'].items()},'training_complete_endpoint_episodes':len(trainix),
              'cohort':'All 15 opening minute bars present; no labels/outcome filter for validation/test scoring',
              'split_counts':{k:{'episodes':int(f.date.isin(v).sum()),'complete_endpoints':int((f.date.isin(v).to_numpy()&data['complete']).sum()),'sessions':len(v),'first':str(v[0]),'last':str(v[-1])} for k,v in data['splits'].items()},
              'decision_minutes_after_945':[0,5,15,30,60,90,120],'forced_exit':'Actual regular-session close, including half-days',
              'execution':'Initial entry at hidden 09:45 open; all later ENTER/EXIT instructions fill at NEXT saved landmark endpoint. No current-endpoint same-price fills. One trade maximum, long only, no leverage, forced flat by close.',
              'reward_modes':MODES,'seeds':SEEDS,'entry_margins':MARGINS,'round_trip_cost_percentage_points':.5,
              'reward':'Terminal net return clipped to [-50,+50] for return mode; losses then multiplied by 3 in loss_averse; +1 for net-positive exit and -2 otherwise in precision mode. No-trade reward=0. No mandatory participation.',
              'preprocessing':'Static feature median imputation (with missing flags), fit on complete training episodes only. Dynamic features use only elapsed horizon information.',
              'selection':'Choose agent using model-validation conservative precision Wilson lower bound, requiring 25 closed trades across 10 dates and positive known net per opportunity. Choose entry margin on later threshold period using 80% conservative win target if supported, otherwise strongest supported lower bound. Test is excluded from selection.',
              'missing':'Training requires complete endpoint paths. At validation/test a missing required landmark stops the episode as unresolved, never silently zero-return. Closed trades before missing later landmarks remain valid. All unresolved episodes join conservative precision denominator, including unresolved flat episodes that could have entered later.',
              'holdout_warning':'This chronological test period was already examined in previous supervised and exit studies. RL selection does not use its results, but it is NOT a genuinely unseen independent strategy holdout.',
              'model_parameters':{'n_estimators':80,'min_samples_leaf':40,'max_depth':14,'max_features':.85,'n_jobs':2},
              'literature':'https://jmlr.org/papers/v6/ernst05a.html'}
    write_json(manifest,out/'manifest.json');joblib.dump({'imputer':data['imputer'],'features':data['features']},out/'preprocessing.joblib',compress=3)
    rows=[];started=time.monotonic()
    for mode in MODES:
        for seed in SEEDS:
            name=f'{mode}_seed{seed}';file=out/'agents'/f'{name}.joblib'
            if file.exists():agent=joblib.load(file);print(name,'resumed',flush=True)
            else:
                print('TRAIN',name,flush=True)
                agent=Agent(mode,seed).fit(data,trainix,progress=lambda t:print(name,'Bellman stage',t,flush=True))
                joblib.dump(agent,file,compress=3)
            episodes,_=replay(agent,data,validix,policy_name=name);m=summarize(episodes);rows.append({'agent':name,'mode':mode,'seed':seed,**m})
            pd.DataFrame(rows).to_csv(out/'validation_metrics.csv',index=False)
            print('VALIDATION',name,m,'minutes',round((time.monotonic()-started)/60,1),flush=True)
    winner=max(rows,key=rank);agent=joblib.load(out/'agents'/f"{winner['agent']}.joblib")
    choices=[]
    for margin in MARGINS:
        episodes,_=replay(agent,data,thresholdix,margin=margin,policy_name=winner['agent']);m=summarize(episodes)
        choices.append({'agent':winner['agent'],'margin':margin,**m})
    pd.DataFrame(choices).to_csv(out/'threshold_metrics.csv',index=False)
    qualified=[m for m in choices if rank(m)[0] and m['win_precision_conservative']>=.8 and m['precision_wilson_low']>=.65]
    policy=max(qualified,key=lambda m:m['closed_trades']) if qualified else max(choices,key=rank)
    frozen={'selected_agent':winner['agent'],'selected_margin':policy['margin'],'threshold_qualified_80pct':bool(qualified),'validation_metrics':winner,'threshold_metrics':policy,'test_used_for_selection':False,'note':'Fallbacks remain research-only; any comparison to cash is reported, including abstention or losses.'}
    write_json(frozen,out/'frozen_selection.json')
    print('FROZEN',frozen,flush=True)
    evaluations=[];all_episodes=[];all_trace=[]
    policies=[(r['agent'],0.) for r in rows]+[('SELECTED',policy['margin'])]
    for name,margin in policies:
        path=winner['agent'] if name=='SELECTED' else name
        a=joblib.load(out/'agents'/f'{path}.joblib')
        episodes,trace=replay(a,data,testix,margin=margin,policy_name=name)
        m=summarize(episodes);m.update(bootstrap_days(episodes))
        evaluations.append({'policy':name,'agent':path,'entry_margin':margin,**m});all_episodes.append(episodes);trace['policy']=name;all_trace.append(trace)
        complete_episodes=episodes[episodes.full_endpoint_coverage];cm=summarize(complete_episodes)
        evaluations.append({'policy':name+'_complete_subset','agent':path,'entry_margin':margin,**cm})
    for name in ['cash','buy945_exit60','buy945_exitclose']:
        episodes=baseline(data,testix,name);m=summarize(episodes);m.update(bootstrap_days(episodes));evaluations.append({'policy':name,'agent':'baseline','entry_margin':0.,**m});all_episodes.append(episodes)
        evaluations.append({'policy':name+'_complete_subset','agent':'baseline','entry_margin':0.,**summarize(episodes[episodes.full_endpoint_coverage])})
    pd.DataFrame(evaluations).to_csv(out/'test_metrics.csv',index=False)
    pd.concat(all_episodes,ignore_index=True).to_parquet(out/'test_episodes.parquet',index=False,compression='zstd')
    pd.concat(all_trace,ignore_index=True).to_parquet(out/'test_actions.parquet',index=False,compression='zstd')
    print('TEST evaluation complete',flush=True)
    from .report import report
    report(out)

if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--source',default='data/master');p.add_argument('--output',default='rl_agent/results')
    with threadpool_limits(limits=2):train(p.parse_args())
