"""Verify causal observations, stored decisions, source hashes and trade accounting."""
import hashlib
import json
from pathlib import Path
import joblib
import numpy as np
import pandas as pd
from threadpoolctl import threadpool_limits
from .data import load,observation
from .evaluate import replay,summarize


def verify(out=Path('rl_agent/results'),source=Path('data/master')):
    out=Path(out);source=Path(source)
    manifest=json.loads((out/'manifest.json').read_text());frozen=json.loads((out/'frozen_selection.json').read_text())
    for name,digest in manifest['source_sha256'].items():assert hashlib.sha256((source/name).read_bytes()).hexdigest()==digest,'Source data changed'
    assert frozen['test_used_for_selection'] is False
    assert (out/'frozen_selection.json').stat().st_mtime<=(out/'test_metrics.csv').stat().st_mtime
    data=load(source);episodes=pd.read_parquet(out/'test_episodes.parquet');actions=pd.read_parquet(out/'test_actions.parquet');metrics=pd.read_csv(out/'test_metrics.csv')
    assert not episodes.duplicated(['policy','episode']).any()
    assert episodes.date.isin(manifest['split_dates']['test']).all()
    assert episodes.groupby('policy').size().eq(1018).all()
    assert not actions.duplicated(['policy','episode','stage']).any()
    assert (actions.loc[actions.action.eq(1),'entry_before']<0).all()
    assert (actions.loc[actions.action.eq(2),'entry_before']>=0).all()
    assert not (actions.action.eq(1)&actions.stage.eq(6)).any()
    later=actions[actions.action.isin([1,2])&actions.stage.gt(0)]
    assert later.fill_index.eq(later.stage+1).all()
    assert actions[actions.action.eq(1)&actions.stage.eq(0)].fill_index.eq(0).all()
    for _,row in episodes.iterrows():
        if row.status=='closed':
            i=int(row.episode);e=int(row.entry_index);x=int(row.exit_index)
            gross=100*(data['price'][i,x]/data['price'][i,e]-1)
            assert e>=0 and x>e and np.isclose(row.gross_pct,gross) and np.isclose(row.net_pct,gross-.5)
        elif row.status=='unresolved':assert pd.isna(row.net_pct)
        else:assert row.status=='no_trade' and row.net_pct==0 and not row.entered
    for policy,d in episodes.groupby('policy'):
        row=metrics[metrics.policy.eq(policy)].iloc[0];m=summarize(d)
        for key,value in m.items():
            if pd.isna(value):assert pd.isna(row[key])
            else:assert np.isclose(row[key],value),(policy,key)
    agent=joblib.load(out/'agents'/f"{frozen['selected_agent']}.joblib")
    ix=np.flatnonzero(data['frame'].date.isin(manifest['split_dates']['test']))
    regenerated,_=replay(agent,data,ix,margin=frozen['selected_margin'],policy_name='SELECTED')
    saved=episodes[episodes.policy.eq('SELECTED')].sort_values('episode').reset_index(drop=True)
    regenerated=regenerated.sort_values('episode').reset_index(drop=True)
    pd.testing.assert_frame_equal(saved,regenerated)
    # Actual learned Q values must ignore every future endpoint/maximum/minimum.
    candidates=ix[data['complete'][ix]][:20]
    t=2;entry=0
    before=agent.values(data,candidates,t,entry)
    altered={**data,'price':data['price'].copy(),'high':data['high'].copy(),'low':data['low'].copy()}
    altered['price'][candidates,t+1:]*=3
    altered['high'][candidates,t+1:]=9999
    altered['low'][candidates,t+1:]=-99
    after=agent.values(altered,candidates,t,entry)
    np.testing.assert_array_equal(observation(data,candidates,t,entry),observation(altered,candidates,t,entry))
    # Parallel tree averaging can differ by floating-point summation roundoff.
    np.testing.assert_allclose(before,after,rtol=1e-12,atol=1e-12)
    np.testing.assert_array_equal(np.argmax(before,axis=1),np.argmax(after,axis=1))
    export=data['frame'].copy();export['episode']=np.arange(len(export))
    selected=episodes[episodes.policy.eq('SELECTED')].copy()
    selected['entry_price']=[data['price'][int(r.episode),int(r.entry_index)] if r.entry_index>=0 else np.nan for r in selected.itertuples()]
    selected['exit_price']=[data['price'][int(r.episode),int(r.exit_index)] if r.exit_index>=0 else np.nan for r in selected.itertuples()]
    selected.to_csv(out/'selected_episodes.csv',index=False)
    selected[selected.status.eq('closed')].to_csv(out/'selected_trades.csv',index=False)
    a=actions[actions.policy.eq('SELECTED')].merge(export,on='episode',validate='many_to_one')
    a['decision_minutes_after_945']=[data['times'][int(r.episode),int(r.stage)] for r in a.itertuples()]
    a['fill_minutes_after_945']=[data['times'][int(r.episode),int(r.fill_index)] if r.fill_index>=0 else np.nan for r in a.itertuples()]
    a['action_name']=[('wait' if r.entry_before<0 else 'hold') if r.action==0 else ('enter' if r.action==1 else 'exit') for r in a.itertuples()]
    a.to_csv(out/'selected_actions.csv',index=False)
    (out/'VERIFIED.txt').write_text('Source hashes, split membership, action masks, delayed fills, P&L, unknown/no-trade accounting, learned-Q future perturbations and exact saved-policy replay passed.\n')
    print(f'Verified {len(episodes):,} policy/episode records and {len(actions):,} decision records; exact selected-agent replay and learned-Q causality checks passed.')

if __name__=='__main__':
    with threadpool_limits(limits=2):verify()
