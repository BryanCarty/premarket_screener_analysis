import numpy as np
import pandas as pd
from rl_agent.data import observation
from rl_agent.evaluate import replay
from rl_agent.agent import WAIT,ENTER,EXIT,utility

def fixture():
    p=np.array([[100,105,110,121,122,123,124,125]],dtype=float)
    return {'price':p,'static':np.ones((1,3),dtype=np.float32),'high':np.zeros((1,8)),'low':np.zeros((1,8)),'times':np.array([[0,5,15,30,60,90,120,375]],dtype=float),'frame':pd.DataFrame([{'date':'2026-01-01','ticker':'X','identity':'X'}]),'complete':np.array([True])}

class FixedAgent:
    mode='fixed';seed=0
    def __init__(self,decisions):self.decisions=decisions
    def actions(self,data,ix,t,entry,margin=0):return np.full(len(ix),self.decisions.get(t,WAIT))

def test_observations_ignore_future_and_hide_initial_fill():
    d=fixture();before=observation(d,[0],1,-1).copy();zero=observation(d,[0],0,-1).copy()
    d['price'][0,2:]=999;d['high'][0,2:]=999;d['low'][0,2:]=-999
    np.testing.assert_array_equal(before,observation(d,[0],1,-1))
    d['price'][0,0]=888
    np.testing.assert_array_equal(zero,observation(d,[0],0,-1))

def test_later_actions_fill_at_next_landmark():
    d=fixture();episodes,trace=replay(FixedAgent({1:ENTER,2:EXIT}),d,[0])
    r=episodes.iloc[0]
    assert r.entry_index==2 and r.exit_index==3
    assert abs(r.net_pct-9.5)<1e-9

def test_initial_entry_and_forced_close_costs():
    d=fixture();episodes,_=replay(FixedAgent({0:ENTER}),d,[0]);r=episodes.iloc[0]
    assert r.exit_reason=='forced_close' and abs(r.net_pct-24.5)<1e-8
    episodes,_=replay(FixedAgent({}),d,[0]);assert episodes.iloc[0].net_pct==0 and episodes.iloc[0].status=='no_trade'

def test_missing_execution_is_unknown_not_zero():
    d=fixture();d['price'][0,2]=np.nan
    e,_=replay(FixedAgent({1:ENTER}),d,[0]);assert e.iloc[0].status=='unresolved' and np.isnan(e.iloc[0].net_pct)

def test_precision_reward_penalizes_losses_and_respects_no_trade_elsewhere():
    assert utility(np.array([1.,-1.,0.]),'precision').tolist()==[1.,-2.,-2.]
    assert utility(np.array([-2.,2.]),'loss_averse').tolist()==[-6.,2.]
