"""Finite-horizon fitted Q-learning with tree-based value functions.

A price-taking simulated market supplies action-counterfactual transitions.
Only completed-horizon observations enter Q(s,a); future prices are rewards/targets.
"""
import numpy as np
from sklearn.ensemble import ExtraTreesRegressor
from .data import observation

WAIT=0
ENTER=1
EXIT=2
COST=.5

def utility(net,mode):
    net=np.asarray(net,dtype=float)
    if mode=='precision':return np.where(net>0,1.,-2.)
    capped=np.clip(net,-50,50)
    if mode=='loss_averse':return np.where(capped<0,3*capped,capped)
    if mode=='return':return capped
    raise ValueError(mode)

class Agent:
    def __init__(self,mode,seed,trees=80):
        self.mode=mode;self.seed=seed;self.trees=trees;self.q={};self.cost=COST

    def values(self,data,ix,t,entry):
        ix=np.asarray(ix,dtype=int);x=observation(data,ix,t,entry)
        out=np.full((len(ix),3),-np.inf)
        if t==6 and entry<0:out[:,WAIT]=0.;return out
        out[:,WAIT]=self.q[(t,WAIT)].predict(x)
        if entry<0 and t<6:out[:,ENTER]=self.q[(t,ENTER)].predict(x)
        if entry>=0:
            out[:,EXIT]=out[:,WAIT] if t==6 else self.q[(t,EXIT)].predict(x)
        return out

    def actions(self,data,ix,t,entry,margin=0.):
        q=self.values(data,ix,t,entry)
        if entry<0:q[:,ENTER]-=margin
        # Ties choose WAIT; no forced trade and no exit/buy action in an invalid state.
        return np.argmax(q,axis=1)

    def fit(self,data,ix,progress=None):
        ix=np.asarray(ix,dtype=int)
        assert data['complete'][ix].all()
        p=data['price'][ix]
        for t in reversed(range(7)):
            entries=[-1]+([0]+list(range(2,t+1)) if t else [])
            xs={a:[] for a in [WAIT,ENTER,EXIT]};ys={a:[] for a in xs}
            for entry in entries:
                x=observation(data,ix,t,entry)
                if entry<0:
                    if t<6:
                        xs[WAIT].append(x);ys[WAIT].append(self.values(data,ix,t+1,-1).max(axis=1))
                        entered=0 if t==0 else t+1
                        xs[ENTER].append(x);ys[ENTER].append(self.values(data,ix,t+1,entered).max(axis=1))
                    # Flat at final decision has exact value zero, not an estimated regression.
                    continue
                future_net=100*(p[:,t+1]/p[:,entry]-1)-self.cost
                terminal=utility(future_net,self.mode)
                xs[WAIT].append(x)
                ys[WAIT].append(terminal if t==6 else self.values(data,ix,t+1,entry).max(axis=1))
                if t<6:xs[EXIT].append(x);ys[EXIT].append(terminal)
            for action in [WAIT,ENTER,EXIT]:
                if not xs[action]:continue
                x=np.concatenate(xs[action]);y=np.concatenate(ys[action])
                model=ExtraTreesRegressor(n_estimators=self.trees,min_samples_leaf=40,max_depth=14,max_features=.85,n_jobs=2,random_state=self.seed+t*11+action)
                model.fit(x,y);self.q[(t,action)]=model
            if progress:progress(t)
        return self
