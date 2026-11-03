"""Importable custom estimator definitions for stable saved-model loading."""
import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator,RegressorMixin
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import FunctionTransformer,StandardScaler
from modeling.core import signed_log


class FractionalLogit(RegressorMixin,BaseEstimator):
    """Expected occupancy fraction; candle dependence is not treated as independent evidence."""
    def fit(self,x,y):
        self.model_=make_pipeline(FunctionTransformer(signed_log),SimpleImputer(strategy='median',add_indicator=True,keep_empty_features=True),StandardScaler(),LogisticRegression(C=.1,max_iter=2000,random_state=731))
        expanded=pd.concat([x,x],ignore_index=True)
        target=np.r_[np.ones(len(y),dtype=int),np.zeros(len(y),dtype=int)]
        weights=np.r_[np.asarray(y)/100,1-np.asarray(y)/100]
        self.model_.fit(expanded,target,logisticregression__sample_weight=weights)
        return self
    def predict(self,x):return 100*self.model_.predict_proba(x)[:,1]
