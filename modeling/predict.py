"""Apply a trusted saved runner model to feature rows with the same schema."""
import argparse
from pathlib import Path
import joblib
import numpy as np
import pandas as pd
from threadpoolctl import threadpool_limits

def main(args):
    artifact=joblib.load(args.model)
    f=pd.read_parquet(args.features)
    cols=artifact['features']
    missing=set(cols+['date','ticker','identity','opening_missing_minutes'])-set(f.columns)
    if missing:raise ValueError(f'Missing features: {sorted(missing)}')
    x=f[cols].replace([np.inf,-np.inf],np.nan)
    eligible=f.opening_missing_minutes.eq(0)
    out=f[['date','ticker','identity']].copy();out['in_model_cohort']=eligible
    out['score']=np.nan
    if eligible.any():out.loc[eligible,'score']=artifact['estimator'].predict_proba(x[eligible])[:,1]
    out['cutoff']=artifact['policy']['cutoff'];out['crosses_research_cutoff']=eligible&out.score.ge(out.cutoff)
    out['threshold_met_validation_target']=artifact['policy']['qualified']
    out['definition']=artifact['definition']
    Path(args.output).parent.mkdir(parents=True,exist_ok=True)
    out.to_parquet(args.output,compression='zstd',index=False)
    print(f'Scored {int(eligible.sum())} eligible rows. Wrote {args.output}')

if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--model',required=True);p.add_argument('--features',required=True);p.add_argument('--output',required=True)
    with threadpool_limits(limits=2):main(p.parse_args())
