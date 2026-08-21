"""Phase 1 evidence gathering: the empirical facts the DGP write-up rests on.

Run:  PYTHONPATH=. python scripts/01_dgp_evidence.py

Every quantitative claim in reports/final/01_dgp.md is produced here, so the
write-up can be re-checked against the data rather than taken on trust.
Output is also persisted to reports/results/01_dgp_evidence.json.
"""
import numpy as np, pandas as pd, json
from src import config
pd.set_option('display.width', 200)
cols = [config.ID_COL, config.TARGET, config.TIME_COL, 'TransactionAmt', 'ProductCD',
        'card1','card2','card3','card4','card5','card6','addr1','addr2',
        'P_emaildomain','R_emaildomain','DeviceType','DeviceInfo','has_identity_record',
        'D1','D2','D3','D10','D15','C1','C13','dist1','dist2']
df = pd.read_parquet(config.INTERIM_DIR/'train_joined.parquet', columns=cols)
df['day'] = (df[config.TIME_COL]//86400).astype(int)

out={}
out['n'] = len(df); out['fraud_rate']=float(df[config.TARGET].mean())
out['n_fraud']=int(df[config.TARGET].sum())
out['days']= [int(df.day.min()), int(df.day.max())]

# --- non-stationarity: weekly fraud rate ---
wk = df.groupby(df.day//7).agg(n=(config.TARGET,'size'), fr=(config.TARGET,'mean'))
out['weekly_fraud_rate_min_max'] = [float(wk.fr.min()), float(wk.fr.max())]
out['weekly_fraud_rate_first4'] = [round(float(x),4) for x in wk.fr.head(4)]
out['weekly_fraud_rate_last4'] = [round(float(x),4) for x in wk.fr.tail(4)]
out['weekly_volume_min_max']=[int(wk.n.min()), int(wk.n.max())]

# --- entity proxy cardinality ---
for c in ['card1','card2','card3','card5','addr1','DeviceInfo','P_emaildomain']:
    out[f'nunique_{c}'] = int(df[c].nunique())
# composite proxy uid
uid = (df['card1'].astype(str)+'_'+df['card2'].astype(str)+'_'+df['card3'].astype(str)
       +'_'+df['card5'].astype(str)+'_'+df['addr1'].astype(str))
out['nunique_uid_card123_5_addr1'] = int(uid.nunique())
vc = uid.value_counts()
out['uid_txn_per_entity'] = {'mean': float(vc.mean()), 'median': float(vc.median()),
                             'p95': float(vc.quantile(.95)), 'max': int(vc.max()),
                             'pct_singleton': float((vc==1).mean())}
# fraud concentration within uid
g = pd.DataFrame({'uid':uid,'y':df[config.TARGET]}).groupby('uid')['y'].agg(['sum','size'])
frauduids = g[g['sum']>0]
out['pct_uids_with_any_fraud'] = float(len(frauduids)/len(g))
out['share_of_fraud_in_multi_fraud_uids'] = float(g.loc[g['sum']>1,'sum'].sum()/g['sum'].sum())

# --- D1 as account-age proxy: D1 vs TransactionDT day ---
out['D1_missing'] = float(df['D1'].isna().mean())
out['corr_D1_day'] = float(np.corrcoef(df.loc[df.D1.notna(),'D1'], df.loc[df.D1.notna(),'day'])[0,1])
out['D1_max']=float(df.D1.max())

# --- selection bias evidence: fraud rate by has_identity_record ---
out['fraud_by_has_identity'] = df.groupby('has_identity_record')[config.TARGET].agg(['mean','size']).to_dict()

# --- ProductCD ---
out['fraud_by_productcd'] = df.groupby('ProductCD', observed=True)[config.TARGET].agg(['mean','size']).sort_values('mean',ascending=False).round(5).to_dict()
# --- DeviceType ---
out['fraud_by_devicetype'] = df.groupby('DeviceType', observed=True, dropna=False)[config.TARGET].agg(['mean','size']).round(5).to_dict()
# --- amount ---
out['amt'] = {k: float(v) for k,v in df.TransactionAmt.describe().items()}
out['amt_fraud_mean']=float(df.loc[df[config.TARGET]==1,'TransactionAmt'].mean())
out['amt_legit_mean']=float(df.loc[df[config.TARGET]==0,'TransactionAmt'].mean())
out['amt_fraud_median']=float(df.loc[df[config.TARGET]==1,'TransactionAmt'].median())
out['amt_legit_median']=float(df.loc[df[config.TARGET]==0,'TransactionAmt'].median())
# cents / round amounts
cents = (df.TransactionAmt - np.floor(df.TransactionAmt))
out['pct_round_amt_all']=float((cents<1e-6).mean())
out['pct_round_amt_fraud']=float((cents[df[config.TARGET]==1]<1e-6).mean())

# --- test-set time gap ---
te = pd.read_parquet(config.INTERIM_DIR/'test_joined.parquet', columns=[config.TIME_COL])
out['train_dt_max_day']=int(df[config.TIME_COL].max()//86400)
out['test_dt_min_day']=int(te[config.TIME_COL].min()//86400)
out['test_dt_max_day']=int(te[config.TIME_COL].max()//86400)

def default(o):
    if isinstance(o,(np.integer,)): return int(o)
    if isinstance(o,(np.floating,)): return float(o)
    return str(o)
txt = json.dumps(out, indent=2, default=default)
print(txt)
(config.RESULTS_DIR / '01_dgp_evidence.json').write_text(txt)
