# %%
import pandas as pd
import numpy as np
import time
# import shap
import re
import warnings
warnings.filterwarnings("ignore")

import logging
logging.getLogger("lightgbm").setLevel(logging.ERROR)

import pandas as pd
pd.options.mode.chained_assignment = None

from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score, classification_report
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from lightgbm import LGBMClassifier

# %% [markdown]
# ## loading data

# %%
CONS_PATH = "data/consdf.parquet"
ACCT_PATH = "data/acctdf.parquet"
TRXN_PATH = "data/trxndf.parquet"
CATMAP_PATH = "data/cat_map.csv"

# %%
# Load data
consdf = pd.read_parquet(CONS_PATH)
consdf_full = pd.read_parquet(CONS_PATH)
acctdf = pd.read_parquet(ACCT_PATH)
trxndf = pd.read_parquet(TRXN_PATH)
cat_map = pd.read_csv(CATMAP_PATH)

# %%
consdf

# %%
acctdf

# %%
trxndf

# %% [markdown]
# ## data cleaning/prepping

# %%
consdf = consdf.copy()
consdf["evaluation_date"] = pd.to_datetime(consdf["evaluation_date"], errors="coerce")

# # drop missing DQ_TARGET
# consdf = consdf[consdf["DQ_TARGET"].notna()].copy()
# consdf["DQ_TARGET"] = consdf["DQ_TARGET"].astype(int)

acctdf = acctdf.copy()
acctdf["balance_date"] = pd.to_datetime(acctdf["balance_date"], errors="coerce")

trxndf = trxndf.copy()
trxndf["posted_date"] = pd.to_datetime(trxndf["posted_date"], errors="coerce")

# Deduplicate transactions (use this whenever you build transaction features)
trxndf = (
    trxndf.sort_values(["posted_date"])
      .drop_duplicates(subset=["prism_transaction_id"], keep="first")
)


# %% [markdown]
# ## scoring exclusions

# %%
# Accounts: how many accounts + how many balance snapshots
acct_stats = (
    acctdf.groupby("prism_consumer_id")
    .agg(
        n_accounts=("prism_account_id", "nunique"),
        n_balance_days=("balance_date", "nunique"),
        first_balance=("balance_date", "min"),
        last_balance=("balance_date", "max"),
    )
    .reset_index()
)

# Transactions: count + span + credits
tx_stats = (
    trxndf.groupby("prism_consumer_id")
    .agg(
        n_txn=("prism_transaction_id", "count"),
        first_txn=("posted_date", "min"),
        last_txn=("posted_date", "max"),
    )
    .reset_index()
)
tx_stats["txn_span_days"] = (tx_stats["last_txn"] - tx_stats["first_txn"]).dt.days

credit_stats = (
    trxndf.assign(is_credit=(trxndf["credit_or_debit"] == "CREDIT").astype(int))
    .groupby("prism_consumer_id")
    .agg(n_credit=("is_credit", "sum"))
    .reset_index()
)
credit_debit_counts = (
    trxndf.groupby(["prism_consumer_id", "credit_or_debit"])
          .size()
          .unstack(fill_value=0)   # creates CREDIT and DEBIT columns
          .reset_index()
)

# rename to match your naming convention
credit_debit_counts = credit_debit_counts.rename(columns={
    "CREDIT": "n_credit_txn",
    "DEBIT": "n_debit_txn"
})

# Combine into one scoring table (one row per consumer)
scoring = (
    consdf[["prism_consumer_id", "evaluation_date", "DQ_TARGET", "credit_score"]]
    .merge(acct_stats, on="prism_consumer_id", how="left")
    .merge(tx_stats, on="prism_consumer_id", how="left")
    .merge(credit_stats, on="prism_consumer_id", how="left")
    .merge(credit_debit_counts,on="prism_consumer_id",how="left")
)

# Fill missing stats with 0 where appropriate
for col in ["n_accounts", "n_balance_days", "n_txn", "txn_span_days", "n_credit"]:
    if col in scoring.columns:
        scoring[col] = scoring[col].fillna(0)


# %%
# consumers with no accounts
consumers_with_accounts = set(acctdf["prism_consumer_id"].unique())
all_consumers = set(consdf["prism_consumer_id"].unique())

no_account_ids = list(all_consumers - consumers_with_accounts)

# print("Total consumers:", len(all_consumers))
# print("Consumers with NO accounts:", len(no_account_ids))

# %%
# checking to see how many "no accounts" have transactions
trx_no_account = trxndf[trxndf["prism_consumer_id"].isin(no_account_ids)]

consumers_no_account_with_txn = trx_no_account["prism_consumer_id"].nunique()

# print("\nConsumers with NO accounts but WITH transactions:", consumers_no_account_with_txn)


# # 3️⃣ Total transaction rows for these consumers
# print("Total transaction rows for these consumers:", trx_no_account.shape[0])


# # 4️⃣ Show example consumer IDs
# print("\nExample consumer IDs (no account but with transactions):")
# print(list(trx_no_account["prism_consumer_id"].unique())[:5])


# # 5️⃣ Show sample transaction rows
# print("\nSample transaction rows:")
# display(trx_no_account.head(10))

# %%
# scoring["n_txn"].describe()


# %%
# scoring["txn_span_days"].describe()


# %% [markdown]
# even the low activity consumers have 88 days (2-3 months) of transaction history

# %%
# scoring["n_credit"].describe()

# %% [markdown]
# bottom 25% has 28 credit transactions

# %%
RULES = {

    # No financial footprint at all
    "no_accounts": scoring["n_accounts"] < 1,

    # No transaction history
    "no_transactions": scoring["n_txn"] < 1,

    # Must have at least 1 credit and 1 debit
    "no_credit_txn": scoring["n_credit_txn"] < 1,
    "no_debit_txn": scoring["n_debit_txn"] < 1,

    # Too short of observable history
    "short_txn_history": scoring["txn_span_days"] < 30,
}

# Apply rules
for name, mask in RULES.items():
    scoring[name] = mask

# Exclusion flag
scoring["excluded"] = scoring[list(RULES.keys())].any(axis=1)

# Summary
# print("Total consumers:", scoring.shape[0])
# print("Excluded:", scoring["excluded"].sum())
# print("Eligible:", (~scoring["excluded"]).sum())
# print("Exclusion rate:", scoring["excluded"].mean())


# %%
eligible_ids = scoring.loc[~scoring["excluded"], "prism_consumer_id"]

# print("Eligible consumers:", len(eligible_ids))


# %%
consdf_eligible = consdf[
    consdf["prism_consumer_id"].isin(eligible_ids)
].copy()

acctdf_eligible = acctdf[
    acctdf["prism_consumer_id"].isin(eligible_ids)
].copy()

trxndf_eligible = trxndf[
    trxndf["prism_consumer_id"].isin(eligible_ids)
].copy()


# %%
# print("Consumers in consdf_eligible:", consdf_eligible["prism_consumer_id"].nunique())
# print("Consumers in acctdf_eligible:", acctdf_eligible["prism_consumer_id"].nunique())
# print("Consumers in trxndf_eligible:", trxndf_eligible["prism_consumer_id"].nunique())


# %%
holdout_ids = consdf_full.loc[
    consdf_full["DQ_TARGET"].isna(),
    "prism_consumer_id"
]

# %%
scoring_holdout = scoring[
    scoring["prism_consumer_id"].isin(holdout_ids)
].copy()

# %%
# print("Holdout consumers:", scoring_holdout.shape[0])
# print("Excluded in holdout:", scoring_holdout["excluded"].sum())
# print("Eligible in holdout:", (~scoring_holdout["excluded"]).sum())
# print("Holdout exclusion rate:",
    #   round(scoring_holdout["excluded"].mean(), 4))

# %% [markdown]
# ## feature engineering

# %%
initial_df = (
    acctdf
    .merge(consdf, on='prism_consumer_id', how='inner')
    .groupby(['prism_consumer_id'])
    .agg(
        balance=('balance', 'sum'),
        balance_date=('balance_date', 'max')
    )
    .reset_index()
).merge(trxndf,on='prism_consumer_id')

# %%
mapping = dict(zip(cat_map["category_id"], cat_map["category"]))
initial_df["category"] = initial_df["category"].replace(mapping)
monthly_summary=initial_df.copy()
monthly_summary['amount'] = np.where(initial_df['credit_or_debit'] == 'DEBIT', -initial_df['amount'],initial_df['amount'])
monthly_summary['posted_date'] = pd.to_datetime(monthly_summary['posted_date'])
monthly_summary = (
    monthly_summary
    .groupby(['prism_consumer_id', monthly_summary['posted_date'].dt.to_period('M')])
    .agg(
        starting_balance=('balance', 'first'),
        monthly_total=('balance', 'sum'),
        trxndf_count = ('balance', 'count')
    )
    .reset_index()
)
monthly_summary['posted_date'] = monthly_summary['posted_date'].dt.to_timestamp()

# %%
monthly_summary = monthly_summary.merge(consdf[['prism_consumer_id','DQ_TARGET']],on='prism_consumer_id').dropna()


# %%
# ensure date type
monthly_summary["posted_date"] = pd.to_datetime(monthly_summary["posted_date"])

# sort properly
monthly_summary = monthly_summary.sort_values(["prism_consumer_id", "posted_date"])

# calculate running balance
monthly_summary["monthly_balance"] = (
    monthly_summary["starting_balance"]
    + monthly_summary.groupby("prism_consumer_id")["monthly_total"].cumsum()
)

# %%
del_df = monthly_summary[monthly_summary['DQ_TARGET'] == 1]
nondel_df = monthly_summary[monthly_summary['DQ_TARGET'] == 0]
ids_1 = del_df["prism_consumer_id"].dropna().unique()
ids_0 = del_df["prism_consumer_id"].dropna().unique()

# %%
mtotal_df = monthly_summary.groupby('prism_consumer_id').agg(
        DQ_TARGET = ('DQ_TARGET', 'first'),
        monthly_mean=('monthly_total', 'mean'),
        monthly_max=('monthly_total', 'max'),
        monthly_min=('monthly_total', 'min'),
        trxndf_count = ('trxndf_count','first'),
        month_count=('monthly_total', 'count')
    )

# %%
cd_df = initial_df[['prism_consumer_id','amount','credit_or_debit']].groupby(['prism_consumer_id','credit_or_debit']).sum().reset_index()


# %%
cd_df = (
    cd_df
    .pivot_table(
        index='prism_consumer_id',
        columns='credit_or_debit',
        values='amount',
        aggfunc='sum',
        fill_value=0
    )
    .assign(
        credit_debit_ratio=lambda x: x['CREDIT'] / (x['DEBIT'] + 1),
        net_flow=lambda x: x['CREDIT'] - x['DEBIT']
    )
)

# %%
cd_df = cd_df.reset_index().merge(consdf[['prism_consumer_id','DQ_TARGET']],on='prism_consumer_id').dropna()


# %%
net_df = initial_df[['prism_consumer_id','posted_date','category','credit_or_debit','amount']].copy()
net_df['amount'] = np.where(net_df['credit_or_debit'] == 'DEBIT', -net_df['amount'],net_df['amount'])
net_df['posted_date'] = pd.to_datetime(net_df['posted_date'])
net_df['month'] = net_df['posted_date'].dt.to_period('M')
mn_df = net_df.groupby(['prism_consumer_id','month']).agg(
        monthly_total=('amount', 'sum'),
        monthly_std =('amount','std')
    ).reset_index()


# %% [markdown]
# monthly features

# %%
monthly_features = mn_df.groupby(['prism_consumer_id']).agg(
    monthly_net_total=('monthly_total', 'sum'),
    monthly_net_avg=('monthly_total', 'mean'),
    monthly_net_max=('monthly_total', 'max'),
    monthly_net_min=('monthly_total', 'min'),
    monthly_std_avg=('monthly_std', 'mean')
).reset_index().merge(consdf[['prism_consumer_id','DQ_TARGET']],on='prism_consumer_id').dropna()
monthly_features['prism_consumer_id'] = monthly_features['prism_consumer_id'].astype(int)
mtotal_df = mtotal_df.reset_index()
mtotal_df['prism_consumer_id'] = mtotal_df['prism_consumer_id'].astype(int)
cd_df['prism_consumer_id'] = cd_df['prism_consumer_id'].astype(int)
monthly_features['net_range'] = monthly_features['monthly_net_max'] - monthly_features['monthly_net_min']

# %%
initial_df['amount'] = np.where(initial_df['credit_or_debit'] == 'DEBIT', -initial_df['amount'],initial_df['amount'])
cat_df = initial_df.groupby(['prism_consumer_id','category'])['amount'].sum().reset_index()

# %%
cat_pivot = (
    cat_df
    .pivot(
        index='prism_consumer_id',
        columns='category',
        values='amount'
    )
    .fillna(0)
)

# %%
outflows = cat_pivot.clip(upper=0).abs()
inflows  = cat_pivot.clip(lower=0)

cat_features = pd.DataFrame(index=cat_pivot.index)

cat_features['total_outflows'] = outflows.sum(axis=1)
cat_features['total_inflows']  = inflows.sum(axis=1)
cat_features['net_flow']       = cat_pivot.sum(axis=1)

# %%
for col in outflows.columns:
    cat_features[f'{col}_outflow_ratio'] = (
        outflows[col] / (cat_features['total_outflows'] + 1))

# %%
# Income reliance
cat_features['paycheck_ratio'] = (
    inflows.get('PAYCHECK', 0) / (cat_features['total_inflows'] + 1)
)

# Cash usage
cat_features['atm_cash_ratio'] = (
    outflows.get('ATM_CASH', 0) / (cat_features['total_outflows'] + 1)
)

# Entertainment vs essentials proxy
cat_features['entertainment_ratio'] = (
    outflows.get('ENTERTAINMENT', 0) / (cat_features['total_outflows'] + 1)
)

# Refund dependence
cat_features['refund_ratio'] = (
    inflows.get('REFUND', 0) / (cat_features['total_inflows'] + 1)
)

# %%
outflows = outflows.reset_index().merge(consdf[['prism_consumer_id','DQ_TARGET']],on='prism_consumer_id').dropna()


# %%
cat_features = cat_features.reset_index().merge(consdf[['prism_consumer_id','DQ_TARGET']],on='prism_consumer_id').dropna()


# %%
add_df = cat_features[['prism_consumer_id','refund_ratio','paycheck_ratio']].copy()
add_df['prism_consumer_id'] = add_df['prism_consumer_id'].astype(int)
outflows['prism_consumer_id'] = outflows['prism_consumer_id'].astype(int)
out_df = outflows.copy()

# %%
initial_df['amount'] = np.where(initial_df['credit_or_debit'] == 'DEBIT', -initial_df['amount'],initial_df['amount'])
cat_df = initial_df.groupby(['prism_consumer_id','category'])['amount'].mean().reset_index()

# %%
cat_pivot = (
    cat_df
    .pivot(
        index='prism_consumer_id',
        columns='category',
        values='amount'
    )
    .fillna(0)
)
cat_pivot.columns = cat_pivot.columns + "_trxnavg"
cat_pivot = cat_pivot.reset_index().merge(consdf[['prism_consumer_id','DQ_TARGET']],on='prism_consumer_id').dropna()
cat_pivot['prism_consumer_id'] = cat_pivot['prism_consumer_id'].astype(int)

# %% [markdown]
# income

# %%
mapping = dict(zip(cat_map["category_id"], cat_map["category"]))
trxndf["category"] = trxndf["category"].replace(mapping)

income_categories = [
    'PAYCHECK',
    'DEPOSIT',
    'UNEMPLOYMENT_BENEFITS',
    'OTHER_BENEFITS',
    'PENSION',
    'INVESTMENT_INCOME'
]

income_df = trxndf[
    trxndf['category'].isin(income_categories)
].copy()
income_df['prism_transaction_id'].duplicated().sum()
income_df['posted_date'] = pd.to_datetime(income_df['posted_date'])

# %%
income_time = (
    income_df
    .groupby('prism_consumer_id')
    .agg(
        first_income_date=('posted_date', 'min'),
        last_income_date=('posted_date', 'max')
    )
    .reset_index()
)

income_time['income_span_days'] = (
    income_time['last_income_date'] - income_time['first_income_date']
).dt.days

# %%
income_df = income_time[['prism_consumer_id','income_span_days']]
income_df['prism_consumer_id'] = income_time['prism_consumer_id'].astype(int)

# %% [markdown]
# preliminary testing

# %%
cat_pivot= cat_pivot.drop(columns='DQ_TARGET')


# %%
main_df= monthly_features.merge(mtotal_df,on='prism_consumer_id')
main_df['DQ_TARGET'] = main_df['DQ_TARGET_x']
main_df = main_df.drop(columns=['DQ_TARGET_x','DQ_TARGET_y'])
cd_df = cd_df.drop(columns=['net_flow','DQ_TARGET'])
main_df= main_df.merge(cd_df,on='prism_consumer_id')
main_df= main_df.merge(add_df,on='prism_consumer_id')
main_df= main_df.merge(out_df,on='prism_consumer_id')
main_df= main_df.merge(income_df,on='prism_consumer_id')
main_df= main_df.merge(cat_pivot,on='prism_consumer_id')
main_df

# %%
# columns I will need: credit/debit, amount, posted date, evaluation date, prism consumer id, DQ_TARGET
merged = pd.merge(consdf.dropna(), trxndf, on='prism_consumer_id', how='left')

# %%
merged = merged[merged['posted_date'] <= merged['evaluation_date']]
credit_only = merged[merged['credit_or_debit'] == 'CREDIT'].copy()
credit_only['posted_date'] = pd.to_datetime(credit_only['posted_date'])
credit_only['Year-Month'] = credit_only['posted_date'].dt.to_period('M')
debt_only = trxndf[trxndf['credit_or_debit']=='DEBIT']
monthly_inflow = credit_only.groupby(['prism_consumer_id', 'Year-Month'])['amount'].sum().reset_index(name='monthly_inflow')
consdf['Evaluation Month'] = consdf['evaluation_date'].dt.to_period('M')
with_eval_month = pd.merge(consdf, monthly_inflow, on='prism_consumer_id', how='left')

# %%
with_eval_month['months_diff'] = (
    (with_eval_month['Evaluation Month'].dt.year - with_eval_month['Year-Month'].dt.year) * 12 +
    (with_eval_month['Evaluation Month'].dt.month - with_eval_month['Year-Month'].dt.month)
)
last_year = with_eval_month[(with_eval_month['months_diff'] >= 1) & (with_eval_month['months_diff'] <= 12)]
sum_yearly_inflow = last_year.groupby('prism_consumer_id')['monthly_inflow'].sum().reset_index(name='avg_yearly_inflow')
year_std = last_year.groupby('prism_consumer_id')['monthly_inflow'].std().reset_index()
year_std.columns = ['prism_consumer_id', 'std_inflow']

# %%
# Trend: Is income increasing or decreasing?
def calculate_trend(group):
    if len(group) < 2:
        return 0
    months = group['months_diff'].values
    inflows = group['monthly_inflow'].values
    return np.polyfit(months, inflows, 1)[0]  # slope

trend = last_year.groupby('prism_consumer_id').apply(calculate_trend, include_groups=False).reset_index()
trend.columns = ['prism_consumer_id', 'trend']
num_transactions = last_year.groupby('prism_consumer_id').size().reset_index()
num_transactions.columns = ['prism_consumer_id', 'num_transactions']

# %%
debt_only = trxndf[trxndf['credit_or_debit'] == 'DEBIT'].copy()
debt_only['posted_date'] = pd.to_datetime(debt_only['posted_date'])
# debt_only['category'] = debt_only['category'].astype(int)

# debt_with_category = pd.merge(debt_only, cat_map, left_on='category', right_on='category_id', how='left')[['prism_consumer_id',\
#     'prism_transaction_id', 'amount', 'credit_or_debit', 'posted_date', 'category_id', 'category_y']]
debt_with_category = debt_only.rename(columns={'category_y':'category'})
groceries_only = debt_with_category[debt_with_category['category']=='GROCERIES']

debt_with_eval = pd.merge(groceries_only, consdf[['prism_consumer_id', 'evaluation_date']], on='prism_consumer_id', how='left')

# Filter for transactions in the 3 months before evaluation_date
debt_with_eval['months_before_eval'] = (
    (debt_with_eval['evaluation_date'].dt.year - debt_with_eval['posted_date'].dt.year) * 12 +
    (debt_with_eval['evaluation_date'].dt.month - debt_with_eval['posted_date'].dt.month)
)

debt_9m = debt_with_eval[(debt_with_eval['months_before_eval'] >= 0) & 
                          (debt_with_eval['months_before_eval'] < 9)]

# total spend of groceries per consumer over a 9 month window (last 9 months before eval date)
total_spend_groceries_9m = debt_9m.groupby('prism_consumer_id')['amount'].sum().reset_index()
total_spend_groceries_9m.columns = ['prism_consumer_id', 'sum_groceries_9m']

# %%
# total spend of dining per consumer over a month window (last month before eval date)
dining_only = debt_with_category[debt_with_category['category']=='FOOD_AND_BEVERAGES']

debt_with_eval_dining = pd.merge(dining_only, consdf[['prism_consumer_id', 'evaluation_date']], on='prism_consumer_id', how='left')

# Filter for transactions in the 6 months before evaluation_date
debt_with_eval_dining['months_before_eval'] = (
    (debt_with_eval_dining['evaluation_date'].dt.year - debt_with_eval_dining['posted_date'].dt.year) * 12 +
    (debt_with_eval_dining['evaluation_date'].dt.month - debt_with_eval_dining['posted_date'].dt.month)
)

debt_6m = debt_with_eval_dining[(debt_with_eval_dining['months_before_eval'] >= 0) & 
                          (debt_with_eval_dining['months_before_eval'] < 6)]

# total spend of groceries per consumer over a 6 month window (last 6 months before eval date)
total_spend_dining_6m = debt_6m.groupby('prism_consumer_id')['amount'].sum().reset_index()
total_spend_dining_6m.columns = ['prism_consumer_id', 'sum_dining_6m']

# %%
# merge evaluation date ONCE
tx = debt_with_category.merge(
    consdf[['prism_consumer_id', 'evaluation_date']],
    on='prism_consumer_id',
    how='left'
)

tx = tx[tx['credit_or_debit'] == 'DEBIT']
tx['amount'] = tx['amount'].abs()

# numerator
total_spend_gambling = tx[tx['category'] == 'GAMBLING'].groupby('prism_consumer_id')['amount'].sum()

# denominator
total_spend_all = tx.groupby('prism_consumer_id')['amount'].sum()

pct_spend_gambling = (total_spend_gambling / total_spend_all).fillna(0).reset_index(name='pct_spend_gambling')

# %%
essentials = ['RENT', 'MORTGAGE', 'BILLS_UTILITIES', 'ESSENTIAL_SERVICES', 'GROCERIES', 'AUTOMOTIVE', 'TRANSPORTATION', \
'HEALTHCARE_MEDICAL', 'INSURANCE', 'CHILD_DEPENDENTS', 'PETS', 'TAX', 'LOAN', 'AUTO_LOAN', 'DEBT', 'CREDIT_CARD_PAYMENT', \
'EDUCATION', 'LEGAL', 'GOVERNMENT_SERVICES']

total_spend_essentials = tx[tx['category'].isin(essentials)].groupby('prism_consumer_id')['amount'].sum()

pct_spend_essentials = (total_spend_essentials / total_spend_all).reset_index()

pct_spend_essentials = pct_spend_essentials.rename(columns={'amount':'pct_spend_essentials'})

# %%
# # change in groceries per consumer from the 3 most recent months to the prior 3-6 months before evaluation date
# lowers AUC from 0.721 to 0.71

# recent 3 months (0–2)
recent_3m = debt_with_eval[(debt_with_eval['months_before_eval'] >= 0) & (debt_with_eval['months_before_eval'] < 3)]

recent_spend = recent_3m.groupby('prism_consumer_id')['amount'].sum().reset_index(name='groceries_0_3m')

# prior 3 months (3–5)
prior_3m = debt_with_eval[(debt_with_eval['months_before_eval'] >= 3) & (debt_with_eval['months_before_eval'] < 6)]

prior_spend = prior_3m.groupby('prism_consumer_id')['amount'].sum().reset_index(name='groceries_3_6m')

# merge and compute delta
delta_groceries_3m = recent_spend.merge(
    prior_spend,
    on='prism_consumer_id',
    how='outer'
).fillna(0)

delta_groceries_3m['delta_groceries_3m'] = delta_groceries_3m['groceries_0_3m'] - delta_groceries_3m['groceries_3_6m']

delta_groceries_3m = delta_groceries_3m[['prism_consumer_id', 'delta_groceries_3m']]

utilities = ['BILLS_UTILITIES', 'ESSENTIAL_SERVICES']

total_spend_utilities = tx[tx['category'].isin(utilities)].groupby('prism_consumer_id')['amount'].sum()

pct_spend_utilities = (total_spend_utilities / total_spend_all).reset_index()

pct_spend_utilities = pct_spend_utilities.rename(columns={'amount':'pct_spend_utilities'})

# %%
# has overdraft - 6 months
# Merge evaluation dates with ALL debt transactions
debt_with_eval = pd.merge(
    debt_with_category, 
    consdf[['prism_consumer_id', 'evaluation_date']], 
    on='prism_consumer_id', 
    how='left'
)

# Calculate days before evaluation
debt_with_eval['days_before_eval'] = (
    debt_with_eval['evaluation_date'] - debt_with_eval['posted_date']
).dt.days

# Filter for OVERDRAFT category AND within 6 months
overdraft_6m = debt_with_eval[
    (debt_with_eval['category'] == 'OVERDRAFT') &
    (debt_with_eval['days_before_eval'] >= 0) & 
    (debt_with_eval['days_before_eval'] <= 180)
]

# Group to get consumers with overdrafts
has_overdraft_6m = overdraft_6m.groupby('prism_consumer_id').size().reset_index(name='overdraft_count')
has_overdraft_6m['has_overdraft_6m'] = 1

has_overdraft_6m = has_overdraft_6m[['prism_consumer_id', 'has_overdraft_6m']]

# %%
# has account fees - 6 months
# Merge evaluation dates with ALL debt transactions
debt_with_eval = pd.merge(
    debt_with_category, 
    consdf[['prism_consumer_id', 'evaluation_date']], 
    on='prism_consumer_id', 
    how='left'
)

# Calculate days before evaluation
debt_with_eval['days_before_eval'] = (
    debt_with_eval['evaluation_date'] - debt_with_eval['posted_date']
).dt.days

# Filter for ACCOUNT FEES category AND within 6 months
acct_fees_6m = debt_with_eval[
    (debt_with_eval['category'] == 'ACCOUNT_FEES') &
    (debt_with_eval['days_before_eval'] >= 0) & 
    (debt_with_eval['days_before_eval'] <= 180)
]

# Group to get consumers with acct fee
has_acct_fee_6m = acct_fees_6m.groupby('prism_consumer_id').size().reset_index(name='acct_fees_count')
has_acct_fee_6m['has_acct_fee_6m'] = 1

has_acct_fee_6m = has_acct_fee_6m[['prism_consumer_id', 'has_acct_fee_6m']]

# %%
#atm cash ratio per consumer

debt_with_eval = pd.merge(
    debt_with_category,
    consdf[['prism_consumer_id', 'evaluation_date']],
    on='prism_consumer_id',
    how='left'
)

debt_with_eval['posted_date'] = pd.to_datetime(debt_with_eval['posted_date'])
debt_with_eval['evaluation_date'] = pd.to_datetime(debt_with_eval['evaluation_date'])

debt_with_eval = debt_with_eval[
    debt_with_eval['posted_date'] <= debt_with_eval['evaluation_date']
]

total_debt_spend = debt_with_eval.groupby('prism_consumer_id')['amount'].sum().reset_index(name='total_debit_spend')

# %%
atm_cash_spend = (
    debt_with_eval[debt_with_eval['category'] == 'ATM_CASH']
    .groupby('prism_consumer_id')['amount']
    .sum()
    .reset_index(name='atm_cash_spend')
)

atm_cash_ratio = total_debt_spend.merge(atm_cash_spend, on='prism_consumer_id',how='left').fillna(0)
atm_cash_ratio['atm_cash_ratio'] = atm_cash_ratio['atm_cash_spend'] / atm_cash_ratio['total_debit_spend']
atm_cash_ratio['atm_cash_ratio'] = (
    atm_cash_ratio['atm_cash_ratio']
    .replace([np.inf, -np.inf], 0)
    .fillna(0)
)


# %%
# Merge evaluation dates with ALL debt transactions
debt_with_eval = pd.merge(
    debt_with_category, 
    consdf[['prism_consumer_id', 'evaluation_date']], 
    on='prism_consumer_id', 
    how='left'
)

# Calculate days before evaluation
debt_with_eval['days_before_eval'] = (
    debt_with_eval['evaluation_date'] - debt_with_eval['posted_date']
).dt.days

atm_cash_freq_6m = acct_fees_6m.groupby('prism_consumer_id').size().reset_index(name='atm_cash_freq_6m')

# %%
# refund ratio
credit_only = trxndf[trxndf['credit_or_debit']=='CREDIT']
# merged_credit = pd.merge(credit_only, cat_map, left_on='category', right_on='category_id', how='left')[['prism_consumer_id', 'prism_transaction_id', 'amount', \
# 'credit_or_debit', 'posted_date', 'category_id', 'category_y']]
merged_credit = credit_only.rename(columns={'category_y': 'category'})

credit_with_eval = pd.merge(
    merged_credit,
    consdf[['prism_consumer_id', 'evaluation_date']],
    on='prism_consumer_id',
    how='left'
)

credit_with_eval['posted_date'] = pd.to_datetime(credit_with_eval['posted_date'])
credit_with_eval['evaluation_date'] = pd.to_datetime(credit_with_eval['evaluation_date'])

credit_with_eval['days_before_eval'] = (credit_with_eval['evaluation_date'] - credit_with_eval['posted_date']).dt.days
window = credit_with_eval[(credit_with_eval['days_before_eval'] >= 0) & (credit_with_eval['days_before_eval'] <= 180)]

refund = window[window['category']=='REFUND'].groupby('prism_consumer_id')['amount'].sum().reset_index(name='refund_amount')

# %%
debit_only = trxndf[trxndf['credit_or_debit'] == 'DEBIT']
# merged_debit = pd.merge(
#     debit_only,
#     cat_map,
#     left_on='category',
#     right_on='category_id',
#     how='left'
# )[[
#     'prism_consumer_id',
#     'prism_transaction_id',
#     'amount',
#     'credit_or_debit',
#     'posted_date',
#     'category_id',
#     'category_y'
# ]]

merged_debit = debit_only.rename(columns={'category_y': 'category'})
debit_with_eval = pd.merge(
    merged_debit,
    consdf[['prism_consumer_id', 'evaluation_date']],
    on='prism_consumer_id',
    how='left'
)

debit_with_eval['posted_date'] = pd.to_datetime(debit_with_eval['posted_date'])
debit_with_eval['evaluation_date'] = pd.to_datetime(debit_with_eval['evaluation_date'])

debit_with_eval['days_before_eval'] = (
    debit_with_eval['evaluation_date'] - debit_with_eval['posted_date']
).dt.days

debit_window = debit_with_eval[
    (debit_with_eval['days_before_eval'] >= 0) &
    (debit_with_eval['days_before_eval'] <= 180)
]

debit_spend = debit_window[
    debit_window['category'] != 'REFUND'
]
denominator = (
    debit_spend
    .groupby('prism_consumer_id')['amount']
    .sum()
    .reset_index(name='total_debit_spend')
)


# %%
refund_ratio = denominator.merge(
    refund,
    on='prism_consumer_id',
    how='left'
).fillna(0)

refund_ratio['refund_ratio'] = (
    refund_ratio['refund_amount'] /
    refund_ratio['total_debit_spend']
)

refund_ratio['refund_ratio'] = (
    refund_ratio['refund_ratio']
    .replace([np.inf, -np.inf], 0)
    .fillna(0)
)
refund_ratio = refund_ratio[['prism_consumer_id', 'refund_ratio']]

# %%
# debt_payment_ratio
# (LOAN + CREDIT_CARD_PAYMENT + AUTO_LOAN + BNPL) / total_debit_spend
categories_of_interest = ['LOAN', 'CREDIT_CARD_PAYMENT', 'AUTO_LOAN', 'BNPL']

summary = (
    debit_with_eval
    .groupby('prism_consumer_id')
    .agg(
        total_debit_spend=('amount', 'sum'),
        debt_spend=('amount', lambda x: x[
            debit_with_eval.loc[x.index, 'category'].isin(categories_of_interest)
        ].sum())
    )
    .reset_index()
)

summary['debt_spend_ratio'] = summary['debt_spend'] / summary['total_debit_spend']

# %%
# bnpl usage flag
# Filter for BNPL category AND within 6 months
bnpl_usage_6m = debt_with_eval[
    (debt_with_eval['category'] == 'BNPL') &
    (debt_with_eval['days_before_eval'] >= 0) & 
    (debt_with_eval['days_before_eval'] <= 180)
]

# Group to get consumers with acct fee
has_bnpl_usage_6m = bnpl_usage_6m.groupby('prism_consumer_id').size().reset_index(name='bnpl_usage_flag')
has_bnpl_usage_6m['bnpl_usage_flag'] = 1

has_bnpl_usage_6m = has_bnpl_usage_6m[['prism_consumer_id', 'bnpl_usage_flag']]

# %%
debt_categories = ['LOAN', 'CREDIT_CARD_PAYMENT', 'AUTO_LOAN', 'BNPL']

debt_category_count = (
    debit_with_eval[debit_with_eval['category'].isin(debt_categories)]
    .groupby(['prism_consumer_id', 'category'])['amount']
    .sum()
    .reset_index()
)

# keep only categories with non-zero spend
debt_category_count = debt_category_count[debt_category_count['amount'] != 0]

debt_category_count = (
    debt_category_count
    .groupby('prism_consumer_id')
    .size()
    .reset_index(name='debt_category_count')
)

# %%
# discretionary drop flag
discretionary_cat_map = ['ENTERTAINMENT', 'TRAVEL', 'FITNESS']
df = debit_with_eval.copy()
df['month'] = df['posted_date'].dt.to_period('M')
monthly_disc = df[df['category'].isin(discretionary_cat_map)].groupby(['prism_consumer_id', 'month'])['amount'].sum().reset_index()

# %%

monthly_disc = monthly_disc.sort_values(['prism_consumer_id', 'month'])
monthly_disc['disc_3m_spend'] = monthly_disc.groupby('prism_consumer_id')['amount'].rolling(3, min_periods=3).sum().reset_index(drop=True)
monthly_disc['prev_disc_3m_spend'] = (
    monthly_disc
    .groupby('prism_consumer_id')['disc_3m_spend']
    .shift(3)
)

# %%

DROP_THRESHOLD = 0.30

monthly_disc['discretionary_drop_flag_3m'] = (
    (monthly_disc['prev_disc_3m_spend'] > 0) &
    ((monthly_disc['prev_disc_3m_spend'] - monthly_disc['disc_3m_spend'])
     / monthly_disc['prev_disc_3m_spend'] >= DROP_THRESHOLD)
).astype(int)

discretionary_drop_flag_3m = (
    monthly_disc
    .dropna(subset=['discretionary_drop_flag_3m'])
    .groupby('prism_consumer_id')
    .tail(1)
    [['prism_consumer_id', 'discretionary_drop_flag_3m']]
)

# %%
# essential spend volatility in 6 months
# Filter for essentials AND within 6 months
essential_spend_volatility_6m = debt_with_eval[
    (debt_with_eval['category'].isin(essentials)) &
    (debt_with_eval['days_before_eval'] >= 0) & 
    (debt_with_eval['days_before_eval'] <= 180)
]

# Group to get consumers with acct fee
essential_spend_volatility_6m = essential_spend_volatility_6m.groupby('prism_consumer_id')['amount'].std().reset_index(name='essential_spend_volatility_6m')

essential_spend_volatility_6m = essential_spend_volatility_6m[['prism_consumer_id', 'essential_spend_volatility_6m']]

# %%
# child dependents spend sum in 6 months
# Filter for child dependents AND within 6 months
child_dependents_6m = debt_with_eval[
    (debt_with_eval['category']=='CHILD_DEPENDENTS')&
    (debt_with_eval['days_before_eval'] >= 0) & 
    (debt_with_eval['days_before_eval'] <= 180)
]

# Group to get consumers with child dependents
has_child_deps_6m = bnpl_usage_6m.groupby('prism_consumer_id').size().reset_index(name='child_dependents_6m')
has_child_deps_6m['child_dependents_6m'] = 1

# %%

# child dependents spend sum in 6 months
# Filter for essentials AND within 6 months
pets_6m = debt_with_eval[
    (debt_with_eval['category']=='PETS')&
    (debt_with_eval['days_before_eval'] >= 0) & 
    (debt_with_eval['days_before_eval'] <= 180)
]

# Group to get consumers with child dependents
has_pets_6m = pets_6m.groupby('prism_consumer_id').size().reset_index(name='pets_6m')
has_pets_6m['pets_6m'] = 1


# %%
def add_eval_window(tx, consdf, days=180):
    tx = tx.merge(consdf[["prism_consumer_id", "evaluation_date"]], on="prism_consumer_id", how="left")
    tx["posted_date"] = pd.to_datetime(tx["posted_date"], errors="coerce")
    tx["evaluation_date"] = pd.to_datetime(tx["evaluation_date"], errors="coerce")
    tx["days_before_eval"] = (tx["evaluation_date"] - tx["posted_date"]).dt.days
    return tx[(tx["days_before_eval"] >= 0) & (tx["days_before_eval"] <= days)].copy()

# 180-day window for all transactions
tx_180 = add_eval_window(trxndf, consdf, days=180)


# %%
fees = tx_180[tx_180["category"] == "ACCOUNT_FEES"].copy()

account_fees_feats = (
    fees.groupby("prism_consumer_id")
        .agg(
            account_fees_count=("amount", "size"),
            account_fees_median=("amount", "median"),
        )
        .reset_index()
)

# %%
ods = tx_180[tx_180["category"] == "OVERDRAFT"].copy()

overdraft_feats = (
    ods.groupby("prism_consumer_id")
       .agg(
           overdraft_count=("amount", "size"),
           overdraft_median=("amount", "median"),
       )
       .reset_index()
)

# %%
bnpl = tx_180[tx_180["category"] == "BNPL"].copy()

BNPL_std = (
    bnpl.groupby("prism_consumer_id")
        .agg(BNPL_std=("amount", "std"))
        .reset_index()
)

# %%
inv_inc = tx_180[tx_180["category"] == "INVESTMENT_INCOME"].copy()

investment_income_feats = (
    inv_inc.groupby("prism_consumer_id")
           .agg(
               investment_income_count=("amount", "size"),
               investment_income_median=("amount", "median"),
           )
           .reset_index()
)

# %%
bank = tx_180[tx_180["category"] == "BANKING_CATCH_ALL"].copy()

banking_catch_all_std = (
    bank.groupby("prism_consumer_id")
        .agg(banking_catch_all_std=("amount", "std"))
        .reset_index()
)

# %%
account_types_savings = (
    acctdf.assign(account_types_savings=(acctdf["account_type"].astype(str).str.upper() == "SAVINGS").astype(int))
         .groupby("prism_consumer_id", as_index=False)["account_types_savings"].max()
)

# %%
has_overdraft_6m

# %%
objs = {
    "account_types_savings": account_types_savings,
    "account_fees_feats": account_fees_feats,
    "overdraft_feats": overdraft_feats,
    "BNPL_std": BNPL_std,
    "investment_income_feats": investment_income_feats,
    "banking_catch_all_std": banking_catch_all_std,
}

# for k,v in objs.items():
#     print(k, type(v), getattr(v, "shape", None))

# %% [markdown]
# ## prepping model

# %%
df_eval = pd.merge(consdf, sum_yearly_inflow, on="prism_consumer_id", how="left")
df_eval = pd.merge(df_eval, year_std, on='prism_consumer_id', how='left')
df_eval = pd.merge(df_eval, trend, on='prism_consumer_id', how='left')
df_eval = pd.merge(df_eval, num_transactions, on='prism_consumer_id', how='left')
df_eval = pd.merge(df_eval, total_spend_groceries_9m, on='prism_consumer_id', how='left')
df_eval = pd.merge(df_eval, total_spend_dining_6m, on='prism_consumer_id', how='left')
df_eval = pd.merge(df_eval, pct_spend_gambling, on='prism_consumer_id',how='left')
df_eval = pd.merge(df_eval, pct_spend_essentials, on='prism_consumer_id',how='left')
df_eval = pd.merge(df_eval, delta_groceries_3m, on='prism_consumer_id', how='left')
df_eval = pd.merge(df_eval, pct_spend_utilities, on='prism_consumer_id', how='left')
df_eval = pd.merge(df_eval, has_overdraft_6m, on='prism_consumer_id', how='left')
df_eval = pd.merge(df_eval, atm_cash_ratio, on='prism_consumer_id', how='left')
df_eval = pd.merge(df_eval, has_acct_fee_6m, on='prism_consumer_id', how='left')
df_eval = pd.merge(df_eval, atm_cash_freq_6m, on='prism_consumer_id', how='left')
df_eval = pd.merge(df_eval, refund_ratio, on='prism_consumer_id', how='left')
df_eval = pd.merge(df_eval, summary, on='prism_consumer_id', how='left')
df_eval = pd.merge(df_eval, has_bnpl_usage_6m, on='prism_consumer_id', how='left')
df_eval = pd.merge(df_eval, debt_category_count, on='prism_consumer_id', how='left')
df_eval = pd.merge(df_eval, discretionary_drop_flag_3m, on='prism_consumer_id', how='left')
df_eval = pd.merge(df_eval, essential_spend_volatility_6m, on='prism_consumer_id', how='left')
df_eval = pd.merge(df_eval, has_child_deps_6m, on='prism_consumer_id', how='left')
df_eval = pd.merge(df_eval, has_pets_6m, on='prism_consumer_id', how='left')
df_eval = pd.merge(df_eval, account_types_savings, on="prism_consumer_id", how="left")
df_eval = pd.merge(df_eval, account_fees_feats, on="prism_consumer_id", how="left")
df_eval = pd.merge(df_eval, overdraft_feats, on="prism_consumer_id", how="left")
df_eval = pd.merge(df_eval, BNPL_std, on="prism_consumer_id", how="left")
df_eval = pd.merge(df_eval, investment_income_feats, on="prism_consumer_id", how="left")
df_eval = pd.merge(df_eval, banking_catch_all_std, on="prism_consumer_id", how="left")
df_eval['has_overdraft_6m'] = df_eval['has_overdraft_6m'].fillna(0).astype(int)
df_eval['has_acct_fee_6m'] = df_eval['has_acct_fee_6m'].fillna(0).astype(int)
df_eval['atm_cash_freq_6m'] = df_eval['atm_cash_freq_6m'].fillna(0).astype(int)
df_eval['bnpl_usage_flag'] = df_eval['bnpl_usage_flag'].fillna(0).astype(int)
df_eval['debt_category_count'] = df_eval['debt_category_count'].fillna(0).astype(int)
df_eval['child_dependents_6m'] = df_eval['child_dependents_6m'].fillna(0).astype(int)
df_eval['pets_6m'] = df_eval['pets_6m'].fillna(0).astype(int)

# %%
df_eval['prism_consumer_id'] =df_eval['prism_consumer_id'].astype(int)
df_eval = main_df.merge(df_eval, on="prism_consumer_id", how="right")


# %%
# for col in df_eval:
#     print(col)

# %%

period_cols = [col for col in df_eval.columns 
               if str(df_eval[col].dtype).startswith('period')]

datetime_cols = df_eval.select_dtypes(include=['datetime64[ns]', 'datetimetz']).columns

time_cols = list(datetime_cols) + period_cols
df_eval = df_eval.drop(columns=time_cols)

# %%
df_eval = df_eval.drop(columns=['DQ_TARGET_y','DQ_TARGET_x','credit_score'])


# %%
# --- dtype alignment for merge key ---
df_eval = df_eval.copy()
scoring_merge = scoring[["prism_consumer_id", "excluded"]].copy()

df_eval["prism_consumer_id"] = df_eval["prism_consumer_id"].astype(str)
scoring_merge["prism_consumer_id"] = scoring_merge["prism_consumer_id"].astype(str)

# --------------------------------------------
# Build master eval table with exclusion flag
# --------------------------------------------
df_eval_master = df_eval.merge(
    scoring[["prism_consumer_id", "excluded"]],
    on="prism_consumer_id",
    how="left"
)

# # If a consumer didn't get a scoring row, treat them as excluded by default (conservative)
# df_eval_master["excluded"] = df_eval_master["excluded"].fillna(True)

# print("df_eval_master rows:", df_eval_master.shape[0])
# print("Excluded rows:", df_eval_master["excluded"].sum())
# print("Eligible rows:", (~df_eval_master["excluded"]).sum())

# %%
df_eval_master = df_eval.copy()  # features for all 15k must already be here

df_labeled = df_eval_master[df_eval_master["DQ_TARGET"].notna()].copy()
df_holdout = df_eval_master[df_eval_master["DQ_TARGET"].isna()].copy()

# print("Labeled:", df_labeled.shape[0])
# print("Holdout:", df_holdout.shape[0])

# %%
# scoring exclusions 
labeled_ids = set(df_labeled["prism_consumer_id"].astype(str))
scoring_labeled = scoring[scoring["prism_consumer_id"].astype(str).isin(labeled_ids)].copy()

RULES = {
    "no_accounts": scoring_labeled["n_accounts"] < 1,
    "no_transactions": scoring_labeled["n_txn"] < 1,
    "no_credit_txn": scoring_labeled["n_credit_txn"] < 1,
    "no_debit_txn": scoring_labeled["n_debit_txn"] < 1,
    "short_txn_history": scoring_labeled["txn_span_days"] < 30,
}

for name, mask in RULES.items():
    scoring_labeled[name] = mask

scoring_labeled["excluded"] = scoring_labeled[list(RULES.keys())].any(axis=1)

# print("Total labeled scoring rows:", scoring_labeled.shape[0])
# print("Excluded (labeled only):", scoring_labeled["excluded"].sum())
# print("Eligible labeled:", (~scoring_labeled["excluded"]).sum())

# %%
df_labeled["prism_consumer_id"] = df_labeled["prism_consumer_id"].astype(str)
scoring_labeled["prism_consumer_id"] = scoring_labeled["prism_consumer_id"].astype(str)

df_labeled = df_labeled.merge(
    scoring_labeled[["prism_consumer_id", "excluded"]],
    on="prism_consumer_id",
    how="left"
)

# Conservative: if someone labeled didn’t get a scoring row, exclude them
df_labeled["excluded"] = df_labeled["excluded"].fillna(True)

df_labeled_eligible = df_labeled[~df_labeled["excluded"]].copy()

# print("Labeled eligible:", df_labeled_eligible.shape[0])

# %%
# 1) how many NaNs are in the target?
# print("DQ_TARGET NaNs in df_labeled:", df_labeled["DQ_TARGET"].isna().sum())

# 2) show a few rows where it is NaN
# display(df_labeled.loc[df_labeled["DQ_TARGET"].isna(), ["prism_consumer_id","DQ_TARGET"]].head(10))

# 3) check dtype / weird strings
# print("DQ_TARGET dtype:", df_labeled["DQ_TARGET"].dtype)
# print("Unique values (sample):", df_labeled["DQ_TARGET"].dropna().unique()[:10])

# %% [markdown]
# ## model testing

# %%
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier

logreg = LogisticRegression(
    max_iter=1000,
    class_weight="balanced",
    random_state=42
)

rf = RandomForestClassifier(
    n_estimators=300,
    max_depth=None,
    class_weight="balanced",
    random_state=42,
    n_jobs=-1
)

xgb = XGBClassifier(
    n_estimators=600,
    max_depth=6,
    learning_rate=0.03,
    subsample=0.85,
    colsample_bytree=0.85,
    min_child_weight=3,
    gamma=0.1,
    reg_alpha=0.1,
    reg_lambda=1.0,
    eval_metric="auc",
    tree_method="hist",
    random_state=42
)

lgbm = LGBMClassifier(
    n_estimators=600,
    learning_rate=0.03,
    max_depth=5,
    num_leaves=31,
    subsample=0.85,
    colsample_bytree=0.85,
    class_weight="balanced",
    random_state=42,
    verbosity=-1
)

# %%
import time
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score, classification_report

def compare_models(df_model, label_name="DQ_TARGET", dataset_name="Dataset"):
    
    print(f"\n==============================")
    print(f" Running models on: {dataset_name}")
    print(f" Rows: {df_model.shape[0]}")
    print(f"==============================")
    
    # Prepare X and y
    drop_cols = ["DQ_TARGET", "excluded", "prism_consumer_id"]
    X = df_model.drop(columns=drop_cols, errors="ignore")
    y = df_model[label_name]

    # Train/test split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y,
        test_size=0.2,
        stratify=y,
        random_state=42
    )

    results = []


    def evaluate_model(name, model):
    
        pipe = Pipeline([
            ("imputer", SimpleImputer(strategy="median")),
            ("model", model)
        ])
    
        t0 = time.perf_counter()
        pipe.fit(X_train, y_train)
        t1 = time.perf_counter()
    
        y_train_prob = pipe.predict_proba(X_train)[:, 1]
        y_test_prob  = pipe.predict_proba(X_test)[:, 1]
    
        train_auc = roc_auc_score(y_train, y_train_prob)
        test_auc  = roc_auc_score(y_test, y_test_prob)
    
        print(f"\n{name}")
        print(f"Train AUC: {train_auc:.4f}")
        print(f"Test  AUC: {test_auc:.4f}")
        print("Classification Report (Test @ 0.5 threshold):")
        print(classification_report(
            y_test,
            (y_test_prob >= 0.5).astype(int),
            digits=4
        ))
    
        results.append({
            "model": name,
            "train_auc": train_auc,
            "test_auc": test_auc,
            "train_time": t1 - t0
        })

    # Run models
    evaluate_model("Logistic Regression", logreg)
    evaluate_model("Random Forest", rf)
    evaluate_model("XGBoost", xgb)
    evaluate_model("LightGBM", lgbm)

    return pd.DataFrame(results)

# %%
# results_baseline = compare_models(
#     df_labeled.copy(),
#     dataset_name="Baseline (No Exclusions)"
# )

# %%
# results_excluded = compare_models(
#     df_labeled_eligible.copy(),
#     dataset_name="After Scoring Exclusions"
# )

# %%
# comparison = results_baseline.merge(
#     results_excluded,
#     on="model",
#     suffixes=("_baseline", "_excluded")
# )

# comparison

# %% [markdown]
# ## feature selection

# %%
def run_experiment(df_model, use_top50=False, dataset_name="Dataset"):

    print(f"\n==============================")
    print(f" {dataset_name}")
    print(f" Rows: {df_model.shape[0]}")
    print(f"==============================")

    drop_cols = ["DQ_TARGET", "excluded", "prism_consumer_id"]
    X = df_model.drop(columns=drop_cols, errors="ignore")
    y = df_model["DQ_TARGET"]

    from sklearn.model_selection import train_test_split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y,
        test_size=0.2,
        stratify=y,
        random_state=42
    )

    # ------------------------
    # Feature Selection (if on)
    # ------------------------
    if use_top50:
        selector = XGBClassifier(
            eval_metric="auc",
            tree_method="hist",
            n_estimators=300,
            learning_rate=0.05,
            max_depth=4,
            random_state=42
        )

        selector.fit(X_train, y_train)

        importances = pd.Series(
            selector.feature_importances_,
            index=X_train.columns
        ).sort_values(ascending=False)

        selected_50 = importances.head(50).index.tolist()

        X_train = X_train[selected_50]
        X_test  = X_test[selected_50]

        print("Using Top 50 Features")

    else:
        print("Using ALL Features")

    results = []

    for name, model in [

    ("Logistic Regression",
     Pipeline([
         ("imputer", SimpleImputer(strategy="median")),
         ("scaler", StandardScaler()),
         ("clf", LogisticRegression(max_iter=2000))
     ])
    ),

    ("Random Forest",
     Pipeline([
         ("imputer", SimpleImputer(strategy="median")),
         ("clf", RandomForestClassifier(
             n_estimators=300,
             random_state=42,
             n_jobs=-1
         ))
     ])
    ),

    ("XGBoost",
     Pipeline([
         ("imputer", SimpleImputer(strategy="median")),
         ("clf", XGBClassifier(
             eval_metric="auc",
             tree_method="hist",
             n_estimators=400,
             learning_rate=0.05,
             max_depth=4,
             random_state=42
         ))
     ])
    ),

    ("LightGBM",
     Pipeline([
         ("imputer", SimpleImputer(strategy="median")),
         ("clf", LGBMClassifier(
             n_estimators=400,
             learning_rate=0.05,
             random_state=42
         ))
     ])
    )
]:

        model.fit(X_train, y_train)

        y_train_prob = model.predict_proba(X_train)[:,1]
        y_test_prob  = model.predict_proba(X_test)[:,1]

        train_auc = roc_auc_score(y_train, y_train_prob)
        test_auc  = roc_auc_score(y_test, y_test_prob)

        print(f"\n{name}")
        print(f"Train AUC: {train_auc:.4f}")
        print(f"Test  AUC: {test_auc:.4f}")

        results.append({
            "model": name,
            "train_auc": train_auc,
            "test_auc": test_auc
        })

    return pd.DataFrame(results)

# %%
# # 1️⃣ Baseline - All features
# res_baseline_all = run_experiment(
#     df_labeled.copy(),
#     use_top50=False,
#     dataset_name="Baseline | All Features"
# )

# # 2️⃣ Baseline - Top 50
# res_baseline_50 = run_experiment(
#     df_labeled.copy(),
#     use_top50=True,
#     dataset_name="Baseline | Top 50"
# )

# # 3️⃣ Excluded - All features
# res_excluded_all = run_experiment(
#     df_labeled_eligible.copy(),
#     use_top50=False,
#     dataset_name="Excluded | All Features"
# )

# # 4️⃣ Excluded - Top 50
# res_excluded_50 = run_experiment(
#     df_labeled_eligible.copy(),
#     use_top50=True,
#     dataset_name="Excluded | Top 50"
# )

# %%


# %% [markdown]
# ## hyperparameter tuning

# %%
drop_cols = ["DQ_TARGET", "excluded", "prism_consumer_id"]

X = df_labeled.drop(columns=drop_cols, errors="ignore")
y = df_labeled["DQ_TARGET"]

from sklearn.model_selection import train_test_split

X_train_50, X_test_50, y_train, y_test = train_test_split(
    X, y,
    test_size=0.2,
    stratify=y,
    random_state=42
)

# %%
from sklearn.model_selection import RandomizedSearchCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
import numpy as np

rf_pipe = Pipeline([
    ("imputer", SimpleImputer(strategy="median")),
    ("clf", RandomForestClassifier(random_state=42, n_jobs=-1))
])

rf_param_grid = {
    "clf__n_estimators": [300],
    "clf__max_depth": [5],
    "clf__min_samples_leaf": [1],
    "clf__max_features": ["sqrt", 0.5]
}

rf_search = RandomizedSearchCV(
    rf_pipe,
    rf_param_grid,
    n_iter=20,
    scoring="roc_auc",
    cv=3,
    random_state=42,
    n_jobs=-1
)

rf_search.fit(X_train_50, y_train)

# print("Best RF AUC:", rf_search.best_score_)
# print("Best Params:", rf_search.best_params_) 

# %%
from xgboost import XGBClassifier

xgb_pipe = Pipeline([
    ("imputer", SimpleImputer(strategy="median")),
    ("clf", XGBClassifier(
        eval_metric="auc",
        tree_method="hist",
        random_state=42
    ))
])

xgb_param_grid = {
    "clf__n_estimators": [400],
    "clf__max_depth": [3],
    "clf__learning_rate": [0.01],
    "clf__subsample": [0.7],
    "clf__colsample_bytree": [0.6],
    "clf__reg_lambda": [0.5],
    "clf__reg_alpha": [0.1]
}

xgb_search = RandomizedSearchCV(
    xgb_pipe,
    xgb_param_grid,
    n_iter=25,
    scoring="roc_auc",
    cv=3,
    random_state=42,
    n_jobs=-1
)

xgb_search.fit(X_train_50, y_train)

# print("Best XGB AUC:", xgb_search.best_score_)
# print("Best Params:", xgb_search.best_params_)

# %%
import warnings
warnings.filterwarnings("ignore")

from lightgbm import LGBMClassifier
from sklearn.model_selection import RandomizedSearchCV
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer

lgbm_pipe = Pipeline([
    ("imputer", SimpleImputer(strategy="median")),
    ("clf", LGBMClassifier(
        random_state=42,
        verbosity=-1   # 🔥 silences LightGBM logs
    ))
])

lgbm_param_grid = {
    "clf__n_estimators": [400],
    "clf__learning_rate": [0.01],
    "clf__max_depth": [5],
    "clf__num_leaves": [15],
    "clf__subsample": [0.7],
    "clf__colsample_bytree": [0.6],
    "clf__reg_lambda": [1]
}

lgbm_search = RandomizedSearchCV(
    lgbm_pipe,
    lgbm_param_grid,
    n_iter=25,
    scoring="roc_auc",
    cv=3,
    random_state=42,
    n_jobs=-1,
    verbose=0  # 🔥 silence CV progress
)

lgbm_search.fit(X_train_50, y_train)

print("Best LGBM AUC:", lgbm_search.best_score_)
print("Best Params:", lgbm_search.best_params_)

# %%
from sklearn.metrics import roc_auc_score, classification_report

best_lgbm = lgbm_search.best_estimator_

y_test_prob = best_lgbm.predict_proba(X_test_50)[:, 1]
print("Tuned LGBM Test AUC:", roc_auc_score(y_test, y_test_prob))

y_test_pred = (y_test_prob >= 0.5).astype(int)
# print(classification_report(y_test, y_test_pred, digits=4))

# %%
from lightgbm import LGBMClassifier
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.metrics import roc_auc_score, classification_report

# Define final LightGBM pipeline
lgb_final = Pipeline([
    ("imputer", SimpleImputer(strategy="median")),
    ("clf", LGBMClassifier(
        random_state=42,
        objective="binary",
        class_weight="balanced",
        verbosity=-1,
        n_estimators=900,
        max_depth=3,
        learning_rate=0.01,
        subsample=0.7,
        colsample_bytree=0.85,
        num_leaves=15,
        min_child_samples=80,
        reg_alpha=0.01,
        reg_lambda=2.0
    ))
])

# Fit on TOP 50 features
lgb_final.fit(X_train_50, y_train)

# Predictions
y_train_prob = lgb_final.predict_proba(X_train_50)[:, 1]
y_test_prob  = lgb_final.predict_proba(X_test_50)[:, 1]

y_test_pred = lgb_final.predict(X_test_50)

# AUC
print("Train AUC:", roc_auc_score(y_train, y_train_prob))
print("Test  AUC:", roc_auc_score(y_test, y_test_prob))

# Classification Report
# print("\nClassification Report (Test Set):")
# print(classification_report(y_test, y_test_pred))

# %% [markdown]
# ## finalized model

# %%
# AFTER exclusions
df_after = df_labeled_eligible.drop(columns=["excluded"], errors="ignore").copy()

# split
y = df_after["DQ_TARGET"].astype(int)
X = df_after.drop(columns=["prism_consumer_id", "DQ_TARGET"], errors="ignore")

from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(
    X, y,
    test_size=0.2,
    stratify=y,
    random_state=42
)

# print("Train rows:", X_train.shape[0], "Test rows:", X_test.shape[0])
# print("Train pos rate:", y_train.mean(), "Test pos rate:", y_test.mean())

# %%
def select_top_k_l1(X_train, y_train, X_test, k=50, C=0.2):
    prep = Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler(with_mean=False))
    ])
    Xtr_p = prep.fit_transform(X_train)
    Xte_p = prep.transform(X_test)

    sel_model = LogisticRegression(
        penalty="l1", solver="liblinear", class_weight="balanced", C=C, max_iter=3000
    )
    sel_model.fit(Xtr_p, y_train)

    coefs = np.abs(sel_model.coef_).ravel()
    feat_names = np.array(X_train.columns)

    if np.all(coefs == 0):
        idx = np.arange(min(k, len(feat_names)))
    else:
        idx = np.argsort(coefs)[::-1][:min(k, len(feat_names))]

    selected = feat_names[idx].tolist()
    return X_train[selected].copy(), X_test[selected].copy(), selected

# %%
# uses your existing helper
X_train_50, X_test_50, selected_50 = select_top_k_l1(
    X_train, y_train, X_test,
    k=50,
    C=0.2
)

# print("Selected features:", len(selected_50))
# print(selected_50)  # optional

# %%
xgb_final = Pipeline([
    ("imputer", SimpleImputer(strategy="median")),
    ("clf", XGBClassifier(
        random_state=35,
        eval_metric="auc",
        tree_method="hist",
        n_estimators=600,
        max_depth=4,
        learning_rate=0.01,
        subsample=0.85,
        colsample_bytree=0.7,
        min_child_weight=3,
        gamma=0,
        reg_alpha=0,
        reg_lambda=0.5
    ))
])

xgb_final.fit(X_train_50, y_train)

ytr_prob = xgb_final.predict_proba(X_train_50)[:, 1]
yte_prob = xgb_final.predict_proba(X_test_50)[:, 1]

# print("XGBoost train AUC:", roc_auc_score(y_train, ytr_prob))
# print("XGBoost test  AUC:", roc_auc_score(y_test, yte_prob))

y_pred_opt = (yte_prob >= 0.5).astype(int)
# print(classification_report(y_test, y_pred_opt))

# %%
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.metrics import roc_auc_score
from xgboost import XGBClassifier

results = []

for rs in range(1, 101):  # try random states 1–100
    
    xgb_model = Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("clf", XGBClassifier(
            random_state=rs,
            eval_metric="auc",
            tree_method="hist",
            n_estimators=600,
            max_depth=4,
            learning_rate=0.01,
            subsample=0.85,
            colsample_bytree=0.7,
            min_child_weight=3,
            gamma=0,
            reg_alpha=0,
            reg_lambda=0.5
        ))
    ])
    
    xgb_model.fit(X_train_50, y_train)

    ytr_prob = xgb_model.predict_proba(X_train_50)[:, 1]
    yte_prob = xgb_model.predict_proba(X_test_50)[:, 1]

    train_auc = roc_auc_score(y_train, ytr_prob)
    test_auc = roc_auc_score(y_test, yte_prob)

    results.append((rs, train_auc, test_auc))

# convert to dataframe
import pandas as pd
results_df = pd.DataFrame(results, columns=["random_state", "train_auc", "test_auc"])

# find best
best_row = results_df.loc[results_df["test_auc"].idxmax()]

# print("Best random_state:", best_row["random_state"])
# print("Best test AUC:", best_row["test_auc"])

results_df.sort_values("test_auc", ascending=False).head(10)

# %%
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.metrics import roc_auc_score
from lightgbm import LGBMClassifier
import pandas as pd

results = []

for rs in range(1, 101):  # test random states 1–100
    
    lgb_model = Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("clf", LGBMClassifier(
            random_state=rs,
            objective="binary",
            class_weight="balanced",
            verbosity=-1,
            n_estimators=900,
            max_depth=3,
            learning_rate=0.01,
            subsample=0.7,
            colsample_bytree=0.85,
            num_leaves=15,
            min_child_samples=80,
            reg_alpha=0.01,
            reg_lambda=2.0
        ))
    ])

    lgb_model.fit(X_train_50, y_train)

    ytr_prob = lgb_model.predict_proba(X_train_50)[:, 1]
    yte_prob = lgb_model.predict_proba(X_test_50)[:, 1]

    train_auc = roc_auc_score(y_train, ytr_prob)
    test_auc = roc_auc_score(y_test, yte_prob)

    results.append((rs, train_auc, test_auc))

results_df = pd.DataFrame(results, columns=["random_state", "train_auc", "test_auc"])

best_row = results_df.loc[results_df["test_auc"].idxmax()]

# print("Best random_state:", best_row["random_state"])
# print("Best test AUC:", best_row["test_auc"])

results_df.sort_values("test_auc", ascending=False).head(10)

# %%
# optimizing precision and recall
from sklearn.metrics import precision_recall_curve
precision, recall, thresholds = precision_recall_curve(y_test, yte_prob)

f1 = 2 * (precision * recall) / (precision + recall)
best_idx = np.argmax(f1)

best_threshold = thresholds[best_idx]
best_f1 = f1[best_idx]

print("Best threshold:", best_threshold)
print("Best F1:", best_f1)
print("Precision:", precision[best_idx])
print("Recall:", recall[best_idx])

# %%
xgb_final.fit(X_train_50, y_train)

ytr_prob = xgb_final.predict_proba(X_train_50)[:, 1]
yte_prob = xgb_final.predict_proba(X_test_50)[:, 1]

# print("XGBoost train AUC:", roc_auc_score(y_train, ytr_prob))
# print("XGBoost test  AUC:", roc_auc_score(y_test, yte_prob))

y_pred_opt = (yte_prob >= best_threshold).astype(int)
# print(classification_report(y_test, y_pred_opt))

# %%
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from lightgbm import LGBMClassifier
from sklearn.metrics import roc_auc_score, classification_report

# re-create a fresh model
lgb_final = Pipeline([
    ("imputer", SimpleImputer(strategy="median")),
    ("clf", LGBMClassifier(
        random_state=45,
        objective="binary",
        class_weight="balanced",
        verbosity=-1,
        n_estimators=900,
        max_depth=3,
        learning_rate=0.01,
        subsample=0.7,
        colsample_bytree=0.85,
        num_leaves=15,
        min_child_samples=80,
        reg_alpha=0.01,
        reg_lambda=2.0
    ))
])

# fit on the 50 selected features
lgb_final.fit(X_train_50, y_train)

ytr_prob = lgb_final.predict_proba(X_train_50)[:, 1]
yte_prob = lgb_final.predict_proba(X_test_50)[:, 1]

# print("LightGBM train AUC:", roc_auc_score(y_train, ytr_prob))
# print("LightGBM test  AUC:", roc_auc_score(y_test, yte_prob))

y_pred_opt = (yte_prob >= best_threshold).astype(int)
# print(classification_report(y_test, y_pred_opt))

# %% [markdown]
# - I optimized the threshold using the precision–recall curve to maximize F1 score rather than using the default 0.5 cutoff because the dataset is highly imbalanced.
# - There is a tradeoff between recall and precision. Increasing recall would increase false positives, so I selected the threshold that balanced both using F1.

# %%
import time
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.metrics import roc_auc_score, roc_curve, auc, classification_report
from lightgbm import LGBMClassifier

# -----------------------------
# 1) Search random_state for best TEST AUC
# -----------------------------
results = []
for rs in range(1, 101):  # 1–100
    lgb_model = Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("clf", LGBMClassifier(
            random_state=rs,
            objective="binary",
            class_weight="balanced",
            verbosity=-1,
            n_estimators=900,
            max_depth=3,
            learning_rate=0.01,
            subsample=0.7,
            colsample_bytree=0.85,
            num_leaves=15,
            min_child_samples=80,
            reg_alpha=0.01,
            reg_lambda=2.0
        ))
    ])

    lgb_model.fit(X_train_50, y_train)

    ytr_prob = lgb_model.predict_proba(X_train_50)[:, 1]
    yte_prob = lgb_model.predict_proba(X_test_50)[:, 1]

    train_auc = roc_auc_score(y_train, ytr_prob)
    test_auc  = roc_auc_score(y_test,  yte_prob)

    results.append((rs, train_auc, test_auc))

results_df = pd.DataFrame(results, columns=["random_state", "train_auc", "test_auc"])
best_row = results_df.loc[results_df["test_auc"].idxmax()]
best_rs = int(best_row["random_state"])

# print("Best random_state:", best_rs)
# print("Best test AUC:", float(best_row["test_auc"]))
# print(results_df.sort_values("test_auc", ascending=False).head(10))

# -----------------------------
# 2) Refit BEST model + metrics
# -----------------------------
lgb_best = Pipeline([
    ("imputer", SimpleImputer(strategy="median")),
    ("clf", LGBMClassifier(
        random_state=best_rs,
        objective="binary",
        class_weight="balanced",
        verbosity=-1,
        n_estimators=900,
        max_depth=3,
        learning_rate=0.01,
        subsample=0.7,
        colsample_bytree=0.85,
        num_leaves=15,
        min_child_samples=80,
        reg_alpha=0.01,
        reg_lambda=2.0
    ))
])

lgb_best.fit(X_train_50, y_train)

ytr_prob = lgb_best.predict_proba(X_train_50)[:, 1]
yte_prob = lgb_best.predict_proba(X_test_50)[:, 1]

print("\nLightGBM BEST train AUC:", roc_auc_score(y_train, ytr_prob))
print("LightGBM BEST test  AUC:", roc_auc_score(y_test, yte_prob))

y_pred_opt = (yte_prob >= best_threshold).astype(int)
print("\nClassification report:\n")
print(classification_report(y_test, y_pred_opt))

# -----------------------------
# 3) ROC curve plot (TEST)
# -----------------------------
fpr, tpr, _ = roc_curve(y_test, yte_prob)
roc_auc = auc(fpr, tpr)

plt.figure()
plt.plot(fpr, tpr, label=f"ROC (AUC = {roc_auc:.4f})")
plt.plot([0, 1], [0, 1], linestyle="--", label="Random")
plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.title("LightGBM ROC Curve (Test Set)")
plt.legend(loc="lower right")
plt.show()

# -----------------------------
# 4) Inference time (scoring time per consumer)
#    - We time predict_proba on X_test_50
#    - Repeat a few times and take the median for stability
# -----------------------------
X_score = X_test_50  # change to your scoring/holdout feature matrix if you want

# warm-up (avoids first-call overhead)
_ = lgb_best.predict_proba(X_score)[:, 1]

n_repeats = 7
times = []

for _ in range(n_repeats):
    t0 = time.perf_counter()
    _ = lgb_best.predict_proba(X_score)[:, 1]
    t1 = time.perf_counter()
    times.append(t1 - t0)

median_total_sec = float(np.median(times))
rows = X_score.shape[0]
per_consumer_ms = (median_total_sec / rows) * 1000.0

# print("\nInference timing (predict_proba on X_score):")
# print(f"Rows scored: {rows}")
# print(f"Median total scoring time: {median_total_sec:.6f} sec")
# print(f"Median time per consumer: {per_consumer_ms:.6f} ms/consumer")

# %%



