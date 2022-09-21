# %%
import os
import os.path as osp

import matplotlib
matplotlib.use('agg')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

from plot_utils import get_stats

sns.set_theme(style="whitegrid")

mpl.rcParams["figure.facecolor"] = "white"

par_dir = "/home/t-kenzaamara/internship2022"
import shutil

import scipy.stats as stats

par_dir = "/home/t-kenzaamara/internship2022/"

# %%
model_res = pd.read_csv(osp.join(par_dir, f"logs/mcs_model_scores_350.csv"))
model_res.loc[model_res['explainer'] =='rf', 'loss'] = 'RF'
model_res.columns
##### STATISTICS - SIGNIFICANTLY DIFFERENT? - p-value ######
###############################################################################


# %%
df_stats_350 = get_stats(350)
n_pairs_train_all = df_stats_350['n_pairs_train'].sum()
n_pairs_test_all = df_stats_350['n_pairs_test'].sum()
df_stats_350['w_train'] = df_stats_350['n_pairs_train']/n_pairs_train_all
df_stats_350['w_test'] = df_stats_350['n_pairs_test']/n_pairs_test_all
df_stats_350['n_pairs'] = df_stats_350['n_pairs_test']+df_stats_350['n_pairs_train']
n_pairs_all = df_stats_350['n_pairs'].sum()
df_stats_350['weight'] = df_stats_350['n_pairs']/n_pairs_all

# %%
w_model_res = model_res.groupby(['target', 'loss']).mean().reset_index()
w_model_res = pd.merge(w_model_res, df_stats_350[['target', 'weight', 'n_pairs']], on='target', how='left')
w_model_res.columns
# %%
w_model_res['w_rmse_test'] = w_model_res.apply(lambda x: x.rmse_test*x.weight, axis = 1)
w_model_res['w_pcc_test'] = w_model_res.apply(lambda x: x.pcc_test*x.weight, axis = 1)

# %%
w_model_res[['loss', 'rmse_test', 'pcc_test']].groupby('loss').mean()[['rmse_test', 'pcc_test']]
# %%
w_model_res[['loss', 'rmse_test', 'pcc_test']].groupby('loss').std()[['rmse_test', 'pcc_test']]
# %%
w_stats = w_model_res[['loss', 'w_rmse_test', 'w_pcc_test']].groupby('loss').sum()[['w_rmse_test', 'w_pcc_test']]


# %%
import numpy as np
def weighted_sd(input_df):
    weights = input_df['n_pairs']
    vals = input_df['rmse_test']

    weighted_avg = np.average(vals, weights=weights)
    
    numer = np.sum(weights * (vals - weighted_avg)**2)
    denom = ((vals.count()-1)/vals.count())*np.sum(weights)
    
    return np.sqrt(numer/denom)

def weighted_avg(input_df):
    weights = input_df['n_pairs']
    vals = input_df['rmse_test']

    weighted_avg = np.average(vals, weights=weights)
    return weighted_avg

df_avg_w = w_model_res.groupby('loss').apply(weighted_avg).reset_index()
df_sd_w = w_model_res.groupby('loss').apply(weighted_sd).reset_index()
df_w = pd.merge(df_avg_w, df_sd_w, on='loss', suffixes=('avg', 'sd'))
df_w['y1'] = df_w['0avg']+df_w['0sd']
df_w['y2'] = df_w['0avg']-df_w['0sd']
print(df_w)
#%%
fig, ax = plt.subplots(figsize=(10,5))
sns.lineplot(data=df_w, x='loss', y='0avg', ax=ax,color="rebeccapurple",
    marker="o",
    markersize=10,)
#ax.errorbar(data=df_w, x='loss', y='0avg', yerr='0sd', ls='', lw=3, color='black')
ax.fill_between(data=df_w, x='loss', y1='y1', y2='y2', color='rebeccapurple', alpha=0.2)

ax.set_title("GNN scores: weighted rmse on test set")
ax.set_ylabel("Root Mean Squared Error")
ax.set_xlabel("Parameters")
ax.set_ylim(0, 1)
#plt.xticks(rotation=45)
plt.tight_layout()




# %%
import numpy as np
def weighted_sd(input_df):
    weights = input_df['n_pairs']
    vals = input_df['pcc_test']

    weighted_avg = np.average(vals, weights=weights)
    
    numer = np.sum(weights * (vals - weighted_avg)**2)
    denom = ((vals.count()-1)/vals.count())*np.sum(weights)
    
    return np.sqrt(numer/denom)

def weighted_avg(input_df):
    weights = input_df['n_pairs']
    vals = input_df['pcc_test']

    weighted_avg = np.average(vals, weights=weights)
    return weighted_avg

df_avg_w = w_model_res.groupby('loss').apply(weighted_avg).reset_index()
df_sd_w = w_model_res.groupby('loss').apply(weighted_sd).reset_index()
df_w = pd.merge(df_avg_w, df_sd_w, on='loss', suffixes=('avg', 'sd'))
df_w['y1'] = df_w['0avg']+df_w['0sd']
df_w['y2'] = df_w['0avg']-df_w['0sd']
print(df_w)

#%%
fig, ax = plt.subplots(figsize=(10,5))
sns.lineplot(data=df_w, x='loss', y='0avg', ax=ax,color="darkcyan",
    marker="o",
    markersize=10,)
#ax.errorbar(data=df_w, x='loss', y='0avg', yerr='0sd', ls='', lw=3, color='black')
ax.fill_between(data=df_w, x='loss', y1='y1', y2='y2', color='darkcyan', alpha=0.2)
ax.set_title("GNN scores: weighted pcc on test set")
ax.set_ylabel("Pearson Correlation Coefficient")
ax.set_xlabel("Parameters")
ax.set_ylim(0, 1)
plt.tight_layout()











# %%
# Create pred scores RF vs GNN
rmse_scores_rf, rmse_scores_gnn = [], []
for target in np.unique(w_model_res.target):
    res_target = w_model_res[w_model_res.target == target]
    rmse_rf = res_target[res_target.loss == "RF"]["rmse_test"].values[0]
    rmse_gnn = res_target[res_target.loss == "MSE+UCN"]
    rmse_gnn = rmse_gnn["rmse_test"].values[0]
    if pd.isna(rmse_rf) | pd.isna(rmse_gnn):
        continue
    else:
        rmse_scores_rf.append(rmse_rf)
        rmse_scores_gnn.append(rmse_gnn)
        assert len(rmse_scores_rf) == len(rmse_scores_gnn)


print("stats rmse: RF vs GNN: ", stats.ttest_rel(rmse_scores_rf, rmse_scores_gnn))
print(
    "Number of targets for which GNN rmse < RF rmse: ",
    np.sum([np.array(rmse_scores_rf) - np.array(rmse_scores_gnn) > 0]),
)
print("Total number of targets: ", len(rmse_scores_rf))


# %%
# Create pred scores RF vs GNN - Weighted
rmse_scores_rf, rmse_scores_gnn = [], []
for target in np.unique(w_model_res.target):
    res_target = w_model_res[w_model_res.target == target]
    rmse_rf = res_target[res_target.loss == "RF"]["w_rmse_test"].values[0]
    rmse_gnn = res_target[res_target.loss == "MSE+UCN"]
    rmse_gnn = rmse_gnn["w_rmse_test"].values[0]
    if pd.isna(rmse_rf) | pd.isna(rmse_gnn):
        continue
    else:
        rmse_scores_rf.append(rmse_rf)
        rmse_scores_gnn.append(rmse_gnn)
        assert len(rmse_scores_rf) == len(rmse_scores_gnn)


print("stats rmse weighted: RF vs GNN: ", stats.ttest_rel(rmse_scores_rf, rmse_scores_gnn))
print(
    "Number of targets for which GNN rmse < RF rmse: ",
    np.sum([np.array(rmse_scores_rf) - np.array(rmse_scores_gnn) > 0]),
)
print("Total number of targets: ", len(rmse_scores_rf))

# %%

# Create pred scores RF vs GNN
pcc_scores_rf, pcc_scores_gnn = [], []
for target in np.unique(w_model_res.target):
    res_target = w_model_res[w_model_res.target == target]
    pcc_rf = res_target[res_target.loss == "RF"]["pcc_test"].values[0]
    pcc_gnn = res_target[res_target.loss == "MSE+UCN"]
    pcc_gnn = pcc_gnn["pcc_test"].values[0]
    if pd.isna(pcc_rf) | pd.isna(pcc_gnn):
        continue
    else:
        pcc_scores_rf.append(pcc_rf)
        pcc_scores_gnn.append(pcc_gnn)
        assert len(pcc_scores_rf) == len(pcc_scores_gnn)


print("stats pcc: RF vs GNN: ", stats.ttest_rel(pcc_scores_rf, pcc_scores_gnn))
print(
    "Number of targets for which GNN pcc > RF pcc: ",
    np.sum([np.array(pcc_scores_rf) - np.array(pcc_scores_gnn) < 0]),
)
print("Total number of targets: ", len(pcc_scores_rf))


# %%
######## Create pred scores MSE vs MSE+UCN
rmse_scores_mse, rmse_scores_mse_ucn = [], []
res_gnn = w_model_res
for target in np.unique(res_gnn.target):
    res_target = res_gnn[(res_gnn.target == target)]
    df_1, df_2 = (
        res_target[res_target.loss == "MSE"]["rmse_test"],
        res_target[res_target.loss == "MSE+UCN"]["rmse_test"],
    )
    if df_1.empty | df_2.empty:
        continue
    else:
        rmse_mse = df_1.values[0]
        rmse_mse_ucn = df_2.values[0]
        if pd.isna(rmse_mse) | pd.isna(rmse_mse_ucn):
            continue
        else:
            rmse_scores_mse.append(rmse_mse)
            rmse_scores_mse_ucn.append(rmse_mse_ucn)
            assert len(rmse_scores_mse) == len(rmse_scores_mse_ucn)

print("stats rmse: GNN MSE+UCN vs GNN MSE: ", stats.ttest_rel(rmse_scores_mse, rmse_scores_mse_ucn))
print(
    "Number of targets for which GNN MSE+UCN rmse < GNN MSE rmse: ",
    np.sum([np.array(rmse_scores_mse_ucn) - np.array(rmse_scores_mse) < 0]),
)
print("Total number of targets: ", len(rmse_scores_mse))

# %%
######## Create pred scores MSE vs MSE+UCN - Weighted
rmse_scores_mse, rmse_scores_mse_ucn = [], []
res_gnn = w_model_res
for target in np.unique(res_gnn.target):
    res_target = res_gnn[(res_gnn.target == target)]
    df_1, df_2 = (
        res_target[res_target.loss == "MSE"]["w_rmse_test"],
        res_target[res_target.loss == "MSE+UCN"]["w_rmse_test"],
    )
    if df_1.empty | df_2.empty:
        continue
    else:
        rmse_mse = df_1.values[0]
        rmse_mse_ucn = df_2.values[0]
        if pd.isna(rmse_mse) | pd.isna(rmse_mse_ucn):
            continue
        else:
            rmse_scores_mse.append(rmse_mse)
            rmse_scores_mse_ucn.append(rmse_mse_ucn)
            assert len(rmse_scores_mse) == len(rmse_scores_mse_ucn)

print("stats rmse: GNN MSE+UCN vs GNN MSE: ", stats.ttest_rel(rmse_scores_mse, rmse_scores_mse_ucn))
print(
    "Number of targets for which GNN MSE+UCN rmse < GNN MSE rmse: ",
    np.sum([np.array(rmse_scores_mse_ucn) - np.array(rmse_scores_mse) < 0]),
)
print("Total number of targets: ", len(rmse_scores_mse))

# %%
fig, ax = plt.subplots(figsize=(10,5))
sns.lineplot(
    x="loss",
    y="rmse_test",
    data=w_model_res,
    color="rebeccapurple",
    marker="o",
    markersize=10,
    # err_style="bars",
    ci="sd",
)
ax.set_title("GNN scores: rmse on test set")
ax.set_ylabel("Root Mean Squared Error")
ax.set_xlabel("Parameters")
ax.set_ylim(0, 1)
plt.xticks(rotation=45)
plt.tight_layout()

# %%
fig, ax = plt.subplots(figsize=(10,5))
sns.lineplot(
    x="loss",
    y="w_rmse_test",
    data=w_model_res,
    color="rebeccapurple",
    marker="o",
    markersize=10,
    # err_style="bars",
    ci="sd",
)
ax.set_title("GNN scores: weighted rmse on test set")
ax.set_ylabel("Root Mean Squared Error")
ax.set_xlabel("Parameters")
plt.xticks(rotation=45)
plt.tight_layout()

# %%

fig, ax = plt.subplots(figsize=(10,5))
sns.lineplot(
    x="loss",
    y="pcc_test",
    data=w_model_res,
    label="PCC",
    color="darkcyan",
    marker="o",
    markersize=10,
    ci='sd',
    # err_style="bars",
)
ax.set_title("GNN scores: pcc on test set")
ax.set_ylabel("Pearson Correlation Coefficient")
ax.set_xlabel("Parameters")
ax.set_ylim(0, 1)
plt.xticks(rotation=45)
plt.tight_layout()


# %%

fig, ax = plt.subplots(figsize=(10,5))
sns.lineplot(
    x="loss",
    y="w_pcc_test",
    data=w_model_res,
    label="PCC",
    color="darkcyan",
    marker="o",
    markersize=10,
    ci='sd',
    # err_style="bars",
)
ax.set_title("GNN scores: weighted pcc on test set")
ax.set_ylabel("Pearson Correlation Coefficient")
ax.set_xlabel("Parameters")
plt.xticks(rotation=45)
plt.tight_layout()