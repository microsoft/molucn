# %%
import os
import os.path as osp

import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('agg')
import seaborn as sns
from collections import Counter

sns.set_theme(style="whitegrid")

matplotlib.rcParams["figure.facecolor"] = "white"

par_dir = "/home/t-kenzaamara/molucn"
import scipy.stats as stats
import shutil

# Option: plot all targets or sub targets for which GNN rmse < RF rmse


# %%
#### Get list of targets for which GNN rmse (gradinput) < RF rmse
#### and filter to plot performance only for those targets
model_res = pd.read_csv(osp.join(par_dir, f"logs/mcs_model_scores_350.csv"))
attr_res = pd.read_csv(osp.join(par_dir, f"results/mcs_attr_scores_350.csv"))



# %%
# Option: plot all targets or sub targets for which GNN rmse < RF rmse
rf = model_res[model_res.explainer=='rf'].groupby('target').mean().reset_index()
gnn = model_res[model_res.explainer!='rf'].groupby(['target', 'loss']).mean().reset_index()
SUBTARGETS = []
for target in np.unique(model_res.target):
    rf_target=rf[rf.target==target]
    gnn_target=gnn[gnn.target==target]
    rmse_rf = rf_target['rmse_test'].values[0]
    rmse_gnn = gnn_target['rmse_test'].values[0]
    if pd.isna(rmse_rf) | pd.isna(rmse_gnn):
        continue
    else:
        if rmse_gnn < rmse_rf:
            SUBTARGETS.append(target)


sub_attr_res = attr_res[attr_res.target.isin(SUBTARGETS)]
sub_attr_res =sub_attr_res.sort_values(by=['explainer', 'pool','loss'], ascending=[False, False, True])
print(len(sub_attr_res))


#status = dict(Counter(res.target))
#res['status'] = res['target'].map(status)
#res = res[res.status==max(res['status'])]


# %%
##### FEATURE ATTRIBUTION ######
###############################################################################
res_50 = sub_attr_res[sub_attr_res.mcs==50]
# %%
##### Feature attribution - Color agreement ######
###############################################################################

fig, ax = plt.subplots(figsize=(8, 5))
sns.lineplot(
    x="explainer",
    y="acc_train",
    data=res_50[res_50.explainer!='rf'],
    hue="loss",
    marker="o",
    markersize=10,
    ci=30,
    err_style="bars",
)
sns.lineplot(x="explainer",  y="acc_train",
    data=res_50[res_50.explainer=='rf'],
    color="black",
    marker="o", markersize=10,)
ax.set_title("Accuracy - Color agreement - Train")
ax.set_ylabel("Accuracy")
ax.set_xlabel("Feature attribution method")
plt.tight_layout()


fig, ax = plt.subplots(figsize=(8, 5))
sns.lineplot(
    x="explainer",
    y="acc_test",
    data=res_50[res_50.explainer!='rf'],
    hue="loss",
    marker="o",
    markersize=10,
    linestyle="dashed",
    ci=30,
    err_style="bars",
)
sns.lineplot(x="explainer",  y="acc_test",
    data=res_50[res_50.explainer=='rf'],
    color="black",
    marker="o", markersize=10,)
ax.set_title("Accuracy - Color agreement - Test")
ax.set_ylabel("Accuracy")
ax.set_xlabel("Feature attribution method")
plt.tight_layout()


# %%

##### Feature attribution - Global direction ######
###############################################################################


# %%

fig, ax = plt.subplots(figsize=(8, 5))
sns.lineplot(
    x="explainer",
    y="global_dir_train",
    data=res_50[res_50.explainer!='rf'],
    hue="loss",
    marker="o",
    markersize=10,
    ci=30,
    err_style="bars",
)
sns.lineplot(x="explainer",  y="global_dir_train",
    data=res_50[res_50.explainer=='rf'],
    color="black",
    marker="o", markersize=10,)
ax.set_title("Global direction - Train")
ax.set_ylabel("Global direction")
ax.set_xlabel("Feature attribution method")
plt.tight_layout()
#fig.savefig(osp.join(par_dir, f"plots/aggr_global_dir.png"))

fig, ax = plt.subplots(figsize=(8, 5))
sns.lineplot(
    x="explainer",
    y="global_dir_test",
    data=res_50[res_50.explainer!='rf'],
    hue="loss",
    marker="o",
    markersize=10,
    linestyle="dashed",
    ci=30,
    err_style="bars",
)
sns.lineplot(x="explainer",  y="global_dir_test",
    data=res_50[res_50.explainer=='rf'],
    color="black",
    marker="o", markersize=10,)
ax.set_title("Global direction - Test")
ax.set_ylabel("Global direction")
ax.set_xlabel("Feature attribution method")
plt.tight_layout()



# %%

##### Feature attribution - Local direction ######
###############################################################################


# %%

fig, ax = plt.subplots(figsize=(8, 5))
sns.lineplot(
    x="explainer",
    y="local_dir_train",
    data=res_50[res_50.explainer!='rf'],
    hue="loss",
    marker="o",
    markersize=10,
    ci=30,
    err_style="bars",
)
sns.lineplot(x="explainer",  y="local_dir_train",
    data=res_50[res_50.explainer=='rf'],
    color="black",
    marker="o", markersize=10,)
ax.set_title("Local direction - Train")
ax.set_ylabel("Local direction")
ax.set_xlabel("Feature attribution method")
plt.tight_layout()

fig, ax = plt.subplots(figsize=(8, 5))
sns.lineplot(
    x="explainer",
    y="local_dir_test",
    data=res_50[res_50.explainer!='rf'],
    hue="loss",
    marker="o",
    markersize=10,
    linestyle="dashed",
    ci=30,
    err_style="bars",
)
sns.lineplot(x="explainer",  y="local_dir_test",
    data=res_50[res_50.explainer=='rf'],
    color="black",
    marker="o", markersize=10,)
ax.set_title("Local direction - Test")
ax.set_ylabel("Local direction")
ax.set_xlabel("Feature attribution method")
plt.tight_layout()


