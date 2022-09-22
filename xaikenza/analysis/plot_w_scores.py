# %%
import os
import os.path as osp
import matplotlib as mpl
#matplotlib.use('agg')
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import seaborn as sns
from collections import Counter
sns.set_theme(style="whitegrid")

mpl.rcParams["figure.facecolor"] = "white"

par_dir = "/home/t-kenzaamara/molucn"

# Option: plot average or weighted average

# %%

sns.set_context("notebook", rc={"legend.fontsize":18, "legend.title_fontsize":20, 
                                "axes.titlesize":22,"axes.labelsize":22,
                               "xtick.labelsize" : 20, "ytick.labelsize" : 20})
sns.set_style("whitegrid")
# %%
model_res = pd.read_csv(osp.join(par_dir, f"logs/mcs_model_scores_350.csv"))
attr_res = pd.read_csv(osp.join(par_dir, f"results/mcs_attr_scores_350.csv"))
attr_res =attr_res.sort_values(by=['explainer','pool', 'loss'], ascending=[False, False, True])
attr_res['n_mcs_pairs'] = attr_res['n_mcs_test']+attr_res['n_mcs_train']
attr_res.loc[attr_res['explainer'] =='rf', 'loss'] = 'RF'
attr_res = attr_res[attr_res['explainer'] !='shap']
#%% 
status = dict(Counter(attr_res.target))
attr_res["status"] = attr_res["target"].map(status)
attr_res = attr_res[attr_res.status==max(attr_res['status'])]
print(len(attr_res['target'].unique()))
# %%
df = attr_res.groupby(['target', 'mcs']).mean().reset_index()[['target', 'mcs', 'n_mcs_train', 'n_mcs_test']]

df_full = df.groupby('mcs').sum().reset_index()

mcs_stats = pd.merge(df, df_full, on='mcs', how='left', suffixes=('', '_full'))
mcs_stats['w_train'] = mcs_stats['n_mcs_train']/mcs_stats['n_mcs_train_full']
mcs_stats['w_test'] = mcs_stats['n_mcs_test']/mcs_stats['n_mcs_test_full']
mcs_stats
# %%
mcs_stats.groupby('mcs').sum()
#%%
w_attr_res = pd.merge(attr_res,mcs_stats, on=['mcs','target'], how='left', suffixes=('', '_full'))
print(w_attr_res)
for col in ['acc', 'global_dir', 'local_dir']:
    for set in ['train', 'test']:
        score = col+'_'+set
        w_attr_res[score]=w_attr_res.apply(lambda x: x[score]*x['w_'+set], axis = 1)

# %% 
w_attr_res = w_attr_res.groupby(['mcs', 'loss', 'explainer']).sum().reset_index()
w_attr_res

# %% 
pal = sns.color_palette("tab10")
dict_color = {"gradinput":pal[0], "ig":pal[1], "cam":pal[2], "gradcam": pal[3], "diff": pal[4], "shap": pal[5], "rf":"black"}
leg_labels = {"gradinput":"GradInput", "ig":"IntegratedGrads", "cam":"CAM", "gradcam": "Grad-CAM", "diff": "Masking (GNN)", "shap": "SHAP", "rf":"Masking (RF)"}
order_items = {"rf":0, "diff": 1, "gradinput":2, "ig":3, "cam":4, "gradcam":5}#, "shap":6}
w_attr_res['order'] = w_attr_res['explainer'].apply(lambda x: order_items[x])
w_attr_res = w_attr_res.sort_values(by='order')
w_attr_res['acc_test_%'] = w_attr_res['acc_test'].apply(lambda x: x*100)
w_attr_res['acc_train_%'] = w_attr_res['acc_train'].apply(lambda x: x*100)
w_attr_res['global_dir_test_%'] = w_attr_res['global_dir_test'].apply(lambda x: x*100)
w_attr_res['global_dir_train_%'] = w_attr_res['global_dir_train'].apply(lambda x: x*100)


# %%
##### Color agreement ######
###############################################################################

# %% 
# Test pairs
fig, axs = plt.subplots(1, 3, figsize=(18,6), sharey=True, sharex=True)

sns.lineplot(
    x="mcs",
    y="acc_test_%",
    data=w_attr_res[(w_attr_res.loss=='MSE')|(w_attr_res.loss=='RF')],
    hue="explainer",
    marker="o",
    markersize=7,
    lw=2.5,
    markeredgecolor='none',
    ci=None,
    legend=False,
    palette=dict_color,
    ax=axs[0]
)
axs[0].set_title("Loss = MSE", pad=10)
axs[0].set_ylabel("Weighted color agreement (%)", labelpad=10)
axs[0].set(xlabel=None)

sns.lineplot(
    x="mcs",
    y="acc_test_%",
    data=w_attr_res[(w_attr_res.loss=='MSE+AC')|(w_attr_res.loss=='RF')],
    hue="explainer",
    marker="o",
    markersize=7,
    lw=2.5,
    markeredgecolor='none',
    ci=None,
    legend=False,
    palette=dict_color,
    ax=axs[1]
)
axs[1].set_title("Loss = MSE + AC", pad=10)
axs[1].set(xlabel=None)

g = sns.lineplot(
    x="mcs",
    y="acc_test_%",
    data=w_attr_res[(w_attr_res.loss=='MSE+UCN')|(w_attr_res.loss=='RF')],
    hue="explainer",
    marker="o",
    markersize=7,
    lw=2.5,
    markeredgecolor='none',
    ci=None,
    palette=dict_color,
    ax=axs[2]
)
axs[2].set_title("Loss = MSE + UCN", pad=10)
axs[2].set(xlabel=None)
#axs[2].set_xlabel("Minimum shared MCS atoms among pairs (%)")
legend = plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left', handlelength=1, frameon=False, borderaxespad=0, title='Feature Attribution')
for t in legend.get_texts():
    t.set_text(leg_labels[t.get_text()])
for i in range(len(order_items.keys())):
    legend.get_lines()[i].set_linewidth(6)

fig.text(0.45, -0.04, "Minimum shared MCS atoms among pairs (%)", ha='center', fontsize=22)
plt.tight_layout()
plt.savefig(os.path.join(par_dir, 'figures/weighted_acc_test.pdf'), bbox_inches="tight")



# %% 
# 
# Train pairs
fig, axs = plt.subplots(1, 3, figsize=(18,6), sharey=True, sharex=True)

sns.lineplot(
    x="mcs",
    y="acc_train_%",
    data=w_attr_res[(w_attr_res.loss=='MSE')|(w_attr_res.loss=='RF')],
    hue="explainer",
    marker="o",
    markersize=7,
    lw=2.5,
    markeredgecolor='none',
    ci=None,
    legend=False,
    palette=dict_color,
    ax=axs[0]
)
axs[0].set_title("Loss = MSE", pad=10)
axs[0].set_ylabel("Weighted color agreement (%)", labelpad=10)
axs[0].set(xlabel=None)

sns.lineplot(
    x="mcs",
    y="acc_train_%",
    data=w_attr_res[(w_attr_res.loss=='MSE+AC')|(w_attr_res.loss=='RF')],
    hue="explainer",
    marker="o",
    markersize=7,
    lw=2.5,
    markeredgecolor='none',
    ci=None,
    legend=False,
    palette=dict_color,
    ax=axs[1]
)
axs[1].set_title("Loss = MSE + AC", pad=10)
axs[1].set(xlabel=None)

g = sns.lineplot(
    x="mcs",
    y="acc_train_%",
    data=w_attr_res[(w_attr_res.loss=='MSE+UCN')|(w_attr_res.loss=='RF')],
    hue="explainer",
    marker="o",
    markersize=7,
    lw=2.5,
    markeredgecolor='none',
    ci=None,
    palette=dict_color,
    ax=axs[2]
)
axs[2].set_title("Loss = MSE + UCN", pad=10)
axs[2].set(xlabel=None)

legend = plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left', handlelength=1, frameon=False, borderaxespad=0, title='Feature Attribution')
for t in legend.get_texts():
    t.set_text(leg_labels[t.get_text()])
for i in range(len(order_items.keys())):
    legend.get_lines()[i].set_linewidth(6)

fig.text(0.45, -0.04, "Minimum shared MCS atoms among pairs (%)", ha='center', fontsize=22)

plt.tight_layout()
plt.savefig(os.path.join(par_dir, 'figures/weighted_acc_train.pdf'), bbox_inches="tight")



# %% 
############ Global Direction ##############
###############################################################################
# %% 
# Test pairs
fig, axs = plt.subplots(1, 3, figsize=(18,6), sharey=True, sharex=True)

sns.lineplot(
    x="mcs",
    y="global_dir_test_%",
    data=w_attr_res[(w_attr_res.loss=='MSE')|(w_attr_res.loss=='RF')],
    hue="explainer",
    marker="o",
    markersize=7,
    lw=2.5,
    markeredgecolor='none',
    ci=None,
    legend=False,
    palette=dict_color,
    ax=axs[0]
)
axs[0].set_title("Loss = MSE", pad=10)
axs[0].set_ylabel("Weighted global direction (%)", labelpad=10)
#axs[0].set_xlabel("Minimum shared MCS atoms among pairs (%)")
axs[0].set(xlabel=None)

sns.lineplot(
    x="mcs",
    y="global_dir_test_%",
    data=w_attr_res[(w_attr_res.loss=='MSE+AC')|(w_attr_res.loss=='RF')],
    hue="explainer",
    marker="o",
    markersize=7,
    lw=2.5,
    markeredgecolor='none',
    ci=None,
    legend=False,
    palette=dict_color,
    ax=axs[1]
)
axs[1].set_title("Loss = MSE + AC", pad=10)
axs[1].set(xlabel=None)
#axs[1].set_xlabel("Minimum shared MCS atoms among pairs (%)")

g = sns.lineplot(
    x="mcs",
    y="global_dir_test_%",
    data=w_attr_res[(w_attr_res.loss=='MSE+UCN')|(w_attr_res.loss=='RF')],
    hue="explainer",
    marker="o",
    markersize=7,
    lw=2.5,
    markeredgecolor='none',
    ci=None,
    palette=dict_color,
    ax=axs[2]
)
axs[2].set_title("Loss = MSE + UCN", pad=10)
axs[2].set(xlabel=None)
#axs[2].set_xlabel("Minimum shared MCS atoms among pairs (%)")
legend = plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left', handlelength=1, frameon=False, borderaxespad=0, title='Feature Attribution')
for t in legend.get_texts():
    t.set_text(leg_labels[t.get_text()])
for i in range(len(order_items.keys())):
    legend.get_lines()[i].set_linewidth(6)

#plt.xlabel("Minimum shared MCS atoms among pairs (%)")
fig.text(0.45, -0.04, "Minimum shared MCS atoms among pairs (%)", ha='center', fontsize=22)
#fig.suptitle("Global direction vs MCS - Test", fontsize=32)
plt.tight_layout()
plt.savefig(os.path.join(par_dir, 'figures/weighted_gdir_test.pdf'), bbox_inches="tight")

########## Global direction #########
# %% 
# 
# Train pairs
fig, axs = plt.subplots(1, 3, figsize=(18,6), sharey=True, sharex=True)

sns.lineplot(
    x="mcs",
    y="global_dir_train_%",
    data=w_attr_res[(w_attr_res.loss=='MSE')|(w_attr_res.loss=='RF')],
    hue="explainer",
    marker="o",
    markersize=7,
    lw=2.5,
    markeredgecolor='none',
    ci=None,
    legend=False,
    palette=dict_color,
    ax=axs[0]
)
axs[0].set_title("Loss = MSE", pad=10)
axs[0].set_ylabel("Weighted global direction (%)", labelpad=10)
axs[0].set(xlabel=None)

sns.lineplot(
    x="mcs",
    y="global_dir_train_%",
    data=w_attr_res[(w_attr_res.loss=='MSE+AC')|(w_attr_res.loss=='RF')],
    hue="explainer",
    marker="o",
    markersize=7,
    lw=2.5,
    markeredgecolor='none',
    ci=None,
    legend=False,
    palette=dict_color,
    ax=axs[1]
)
axs[1].set_title("Loss = MSE + AC", pad=10)
axs[1].set(xlabel=None)

g = sns.lineplot(
    x="mcs",
    y="global_dir_train_%",
    data=w_attr_res[(w_attr_res.loss=='MSE+UCN')|(w_attr_res.loss=='RF')],
    hue="explainer",
    marker="o",
    markersize=7,
    lw=2.5,
    markeredgecolor='none',
    ci=None,
    palette=dict_color,
    ax=axs[2]
)
axs[2].set_title("Loss = MSE + UCN", pad=10)
axs[2].set(xlabel=None)

legend = plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left', handlelength=1, frameon=False, borderaxespad=0, title='Feature Attribution')
for t in legend.get_texts():
    t.set_text(leg_labels[t.get_text()])
for i in range(len(order_items.keys())):
    legend.get_lines()[i].set_linewidth(6)

fig.text(0.45, -0.04, "Minimum shared MCS atoms among pairs (%)", ha='center', fontsize=22)

plt.tight_layout()
plt.savefig(os.path.join(par_dir, 'figures/weighted_gdir_train.pdf'), bbox_inches="tight")
# %%

























# %%
res_50 = w_attr_res[w_attr_res.mcs==50]
res_50 = res_50.groupby(['loss', 'explainer']).sum().reset_index()
##### FEATURE ATTRIBUTION ######
###############################################################################

# %%
res_50 
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
ax.set_ylabel("Weighted color accuracy")
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
ax.set_ylabel("Weighted color accuracy")
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
ax.set_ylabel("Weighted global direction")
ax.set_xlabel("Feature attribution method")
plt.tight_layout()

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
ax.set_ylabel("Weighted global direction")
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


