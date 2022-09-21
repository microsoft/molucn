# %%
import os
import os.path as osp
import matplotlib
matplotlib.use('agg')
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import seaborn as sns
import scipy.stats as stats
import shutil

sns.set_theme(style="whitegrid")

matplotlib.rcParams["figure.facecolor"] = "white"

par_dir = "/home/t-kenzaamara/internship2022"


# Option: plot scores vs mcs


# %%

sns.set_context("notebook", rc={"legend.fontsize":18, "legend.title_fontsize":20, 
                                "axes.titlesize":22,"axes.labelsize":22,
                               "xtick.labelsize" : 20, "ytick.labelsize" : 20})
sns.set_style("whitegrid")

#%%
model_res = pd.read_csv(osp.join(par_dir, f"logs/mcs_model_scores_350.csv"))
attr_res = pd.read_csv(osp.join(par_dir, f"results/mcs_attr_scores_350.csv"))
attr_res =attr_res.sort_values(by=['explainer','pool', 'loss'], ascending=[False, False, True])
attr_res.loc[attr_res['explainer'] =='rf', 'loss'] = 'RF'


# %% 
pal = sns.color_palette("tab10")
dict_color = {"gradinput":pal[0], "ig":pal[1], "cam":pal[2], "gradcam": pal[3], "diff": pal[4], "shap": pal[5], "rf":"black"}
leg_labels = {"gradinput":"GradInput", "ig":"IntegratedGrads", "cam":"CAM", "gradcam": "Grad-CAM", "diff": "Masking (GNN)", "shap": "SHAP", "rf":"Masking (RF)"}
order_items = {"rf":0, "diff": 1, "gradinput":2, "ig":3, "cam":4, "gradcam":5, "shap":6}
attr_res['order'] = attr_res['explainer'].apply(lambda x: order_items[x])
attr_res = attr_res.sort_values(by='order')
attr_res['acc_test_%'] = attr_res['acc_test'].apply(lambda x: x*100)
attr_res['acc_train_%'] = attr_res['acc_train'].apply(lambda x: x*100)
attr_res['global_dir_test_%'] = attr_res['global_dir_test'].apply(lambda x: x*100)
attr_res['global_dir_train_%'] = attr_res['global_dir_train'].apply(lambda x: x*100)


# %%
##### Color agreement ######
###############################################################################

# %% 
# Test pairs
fig, axs = plt.subplots(1, 3, figsize=(18,6), sharey=True, sharex=True)

sns.lineplot(
    x="mcs",
    y="acc_test_%",
    data=attr_res[(attr_res.loss=='MSE')|(attr_res.loss=='RF')],
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
axs[0].set_ylabel("Color agreement (%)", labelpad=10)
axs[0].set(xlabel=None)

sns.lineplot(
    x="mcs",
    y="acc_test_%",
    data=attr_res[(attr_res.loss=='MSE+AC')|(attr_res.loss=='RF')],
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
    data=attr_res[(attr_res.loss=='MSE+UCN')|(attr_res.loss=='RF')],
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
for i in range(7):
    legend.get_lines()[i].set_linewidth(6)

fig.text(0.45, -0.04, "Minimum shared MCS atoms among pairs (%)", ha='center', fontsize=22)
plt.tight_layout()
plt.savefig(os.path.join(par_dir, 'figures/acc_test.pdf'), bbox_inches="tight")


# %% 
# 
# Train pairs
fig, axs = plt.subplots(1, 3, figsize=(18,6), sharey=True, sharex=True)

sns.lineplot(
    x="mcs",
    y="acc_train_%",
    data=attr_res[(attr_res.loss=='MSE')|(attr_res.loss=='RF')],
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
axs[0].set_ylabel("Color agreement (%)", labelpad=10)
axs[0].set(xlabel=None)

sns.lineplot(
    x="mcs",
    y="acc_train_%",
    data=attr_res[(attr_res.loss=='MSE+AC')|(attr_res.loss=='RF')],
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
    data=attr_res[(attr_res.loss=='MSE+UCN')|(attr_res.loss=='RF')],
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
for i in range(7):
    legend.get_lines()[i].set_linewidth(6)

fig.text(0.45, -0.04, "Minimum shared MCS atoms among pairs (%)", ha='center', fontsize=22)

plt.tight_layout()
plt.savefig(os.path.join(par_dir, 'figures/acc_train.pdf'), bbox_inches="tight")



# %% 
############ Global Direction ##############
###############################################################################
# %% 
# Test pairs
fig, axs = plt.subplots(1, 3, figsize=(18,6), sharey=True, sharex=True)

sns.lineplot(
    x="mcs",
    y="global_dir_test_%",
    data=attr_res[(attr_res.loss=='MSE')|(attr_res.loss=='RF')],
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
axs[0].set_ylabel("Global direction (%)", labelpad=10)
#axs[0].set_xlabel("Minimum shared MCS atoms among pairs (%)")
axs[0].set(xlabel=None)

sns.lineplot(
    x="mcs",
    y="global_dir_test_%",
    data=attr_res[(attr_res.loss=='MSE+AC')|(attr_res.loss=='RF')],
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
    data=attr_res[(attr_res.loss=='MSE+UCN')|(attr_res.loss=='RF')],
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
for i in range(7):
    legend.get_lines()[i].set_linewidth(6)

#plt.xlabel("Minimum shared MCS atoms among pairs (%)")
fig.text(0.45, -0.04, "Minimum shared MCS atoms among pairs (%)", ha='center', fontsize=22)
#fig.suptitle("Global direction vs MCS - Test", fontsize=32)
plt.tight_layout()
plt.savefig(os.path.join(par_dir, 'figures/gdir_test.pdf'), bbox_inches="tight")

########## Global direction #########
# %% 
# 
# Train pairs
fig, axs = plt.subplots(1, 3, figsize=(18,6), sharey=True, sharex=True)

sns.lineplot(
    x="mcs",
    y="global_dir_train_%",
    data=attr_res[(attr_res.loss=='MSE')|(attr_res.loss=='RF')],
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
axs[0].set_ylabel("Global direction (%)", labelpad=10)
axs[0].set(xlabel=None)

sns.lineplot(
    x="mcs",
    y="global_dir_train_%",
    data=attr_res[(attr_res.loss=='MSE+AC')|(attr_res.loss=='RF')],
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
    data=attr_res[(attr_res.loss=='MSE+UCN')|(attr_res.loss=='RF')],
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
for i in range(7):
    legend.get_lines()[i].set_linewidth(6)

fig.text(0.45, -0.04, "Minimum shared MCS atoms among pairs (%)", ha='center', fontsize=22)

plt.tight_layout()
plt.savefig(os.path.join(par_dir, 'figures/gdir_train.pdf'), bbox_inches="tight")
# %%
