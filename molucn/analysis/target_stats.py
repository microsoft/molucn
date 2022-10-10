# %%
import os

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

# matplotlib.use('agg')
import seaborn as sns

from plot_utils import get_stats

print(os.getcwd())
import sys

sys.path.append("/home/t-kenzaamara/molucn/")
matplotlib.rc("text", usetex="True")
par_dir = "/home/t-kenzaamara/molucn/"
pal = sns.color_palette("tab10")
# %%

sns.set_context(
    "notebook",
    rc={
        "legend.fontsize": 18,
        "legend.title_fontsize": 20,
        "axes.titlesize": 24,
        "axes.labelsize": 24,
        "xtick.labelsize": 22,
        "ytick.labelsize": 22,
    },
)
sns.set_style("whitegrid")

# %%
df_stats_350 = get_stats(350).reset_index()
df_stats_350 = df_stats_350.drop_duplicates("target")
attr_res = pd.read_csv(os.path.join(par_dir, f"results/mcs_attr_scores_350.csv"))
model_scores = pd.read_csv(os.path.join(par_dir, f"logs/mcs_model_scores_350.csv"))

attr_res = pd.merge(
    attr_res,
    model_scores,
    on=["target", "conv", "pool", "loss", "lambda1", "explainer", "seed"],
    how="left",
)
attr_res = pd.merge(attr_res, df_stats_350, on="target", how="left")
n_targets = len(np.unique(attr_res["target"]))
attr_res["n_mcs_pairs"] = attr_res["n_mcs_test"] + attr_res["n_mcs_train"]
attr_res.loc[attr_res["explainer"] == "rf", "loss"] = "RF"
attr_res = attr_res.groupby(["target", "loss", "explainer", "mcs"]).mean().reset_index()
attr_res.to_csv(
    os.path.join(par_dir, f"results/mcs_attr_scores_stats_{n_targets}.csv"), index=False
)

#%%
res_50 = attr_res[attr_res.mcs == 50]
res_50.columns
#%%
sns.set_context(
    "notebook",
    rc={
        "legend.fontsize": 18,
        "legend.title_fontsize": 20,
        "axes.titlesize": 24,
        "axes.labelsize": 27,
        "xtick.labelsize": 22,
        "ytick.labelsize": 22,
    },
)
df = attr_res.groupby(["mcs", "target"]).mean().reset_index()
df = df.groupby(["mcs"]).sum().reset_index()
fig, ax = plt.subplots(figsize=(8, 9))
sns.lineplot(
    x="mcs",
    y="n_mcs_pairs",
    data=df,
    marker="o",
    color=pal[0],
    markersize=12,
    lw=3,
    ci=None,
)
ax.set(ylim=(-0.5, 600000))
ylabels = [str(y)[:-2] + "K" for y in ax.get_yticks() / 1000]
ax.set_yticklabels(ylabels)
ax.set_ylabel("Number of pairs", labelpad=10)
ax.set_xlabel("Minimum shared MCS atoms among pairs (\%)", labelpad=10)
plt.tight_layout()
plt.savefig(os.path.join(par_dir, "figures/n_pairs_vs_mcs.pdf"), bbox_inches="tight")

#%%
#
q1 = df_stats_350["n_compounds"].quantile(q=0.25)
q3 = df_stats_350["n_compounds"].quantile(q=0.75)
median = df_stats_350["n_compounds"].quantile(q=0.5)
print(median)
print(q3 - q1)

#%%

q1 = df_stats_350["n_pairs"].quantile(q=0.25)
q3 = df_stats_350["n_pairs"].quantile(q=0.75)
median = df_stats_350["n_pairs"].quantile(q=0.5)
print(median)
print(q3 - q1)
#%%
df = df_stats_350[["n_pairs", "n_compounds"]]
df = df.astype(int)
fig, ax = plt.subplots(figsize=(3, 9))
g = sns.boxplot(data=df, y="n_pairs", linewidth=2.5, width=0.5, color=pal[0], ax=ax)
ax.set(ylabel=None)
ax.set_xlabel("Number of pairs by target", fontsize=27, labelpad=10)
ax.set_yscale("log")
new_yticks = np.array([int(eval(str(y))) for y in ax.get_yticks()])
ylabels = []
for k in range(len(new_yticks)):
    if new_yticks[k] < 1000:
        ylabels.append(str(new_yticks[k]))
    else:
        ylabels.append(str(int(new_yticks[k] / 1000)) + "K")
print(ylabels)
ax.set_yticklabels(ylabels)
plt.tight_layout()
plt.savefig(os.path.join(par_dir, "figures/pairs_boxplot.pdf"), bbox_inches="tight")


#%%
fig, ax = plt.subplots(figsize=(3, 6))
sns.boxplot(data=df, y="n_compounds", linewidth=2.5, width=0.5)
ax.set(ylabel=None)
ax.set_title("Number of compounds\n by target", fontsize=22, pad=20)
# plt.savefig(os.path.join(par_dir, 'figures/compounds_boxplot.pdf'), bbox_inches="tight")

#%%
fig, ax = plt.subplots(figsize=(6, 6))
sns.histplot(x="n_pairs", data=df_stats_350, binwidth=500)
ax.set_xlabel("Pairs by target", labelpad=10)
ax.set(ylabel=None)

ax.set_xlim(0, 32000)
xlabels = [str(x)[:-2] + "K" for x in ax.get_xticks() / 1000]
ax.set_xticklabels(xlabels)

plt.tight_layout()
plt.savefig(
    os.path.join(par_dir, "figures/density_pairs_by_target.pdf"), bbox_inches="tight"
)

#%%
fig, ax = plt.subplots(figsize=(6, 9))
sns.histplot(x="n_compounds", data=df_stats_350, color=pal[0])
ax.set_xlabel("Compounds by target", labelpad=10)
ax.set(ylabel=None)
plt.tight_layout()
plt.savefig(
    os.path.join(par_dir, "figures/density_compounds_by_target.pdf"),
    bbox_inches="tight",
)

#%%


def clean(df):
    df = df[(df.rmse_test < 0.4) & (df.pcc_test > 0.6) & (df.n_pairs < 10000)]
    arr_indices_top_drop = np.random.default_rng().choice(
        df.index, size=int(0.3 * len(df)), replace=False
    )
    df_sample = df.drop(index=arr_indices_top_drop)
    return df_sample


#%%
# create a function
def is_bin(g):
    if g <= q1:
        return 1
    elif (g > q1) & (g <= q2):
        return 2
    elif g > q2:
        return 3
    else:
        return np.nan


def is_rmse(g):
    if g <= rmse_median:
        return "<=RMSE median"
    else:
        return ">RMSE median"


#%%
########## stats on mean pool - Global direction


df = res_50[(res_50.pool == "mean") & (res_50.loss == "MSE")]
df = df.groupby("target").mean().reset_index()
q1 = df["acc_train"].quantile(q=0.333)
q2 = df["acc_train"].quantile(q=0.666)
bin1 = df[df.acc_train <= q1]
bin2 = df[(df.acc_train > q1) & (df.acc_train <= q2)]
bin3 = df[df.acc_train > q2]

rmse_median = df["rmse_test"].quantile(q=0.5)
rmse_median

#%%

# create a new column based on condition
df["bin"] = df["acc_train"].apply(is_bin)
df["rmse_median"] = df["rmse_test"].apply(is_rmse)
df = df.sort_values("bin")
#%%
fig, ax = plt.subplots(1, figsize=(6, 4))
g = sns.histplot(
    data=df,
    x="bin",
    hue="rmse_median",
    multiple="stack",
    palette="tab20",
    bins=3,
    binwidth=0.3,
    legend=False,
    ax=ax,
)
plt.xlabel("Color accuracy")
plt.title("Loss=MSE")
locs, labels = plt.xticks()
plt.xticks([1, 2, 3], ["Q1", "Q2", "Q3"])
plt.tight_layout()
# g.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
plt.show()


#%%
########## stats on mean pool - Global direction

df = res_50[(res_50.pool == "mean") & (res_50.loss == "MSE")]
df = df.groupby("target").mean().reset_index()
q1 = df["global_dir_train"].quantile(q=0.333)
q2 = df["global_dir_train"].quantile(q=0.666)
bin1 = df[df.global_dir_train <= q1]
bin2 = df[(df.global_dir_train > q1) & (df.global_dir_train <= q2)]
bin3 = df[df.global_dir_train > q2]

rmse_median = df["rmse_test"].quantile(q=0.5)
rmse_median

#%%

# create a new column based on condition
df["bin"] = df["global_dir_train"].apply(is_bin)
df["rmse_median"] = df["rmse_test"].apply(is_rmse)
#%%
fig, ax = plt.subplots(1, figsize=(6, 4))
g = sns.histplot(
    data=df,
    x="bin",
    hue="rmse_median",
    multiple="stack",
    palette="tab20",
    bins=3,
    binwidth=0.3,
    legend=False,
    ax=ax,
)

ax.axes.xaxis.set_ticklabels([])
plt.xlabel("global direction")
plt.title("loss=MSE")
plt.xticks([1, 2, 3], ["Q1", "Q2", "Q3"])
plt.show()
plt.tight_layout()


# %%
def is_above_median(g):
    if g <= median:
        return "<=median"
    else:
        return ">median"


# %%
df = res_50[(res_50.pool == "mean") & (res_50.loss == "MSE+UCN")]
# df = df.groupby('target').mean().reset_index()
print(df.columns)
q1 = df["n_compounds"].quantile(q=0.333)
q2 = df["n_compounds"].quantile(q=0.666)
bin1 = df[df.n_compounds <= q1]
bin2 = df[(df.n_compounds > q1) & (df.n_compounds <= q2)]
bin3 = df[df.n_compounds > q2]
median = df["global_dir_test"].quantile(q=0.5)
print(q1)

# create a new column based on condition
df["bin"] = df["n_compounds"].apply(is_bin)
df["global_dir_median"] = df["global_dir_test"].apply(is_above_median)
df = df.sort_values("bin")

fig, ax = plt.subplots(1, figsize=(6, 4))
g = sns.histplot(
    data=df,
    x="bin",
    hue="global_dir_median",
    multiple="stack",
    palette="tab20",
    bins=3,
    binwidth=0.3,
    legend=False,
    ax=ax,
)
plt.xlabel("Number of compounds")
plt.title("Global direction - Loss=MSE+UCN")
locs, labels = plt.xticks()
plt.xticks([1, 2, 3], ["Q1", "Q2", "Q3"])
plt.tight_layout()
# g.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
plt.show()

# %%
# %%
df = res_50[(res_50.pool == "mean") & (res_50.loss == "MSE+UCN")]
# df = df.groupby('target').mean().reset_index()
q1 = df["n_pairs"].quantile(q=0.333)
q2 = df["n_pairs"].quantile(q=0.666)
bin1 = df[df.n_pairs <= q1]
bin2 = df[(df.n_pairs > q1) & (df.n_pairs <= q2)]
bin3 = df[df.n_pairs > q2]
median = df["global_dir_test"].quantile(q=0.5)
print(q1)

# create a new column based on condition
df["bin"] = df["n_pairs"].apply(is_bin)
df["global_dir_median"] = df["global_dir_test"].apply(is_above_median)
df = df.sort_values("bin")

fig, ax = plt.subplots(1, figsize=(6, 4))
g = sns.histplot(
    data=df,
    x="bin",
    hue="global_dir_median",
    multiple="stack",
    palette="tab20",
    bins=3,
    binwidth=0.3,
    legend=False,
    ax=ax,
)
plt.xlabel("Number of pairs")
plt.title("Global direction - Loss=MSE+UCN")
locs, labels = plt.xticks()
plt.xticks([1, 2, 3], ["Q1", "Q2", "Q3"])
plt.tight_layout()
# g.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
plt.show()
# %%


# %%
df = res_50[(res_50.pool == "mean") & (res_50.loss == "MSE+UCN")]
# df = df.groupby('target').mean().reset_index()
print(df.columns)
q1 = df["n_compounds"].quantile(q=0.333)
q2 = df["n_compounds"].quantile(q=0.666)
bin1 = df[df.n_compounds <= q1]
bin2 = df[(df.n_compounds > q1) & (df.n_compounds <= q2)]
bin3 = df[df.n_compounds > q2]
median = df["acc_test"].quantile(q=0.5)

# create a new column based on condition
df["bin"] = df["n_compounds"].apply(is_bin)
df["acc_median"] = df["acc_test"].apply(is_above_median)
df = df.sort_values("bin")

fig, ax = plt.subplots(1, figsize=(6, 4))
g = sns.histplot(
    data=df,
    x="bin",
    hue="acc_median",
    multiple="stack",
    palette="tab20",
    bins=3,
    binwidth=0.3,
    legend=False,
    ax=ax,
)
plt.xlabel("Number of compounds")
plt.title("Color agreement - Loss=MSE+UCN")
locs, labels = plt.xticks()
plt.xticks([1, 2, 3], ["Q1", "Q2", "Q3"])
plt.tight_layout()
# g.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
plt.show()

# %%
df = res_50[(res_50.pool == "mean") & (res_50.loss == "MSE+UCN")]
# df = df.groupby('target').mean().reset_index()
q1 = df["n_pairs"].quantile(q=0.333)
q2 = df["n_pairs"].quantile(q=0.666)
bin1 = df[df.n_pairs <= q1]
bin2 = df[(df.n_pairs > q1) & (df.n_pairs <= q2)]
bin3 = df[df.n_pairs > q2]
median = df["acc_test"].quantile(q=0.5)

# create a new column based on condition
df["bin"] = df["n_pairs"].apply(is_bin)
df["acc_median"] = df["acc_test"].apply(is_above_median)
df = df.sort_values("bin")

fig, ax = plt.subplots(1, figsize=(6, 4))
g = sns.histplot(
    data=df,
    x="bin",
    hue="acc_median",
    multiple="stack",
    palette="tab20",
    bins=3,
    binwidth=0.3,
    legend=False,
    ax=ax,
)
plt.xlabel("Number of pairs")
plt.title("Color agreement - Loss=MSE+UCN")
locs, labels = plt.xticks()
plt.xticks([1, 2, 3], ["Q1", "Q2", "Q3"])
plt.tight_layout()
# g.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
plt.show()
# %%
