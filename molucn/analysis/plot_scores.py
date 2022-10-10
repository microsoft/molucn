# %%
import os
import os.path as osp
import matplotlib
#matplotlib.use('agg')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.style
import matplotlib as mpl
mpl.style.use('classic')
#matplotlib.rc('font', family='sans-serif') 
matplotlib.rc('text', usetex='True')
from collections import Counter
sns.set_theme(style="whitegrid")
matplotlib.rcParams["figure.facecolor"] = "white"
par_dir = '/home/t-kenzaamara/internship2022'

# %%

sns.set_context("notebook", rc={"legend.fontsize":18, "legend.title_fontsize":27, 
                                "axes.titlesize":27,"axes.labelsize":27,
                               "xtick.labelsize" : 26, "ytick.labelsize" : 26})
sns.set_style("whitegrid")

#%%
model_res = pd.read_csv(osp.join(par_dir, f"logs/mcs_model_scores_350.csv"))
attr_res = pd.read_csv(osp.join(par_dir, f"results/mcs_attr_scores_350.csv"))
attr_res =attr_res.sort_values(by=['explainer','pool', 'loss'], ascending=[False, False, True])
attr_res.loc[attr_res['explainer'] =='rf', 'loss'] = 'RF'
attr_res = attr_res[attr_res['explainer'] !='shap']
#%% 
status = dict(Counter(attr_res.target))
attr_res["status"] = attr_res["target"].map(status)
attr_res = attr_res[attr_res.status==max(attr_res['status'])]
print(len(attr_res['target'].unique()))
 #%%
 
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
res_50 = attr_res[attr_res.mcs==50]
df = res_50.groupby(['mcs', 'loss', 'explainer']).mean().reset_index()
df.pivot(index=['explainer'], columns='loss', values='global_dir_test').reset_index()
# %%
res_50 = attr_res[attr_res.mcs==50]
df = res_50.groupby(['mcs', 'loss', 'explainer']).std().reset_index()
df.pivot(index=['explainer'], columns='loss', values='global_dir_test').reset_index()

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
res_50 = attr_res[attr_res.mcs==50]
df= res_50[res_50.explainer!='rf']
df = df[['target', 'explainer', 'loss', 'global_dir_test_%', 'global_dir_train_%']]
df = df.pivot(index=['target', 'explainer'], columns='loss', values='global_dir_test_%').reset_index()

df.loc[df['MSE+UCN'] >= df['MSE'] , 'better'] = 1
df.loc[df['MSE+UCN'] < df['MSE'] , 'better'] = 0
df['delta_ucn'] = np.abs(df['MSE+UCN'] - df['MSE'])/df['MSE'] *100
df['delta_ucn']

#%%
# create a function
def diff(g):
    if g<5:
        return 0.9
    if (g>=5)&(g<10):
        return 2.1
    elif (g>=10)&(g<20):
        return 3.05
    elif g>20:
        return 3.95
    else:
        return np.nan
    
# %%
df['bin'] = df['delta_ucn'].apply(diff)
df = df.sort_values('bin')

print(df.delta_ucn.max())

#%%
########## BARPLOT ############
sns.set_context("notebook", rc={"legend.fontsize":24, "legend.title_fontsize":26, 
                                "axes.titlesize":27,"axes.labelsize":27,
                               "xtick.labelsize" : 18, "ytick.labelsize" : 18})


import matplotlib.patches as mpatches
fig, axs = plt.subplots(2, 3, figsize=(16,8), sharey=True)
gnn_explainers=["diff", "gradinput", "ig", "cam", "gradcam"]
for i, explainer in enumerate(gnn_explainers): 
    df_expl = df[df.explainer==explainer]   
    sns.histplot(data=df_expl, x='bin', hue='better', multiple='dodge', palette=pal[:2], ax=axs[i//3,i%3], binwidth=0.42, legend=False)
    axs[i//3,i%3].set_title(f"{leg_labels[explainer]}", pad=10)
    axs[i//3,i%3].set(xlabel=None, ylabel=None)
    axs[i//3,i%3].set_xticks([1, 2,3.1,4.1],[r'$X\leq 5$', r'$5\leq X<10$', r'$10\leq X<20$', r'$20\leq X$'])
    axs[i//3,i%3].set_xlim(0.8, 4.3)
    
axs[1,2].set_visible(False)
red_patch = mpatches.Patch(color=pal[0], label='MSE', alpha=0.7)
blue_patch = mpatches.Patch(color=pal[1], label='MSE+UCN', alpha=0.7)
fig.legend(handles=[red_patch, blue_patch], title = 'Loss', bbox_to_anchor=(0.85, 0.29), handlelength=1, frameon=True, borderaxespad=0)
fig.text(0.5, -0.04, "\% Global direction improvement", ha='center', fontsize=30)
fig.text(-0.01, 0.27, "Number of targets", ha='center',rotation=90, fontsize=30)
plt.tight_layout(pad=2)
plt.savefig(os.path.join(par_dir, 'figures/gdir_test_by_target_barplot.pdf'), bbox_inches="tight")



########## HEATMAP ############
#%%
bins = np.arange(0, 105, 5)
print(bins)
df['binned_MSE'] = pd.cut(x=df['MSE'], bins=bins)
df['binned_MSE+UCN'] = pd.cut(x=df['MSE+UCN'], bins=bins)
df['ticks_MSE'] = df['binned_MSE'].apply(lambda x: x.right)
df['ticks_MSE+UCN'] = df['binned_MSE+UCN'].apply(lambda x: x.right)


# %% 


sns.set_context("notebook", rc={"legend.fontsize":27, "legend.title_fontsize":27, 
                                "axes.titlesize":35,"axes.labelsize":27,
                               "xtick.labelsize" : 24, "ytick.labelsize" : 24})


fig, axs = plt.subplots(2, 3, figsize=(18,12), sharey=True, sharex=True)
cbar_ax = fig.add_axes([.7, .04, .04, .41])
gnn_explainers=["diff", "gradinput", "ig", "cam", "gradcam"]#, "shap"]
for i, explainer in enumerate(gnn_explainers):
    df_expl = df[df.explainer==explainer]
    df_expl = df_expl[['ticks_MSE', 'ticks_MSE+UCN', 'target']]
    df_expl = df_expl.groupby(['ticks_MSE', 'ticks_MSE+UCN']).count().reset_index()
    df_expl = df_expl.pivot('ticks_MSE+UCN', 'ticks_MSE', 'target')
    #df_expl = df_expl.reindex(df_expl.sort_values(by='ticks_MSE+UCN', ascending=False).index)
    xticklabels = range(0, 105, 5)
    # the index position of the tick labels
    xticks = []
    for label in xticklabels:
        idx_pos = df.index.get_loc(label)
        xticks.append(idx_pos)
    g = sns.heatmap(df_expl, cmap="Blues", ax=axs[i//3,i%3], cbar=i==0, cbar_ax=None if i else cbar_ax, square=True,
                vmin=0)#, vmax=70)
    axs[i//3,i%3].axes.invert_yaxis()
    axs[i//3,i%3].plot([0,100], [0,100], ls="--", c='black', lw=2)
    axs[i//3,i%3].set_title(f"{leg_labels[explainer]}", pad=30)
    axs[i//3,i%3].set(xlabel=None, ylabel=None)
    axs[i//3,i%3].set_xticks(np.linspace(0,20,6))
    axs[i//3,i%3].set_yticks(np.linspace(0,20,6))
    axs[i//3,i%3].set_xticklabels([0, 20, 40, 60, 80, 100], rotation=0)
    axs[i//3,i%3].set_yticklabels([0, 20, 40, 60, 80, 100], rotation=0)
    cbar_ax.set_ylabel('Number of targets', rotation=270, fontsize=27, labelpad=50)
    cbar_ax.tick_params(labelsize=26, pad=10)

axs[1,2].set_visible(False)
fig.text(0.53, -0.05,'Global direction with '+r'$\mathcal{L}_{\mathrm{MSE}}$'+' (\%)', ha='center', fontsize=35)
fig.text(0, 0.17, 'Global direction with '+r'$\mathcal{L}_{\mathrm{MSE+UCN}}$'+' (\%)', ha='center',rotation=90, fontsize=35)
fig.tight_layout()
plt.savefig(os.path.join(par_dir, 'figures/gdir_heatmap.pdf'), bbox_inches="tight")











# %% 
###### MANTA RAY PLOT ##########

sns.set_context("notebook", rc={"legend.fontsize":27, "legend.title_fontsize":27, 
                                "axes.titlesize":35,"axes.labelsize":27,
                               "xtick.labelsize" : 24, "ytick.labelsize" : 24})

fig, axs = plt.subplots(2, 3, figsize=(18,12), sharey=True, sharex=True)

gnn_explainers=["diff", "gradinput", "ig", "cam", "gradcam"]
for i, explainer in enumerate(gnn_explainers):
    df_expl = df[df.explainer==explainer]
    sns.kdeplot(
        x="MSE",
        y="MSE+UCN",
        data=df_expl,
        #color='tab:brown',
        color=dict_color[explainer],
        #marker="o",
        legend=False,
        ax=axs[i//3,i%3],
        fill=True,
        alpha=.6
    )
    axs[i//3,i%3].plot([0,100], [0,100], ls="--", c='black', lw=2)
    axs[i//3,i%3].set_title(f"{leg_labels[explainer]}", pad=10)
    axs[i//3,i%3].set(xlabel=None, ylabel=None)
    axs[i//3,i%3].set_xticks([0,20,40,60,80,100])
    axs[i//3,i%3].set_yticks([0,20,40,60,80,100])
    alpha = round(np.mean(df_expl.better)*100, 1)
    axs[i//3,i%3].text(-10,110,r'$g_{\mathrm{dir}}$'+' higher\nwith UCN: '+r'\textbf{'+'{}'.format(alpha)+r'}\%', fontsize=20, 
                       bbox =dict(facecolor=pal[0], edgecolor='black', linewidth=4, boxstyle='round,pad=0.4',alpha=0.3))
axs[1,2].set_visible(False)
fig.text(0.53, -0.05, 'Global direction with '+r'$\mathcal{L}_{\mathrm{MSE}}$'+' (\%)', ha='center', fontsize=35)
fig.text(-0.02, 0.17, 'Global direction with'+r'$\mathcal{L}_{\mathrm{MSE+UCN}}$'+' (\%)', ha='center',rotation=90, fontsize=35)
plt.ylim(-10,)
plt.tight_layout()
plt.savefig(os.path.join(par_dir, 'figures/gdir_test_by_target_colors.pdf'), bbox_inches="tight")

##### FEATURE ATTRIBUTION ######
###############################################################################

# %%
res_50 = attr_res[attr_res.mcs == 50]
# %%
##### Feature attribution - Color agreement ######
###############################################################################

fig, ax = plt.subplots(figsize=(8, 5))
sns.lineplot(
    x="explainer",
    y="acc_train",
    data=res_50[res_50.explainer != "rf"],
    hue="loss",
    marker="o",
    markersize=10,
    ci=30,
    err_style="bars",
)
sns.lineplot(
    x="explainer",
    y="acc_train",
    data=res_50[res_50.explainer == "rf"],
    color="black",
    marker="o",
    markersize=10,
)
ax.set_title("Accuracy - Color agreement - Train")
ax.set_ylabel("Accuracy")
ax.set_xlabel("Feature attribution method", labelpad=10)
plt.tight_layout()
# fig.savefig(osp.join(par_dir, f"plots/aggr_global_dir.png"))


fig, ax = plt.subplots(figsize=(8, 5))
sns.lineplot(
    x="explainer",
    y="acc_test",
    data=res_50[res_50.explainer != "rf"],
    hue="loss",
    marker="o",
    markersize=10,
    linestyle="dashed",
    ci=30,
    err_style="bars",
)
sns.lineplot(
    x="explainer",
    y="acc_test",
    data=res_50[res_50.explainer == "rf"],
    color="black",
    marker="o",
    markersize=10,
)
ax.set_title("Accuracy - Color agreement - Test")
ax.set_ylabel("Accuracy")
ax.set_xlabel("Feature attribution method", labelpad=10)
plt.tight_layout()

# %%

##### Feature attribution - Global direction ######
###############################################################################


# %%

fig, ax = plt.subplots(figsize=(8, 5))
sns.lineplot(
    x="explainer",
    y="global_dir_train",
    data=res_50[res_50.explainer != "rf"],
    hue="loss",
    marker="o",
    markersize=10,
    ci=30,
    err_style="bars",
)
sns.lineplot(
    x="explainer",
    y="global_dir_train",
    data=res_50[res_50.explainer == "rf"],
    color="black",
    marker="o",
    markersize=10,
)
ax.set_title("Global direction - Train")
ax.set_ylabel("Global direction")
ax.set_xlabel("Feature attribution method", labelpad=10)
plt.tight_layout()
# fig.savefig(osp.join(par_dir, f"plots/aggr_global_dir.png"))

fig, ax = plt.subplots(figsize=(8, 5))
sns.lineplot(
    x="explainer",
    y="global_dir_test",
    data=res_50[res_50.explainer != "rf"],
    hue="loss",
    marker="o",
    markersize=10,
    linestyle="dashed",
    ci=30,
    err_style="bars",
)
sns.lineplot(
    x="explainer",
    y="global_dir_test",
    data=res_50[res_50.explainer == "rf"],
    color="black",
    marker="o",
    markersize=10,
)
ax.set_title("Global direction - Test")
ax.set_ylabel("Global direction")
ax.set_xlabel("Feature attribution method", labelpad=10)
plt.tight_layout()
# fig.savefig(osp.join(par_dir, f"plots/aggr_global_dir.png"))


# %%

##### Feature attribution - Local direction ######
###############################################################################


# %%

fig, ax = plt.subplots(figsize=(8, 5))
sns.lineplot(
    x="explainer",
    y="local_dir_train",
    data=res_50[res_50.explainer != "rf"],
    hue="loss",
    marker="o",
    markersize=10,
    ci=30,
    err_style="bars",
)
sns.lineplot(
    x="explainer",
    y="local_dir_train",
    data=res_50[res_50.explainer == "rf"],
    color="black",
    marker="o",
    markersize=10,
)
ax.set_title("Local direction - Train")
ax.set_ylabel("Local direction")
ax.set_xlabel("Feature attribution method", labelpad=10)
plt.tight_layout()

fig, ax = plt.subplots(figsize=(8, 5))
sns.lineplot(
    x="explainer",
    y="local_dir_test",
    data=res_50[res_50.explainer != "rf"],
    hue="loss",
    marker="o",
    markersize=10,
    linestyle="dashed",
    ci=30,
    err_style="bars",
)
sns.lineplot(
    x="explainer",
    y="local_dir_test",
    data=res_50[res_50.explainer == "rf"],
    color="black",
    marker="o",
    markersize=10,
)
ax.set_title("Local direction - Test")
ax.set_ylabel("Local direction")
ax.set_xlabel("Feature attribution method", labelpad=10)
plt.tight_layout()

# %%
df_ucn_higher = df_expl[df_expl.ucn_better=='MSE+UCN higher MSE']
df_ucn_lower = df_expl[df_expl.ucn_better=='MSE+UCN lower MSE']

n1 = len(df_ucn_higher[df_ucn_higher.delta_ucn>=5])
n2 = len(df_ucn_higher[df_ucn_higher.delta_ucn>=10])
n3 = len(df_ucn_higher[df_ucn_higher.delta_ucn>=20])


n4 = len(df_ucn_lower[df_ucn_lower.delta_ucn>=5])
n5 = len(df_ucn_lower[df_ucn_lower.delta_ucn>=10])
n6 = len(df_ucn_lower[df_ucn_lower.delta_ucn>=20])

df_t = pd.DataFrame({'ucn_better':[1,1,1,0,0,0], 'count':[n1,n2,n3,n4,n5,n6], 'percent':['5\%', '10\%', '20\%','5\%', '10\%', '20\%']})