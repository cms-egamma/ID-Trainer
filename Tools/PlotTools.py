#Plotting Tools

import matplotlib.pyplot as plt

def prGreen(prt): print("\033[92m {}\033[00m" .format(prt))
    
def plot_mva(df, column, bins, logscale=False, ax=None, title=None, ls='dashed', alpha=0.5, sample='',cat="Matchlabel",Wt="Wt"):
    histtype="bar" 
    if sample is 'test':
        histtype="step"      
    if ax is None:
        ax = plt.gca()
    for name, group in df.groupby(cat):
        if name == 0:
            label="background"
        else:
            label="signal"
        group[column].hist(bins=bins, histtype=histtype, alpha=1,
                           label=label+' '+sample, ax=ax, density=True, ls=ls, weights =group[Wt],linewidth=2)
    #ax.set_ylabel("density")
    ax.set_xlabel(column)
    ax.set_title(title)
    if logscale:
        ax.set_yscale("log", nonposy='clip')
    ax.legend(loc='best')

def plot_roc_curve(df, score_column, tpr_threshold=0, ax=None, color=None, linestyle='-', label=None,cat="Matchlabel",Wt="Wt"):
    from sklearn import metrics
    if ax is None: ax = plt.gca()
    if label is None: label = score_column
    fpr, tpr, thresholds = metrics.roc_curve(df[cat], df[score_column],sample_weight=df[Wt])
    mask = tpr > tpr_threshold
    fpr, tpr = fpr[mask], tpr[mask]
    auc=metrics.auc(fpr, tpr)
    label=label+' auc='+str(round(auc*100,1))+'%'
    ax.plot(tpr, fpr, label=label, color=color, linestyle=linestyle,linewidth=1,alpha=0.7)
    ax.set_yscale("log")
    ax.legend(loc='best')
    return auc

def plot_single_roc_point(df, var='Fall17isoV1wpLoose', 
                          ax=None , marker='o', 
                          markersize=6, color="red", label='', cat="Matchlabel",Wt="Wt"):
    backgroundpass=df.loc[(df[var] == 1) & (df[cat] == 0),Wt].sum()
    backgroundrej=df.loc[(df[var] == 0) & (df[cat] == 0),Wt].sum()
    signalpass=df.loc[(df[var] == 1) & (df[cat] == 1),Wt].sum()
    signalrej=df.loc[(df[var] == 0) & (df[cat] == 1),Wt].sum()
    backgroundrej=backgroundrej/(backgroundpass+backgroundrej)
    signaleff=signalpass/(signalpass+signalrej)
    ax.plot([signaleff], [1-backgroundrej], marker=marker, color=color, markersize=markersize, label=label)
    ax.set_yscale("log")
    ax.legend(loc='best')
    
def pngtopdf(ListPattern=[],Save="mydoc.pdf"):
    import glob, PIL.Image
    L=[]
    for List in ListPattern:
        L+= [PIL.Image.open(f) for f in glob.glob(List)]
    for i,Li in enumerate(L):
        rgb = PIL.Image.new('RGB', Li.size, (255, 255, 255))
        rgb.paste(Li, mask=Li.split()[3])
        L[i]=rgb
    L[0].save(Save, "PDF" ,resolution=100.0, save_all=True, append_images=L[1:])

def MakeFeaturePlots(df_final,features,feature_bins,Set="Train",MVA="XGB_1",OutputDirName='Output',cat="EleType",label=["Background","Signal"],weight="NewWt"):
    fig, axes = plt.subplots(1, len(features), figsize=(len(features)*5, 5))
    prGreen("Making"+Set+" dataset feature plots")
    for m in range(len(features)):
        for i,group_df in df_final[df_final['Dataset'] == Set].groupby(cat):
            group_df[features[m-1]].hist(histtype='step', bins=feature_bins[m-1], alpha=0.7,label=label[i], ax=axes[m-1], density=False, ls='-', weights =group_df[weight]/group_df[weight].sum(),linewidth=4)
            #df_new = pd.concat([group_df, df_new],ignore_index=True, sort=False)                                                                                            
        axes[m-1].legend(loc='upper right')
        axes[m-1].set_xlabel(features[m-1])
        axes[m-1].set_yscale("log")
        axes[m-1].set_title(features[m-1]+" ("+Set+" Dataset)")
    plt.savefig(OutputDirName+"/"+MVA+"/"+MVA+"_"+"featureplots_"+Set+".png")
