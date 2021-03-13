#!/usr/bin/env python
# coding: utf-8

# In[1]:


import sys
import os
import uproot
import glob
import pandas as pd
import numpy as np
import ROOT
import matplotlib.pyplot as plt
import json
os.system("")

import numpy as np
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Activation
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.layers import Dropout

def prGreen(prt): print("\033[92m {}\033[00m" .format(prt))


# In[10]:


def in_ipynb():
    try:
        cfg = get_ipython().config 
        print(cfg)
        if 'jupyter' in cfg['IPKernelApp']['connection_file']:
            return True
        else:
            return False
    except NameError:
        return False
    
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
    ax.legend(fontsize=10)
    ax.set_title(title)
    if logscale:
        ax.set_yscale("log", nonposy='clip')


# In[11]:


def plot_roc_curve(df, score_column, tpr_threshold=0, ax=None, color=None, linestyle='-', label=None,cat="Matchlabel",Wt="Wt"):
    from sklearn import metrics
    if ax is None: ax = plt.gca()
    if label is None: label = score_column
    fpr, tpr, thresholds = metrics.roc_curve(df[cat], df[score_column],sample_weight=df[Wt])
    mask = tpr > tpr_threshold
    fpr, tpr = fpr[mask], tpr[mask]
    auc=metrics.auc(fpr, tpr)
    label=label+' auc='+str(round(auc*100,1))+'%'
    ax.plot(tpr, fpr, label=label, color=color, linestyle=linestyle,linewidth=4)
    ax.legend()
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
    ax.legend()


# In[12]:


if in_ipynb(): 
    print("In IPython")
    exec("import Tools.TrainConfig as Conf")
else:
    TrainConfig=sys.argv[1]
    prGreen("Importing settings from "+ TrainConfig.replace("/", "."))
    #exec("from "+TrainConfig+" import *")
    importConfig=TrainConfig.replace("/", ".")
    exec("import "+importConfig+" as Conf")


# In[ ]:


if Conf.Debug==True:
    prGreen("Running in debug mode : Only every 10th event will be used")


# In[ ]:



prGreen("Making output directory")
os.system("mkdir -p " + Conf.OutputDirName)
os.system("cp "+TrainConfig+".py ./"+ Conf.OutputDirName+"/")
os.system("cp Trainer.py ./"+ Conf.OutputDirName+"/")


# In[ ]:


cat='EleType'
weight="NewWt"
label=["Background","Signal"]


# In[ ]:


#Works in uproot3
prGreen("Making data frames")
Sigdf=pd.DataFrame()
Bkgdf=pd.DataFrame()

for SigFile,SigXsecWt,SigCut in zip(Conf.SigFiles,Conf.SigXsecWts,Conf.SigCuts):
    if Conf.Debug==True:
        Sigdfi = uproot.open(SigFile).get(Conf.Tree).pandas.df().query(SigCut+' & '+Conf.CommonCut).iloc[::10]
    else:
        Sigdfi = uproot.open(SigFile).get(Conf.Tree).pandas.df().query(SigCut+' & '+Conf.CommonCut)
    Sigdfi['xsecwt']=SigXsecWt
    Sigdf=pd.concat([Sigdf,Sigdfi],ignore_index=True, sort=False)
for BkgFile,BkgXsecWt,BkgCut in zip(Conf.BkgFiles,Conf.BkgXsecWts,Conf.BkgCuts):
    if Conf.Debug==True:
        Bkgdfi = uproot.open(BkgFile).get(Conf.Tree).pandas.df().query(BkgCut+' & '+Conf.CommonCut).iloc[::10]
    else:
        Bkgdfi = uproot.open(BkgFile).get(Conf.Tree).pandas.df().query(BkgCut+' & '+Conf.CommonCut)
    Bkgdfi['xsecwt']=BkgXsecWt
    Bkgdf=pd.concat([Bkgdf,Bkgdfi],ignore_index=True, sort=False)


# In[ ]:


Sigdf[cat]=1
Bkgdf[cat]=0

Sigdf["Type"]="Signal"
Bkgdf["Type"]="Background"

#Reweighing
Sigdf[weight]=1
Bkgdf[weight]=1

df_final=pd.concat([Sigdf,Bkgdf],ignore_index=True, sort=False)
from sklearn.model_selection import train_test_split
TrainIndices, TestIndices = train_test_split(df_final.index.values.tolist(), test_size=Conf.testsize, random_state=Conf.RandomState, shuffle=True)

df_final.loc[TrainIndices,'Dataset'] = "Train"
df_final.loc[TestIndices,'Dataset'] = "Test"

df_final.loc[TrainIndices,'TrainDataset'] = 1
df_final.loc[TestIndices,'TrainDataset'] = 0

df_final["NewWt"]=1


# In[ ]:





# In[ ]:


import seaborn as sns
fig, axes = plt.subplots(1, 1, figsize=(10, 5))
sns.countplot(x="Type", data=df_final, ax=axes,hue='Dataset',palette=['#432371',"#FAAE7B"])
axes.set_title("Number of samples")
#axes.set_yscale("log")
plt.savefig(Conf.OutputDirName+"/TotalStat_TrainANDTest.png")
    


# In[ ]:


fig, axes = plt.subplots(1, len(Conf.features), figsize=(len(Conf.features)*5, 5))
prGreen("Making training dataset feature plots")
for m in range(len(Conf.features)):
    for i,group_df in df_final[df_final['Dataset'] == "Train"].groupby(cat):
        group_df[Conf.features[m-1]].hist(histtype='step', bins=Conf.feature_bins[m-1], alpha=0.7,label=label[i], ax=axes[m-1], density=False, ls='-', weights =group_df[weight]/group_df[weight].sum(),linewidth=4)
        #df_new = pd.concat([group_df, df_new],ignore_index=True, sort=False)                                                                                            
    axes[m-1].legend(loc='upper right')
    axes[m-1].set_xlabel(Conf.features[m-1])
    axes[m-1].set_yscale("log")
    axes[m-1].set_title(Conf.features[m-1]+" (Training Dataset)")
plt.savefig(Conf.OutputDirName+"/featureplots_Training.png")


# In[ ]:


import matplotlib.pyplot as plt
fig, axes = plt.subplots(1, len(Conf.features), figsize=(len(Conf.features)*5, 5))
prGreen("Making testing dataset feature plots")
for m in range(len(Conf.features)):
    for i,group_df in df_final[df_final['Dataset'] == "Test"].groupby(cat):
        group_df[Conf.features[m-1]].hist(histtype='step', bins=Conf.feature_bins[m-1], alpha=0.7,label=label[i], ax=axes[m-1], density=False, ls='-', weights =group_df[weight]/group_df[weight].sum(),linewidth=4)
        #df_new = pd.concat([group_df, df_new],ignore_index=True, sort=False)                                                                                            
    axes[m-1].legend(loc='upper right')
    axes[m-1].set_xlabel(Conf.features[m-1])
    axes[m-1].set_yscale("log")
    axes[m-1].set_title(Conf.features[m-1]+" (Testing Dataset)")
plt.savefig(Conf.OutputDirName+"/featureplots_Testing.png")


# In[ ]:


X_train = df_final.loc[TrainIndices,Conf.features]
Y_train = df_final.loc[TrainIndices,cat]
Wt_train = df_final.loc[TrainIndices,weight]
    
X_test = df_final.loc[TestIndices,Conf.features]
Y_test = df_final.loc[TestIndices,cat]
Wt_test = df_final.loc[TestIndices,weight]

X_train = np.asarray(X_train)
Y_train = np.asarray(Y_train)
Wt_train = np.asarray(Wt_train)
    
X_test = np.asarray(X_test)
Y_test = np.asarray(Y_test)
Wt_test = np.asarray(Wt_test)


# In[ ]:


if 'XGB' in Conf.MVAs:
    prGreen("XGB Training starting")
    import xgboost as xgb
    from sklearn.model_selection import cross_val_score, GridSearchCV
    xgb_model = xgb.XGBClassifier(objective="binary:logistic", random_state=Conf.RandomState)
    #xgb_model.set_config(verbosity=2)
    prGreen("Performing XGB grid search")
    cv = GridSearchCV(xgb_model, Conf.XGBGridSearch,scoring = 'accuracy',cv=3,verbose=1)
    search=cv.fit(X_train, Y_train, sample_weight=Wt_train,verbose=1)

    prGreen("Expected accuracy of XGB model = "+str((np.round(np.average(search.best_score_),3))*100)+'%')
    #prGreen("Expected accuracy of XGB model = "+str((np.average(search.best_score_))*100)+'%')
    prGreen("XGB Best Parameters")
    
    #json.dumps(search.best_params_)
    prGreen(str(search.best_params_))
    
    df_final.loc[TrainIndices,"XGB_pred"]=cv.predict_proba(X_train)[:,1]
    df_final.loc[TestIndices,"XGB_pred"]=cv.predict_proba(X_test)[:,1]
    
    prGreen("Plotting output response for XGB")
    fig, axes = plt.subplots(1, 1, figsize=(5, 5))
    plot_mva(df_final.query('TrainDataset==1'),"XGB_pred",bins=50,cat=cat,Wt=weight,ax=axes,sample='train',ls='dashed',logscale=True)
    plot_mva(df_final.query('TrainDataset==0'),"XGB_pred",bins=50,cat=cat,Wt=weight,ax=axes,sample='test',ls='dotted',logscale=True)
    plt.savefig(Conf.OutputDirName+"/XGBMVA.png")
    
    prGreen("Plotting ROC for XGB")
    fig, axes = plt.subplots(1, 1, figsize=(5, 5))
    plot_roc_curve(df_final.query('TrainDataset==1'),"XGB_pred", tpr_threshold=0, ax=axes, color=None, linestyle='-', label='Training',cat=cat,Wt=weight)
    plot_roc_curve(df_final.query('TrainDataset==0'),"XGB_pred", tpr_threshold=0, ax=axes, color=None, linestyle='--', label='Testing',cat=cat,Wt=weight)
    if len(Conf.OverlayWP)>0:
        for color,OverlayWpi in zip(Conf.OverlayWPColors,Conf.OverlayWP):
            plot_single_roc_point(df_final.query('TrainDataset==0'), var=OverlayWpi, ax=axes, color=color, marker='o', markersize=6, label=OverlayWpi+" Test dataset", cat=cat,Wt=weight)
    axes.set_ylabel("Background efficiency")
    axes.set_xlabel("Signal efficiency")
    axes.set_title("XGB")
    plt.savefig(Conf.OutputDirName+"/XGBROC.png")


# In[ ]:


from tensorflow.keras.callbacks import EarlyStopping
if 'DNN' in Conf.MVAs:
    prGreen("DNN fitting running")
    es = EarlyStopping(monitor='val_loss', mode='min', verbose=1, patience=10)
    modelDNN=Conf.modelDNN
    modelDNN.compile(loss='binary_crossentropy', optimizer=Adam(lr=Conf.DNNDict['lr']), metrics=['accuracy',])
    train_history = modelDNN.fit(X_train,Y_train,epochs=Conf.DNNDict['epochs'],batch_size=Conf.DNNDict['batchsize'],validation_data=(X_test,Y_test, Wt_test),
                                 verbose=1,callbacks=[es], sample_weight=Wt_train)
    modelDNN.save(Conf.OutputDirName+"/modelDNN.h5")
    df_final.loc[TrainIndices,"DNN_pred"]=modelDNN.predict(X_train)
    df_final.loc[TestIndices,"DNN_pred"]=modelDNN.predict(X_test)
    
    prGreen("Plotting output response for DNN")
    fig, axes = plt.subplots(1, 1, figsize=(5, 5))
    plot_mva(df_final.query('TrainDataset==1'),"DNN_pred",bins=50,cat=cat,Wt=weight,ax=axes,sample='train',ls='dashed',logscale=True)
    plot_mva(df_final.query('TrainDataset==0'),"DNN_pred",bins=50,cat=cat,Wt=weight,ax=axes,sample='test',ls='dotted',logscale=True)
    plt.savefig(Conf.OutputDirName+"/DNNMVA.png")
    
    prGreen("Plotting ROC for DNN")
    fig, axes = plt.subplots(1, 1, figsize=(5, 5))
    plot_roc_curve(df_final.query('TrainDataset==1'),"DNN_pred", tpr_threshold=0, ax=axes, color=None, linestyle='-', label='DNN Training',cat=cat,Wt=weight)
    plot_roc_curve(df_final.query('TrainDataset==0'),"DNN_pred", tpr_threshold=0, ax=axes, color=None, linestyle='--', label='DNN Testing',cat=cat,Wt=weight)
    if len(Conf.OverlayWP)>0:
        for color,OverlayWpi in zip(Conf.OverlayWPColors,Conf.OverlayWP):
            plot_single_roc_point(df_final.query('TrainDataset==0'), var=OverlayWpi, ax=axes, color=color, marker='o', markersize=6, label=OverlayWpi+" Test dataset", cat=cat,Wt=weight)
    axes.set_ylabel("Background efficiency")
    axes.set_xlabel("Signal efficiency")
    axes.set_title("DNN")
    plt.savefig(Conf.OutputDirName+"/DNNROC.png")


# In[ ]:


if 'Genetic' in Conf.MVAs:
    prGreen("Sorry Genetic algo not implemented yet! Coming Soon")


# In[ ]:


##PlotFinalROC
prGreen("Plotting Final ROC")
fig, axes = plt.subplots(1, 1, figsize=(5, 5))
if len(Conf.OverlayWP)>0:
    for color,OverlayWpi in zip(Conf.OverlayWPColors,Conf.OverlayWP):
        plot_single_roc_point(df_final.query('TrainDataset==0'), var=OverlayWpi, ax=axes, color=color, marker='o', markersize=6, label=OverlayWpi+" Test dataset", cat=cat,Wt=weight)
if len(Conf.MVAs)>0:
    for color,MVAi in zip(Conf.MVAColors,Conf.MVAs):
        plot_roc_curve(df_final.query('TrainDataset==0'),MVAi+"_pred", tpr_threshold=0.7, ax=axes, color=color, linestyle='--', label=MVAi+' Testing',cat=cat,Wt=weight)
        plot_roc_curve(df_final.query('TrainDataset==1'),MVAi+"_pred", tpr_threshold=0.7, ax=axes, color=color, linestyle='-', label=MVAi+' Training',cat=cat,Wt=weight)
    axes.set_ylabel("Background efficiency")
    axes.set_xlabel("Signal efficiency")
    axes.set_title("Final")
plt.savefig(Conf.OutputDirName+"/ROCFinal.png")


# In[ ]:


os.system("convert "+Conf.OutputDirName+"/featureplots_T* "+Conf.OutputDirName+"/TotalStat_TrainANDTest.png "+Conf.OutputDirName+"/XGB* "+Conf.OutputDirName+"/DNN* "+Conf.OutputDirName+"/ROCFinal.png "+Conf.OutputDirName+"/mydoc.pdf")
prGreen("Done!! Please find the quick look pdf here "+Conf.OutputDirName+"/mydoc.pdf")


# In[ ]:





# In[ ]:





# In[ ]:




