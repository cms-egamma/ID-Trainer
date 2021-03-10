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


# In[13]:


TrainConfig=str(sys.argv[1])
print("Importing settings from "+ TrainConfig)
exec("from "+TrainConfig+" import *")


# In[3]:


print("Making output directory")
os.system("mkdir -p " + OutputDirName)
os.system("cp TrainConfig.py ./"+ OutputDirName+"/")
os.system("cp Trainer.py ./"+ OutputDirName+"/")


# In[12]:


cat='EleType'
weight="NewWt",""
label=["Background","Signal"]


# In[17]:


#Works in uproot3
print("Making data frames")
Sigdf=pd.DataFrame()
Bkgdf=pd.DataFrame()
from tqdm import tqdm
for SigFile,SigXsecWt in tqdm(zip(SigFiles,SigXsecWts)):
    Sigdfi = uproot.open(SigFile).get(Tree).pandas.df().query(SigCut+' & '+CommonCut)
    Sigdfi['xsecwt']=SigXsecWt
    Sigdf=pd.concat([Sigdf,Sigdfi],ignore_index=True, sort=False)
for BkgFile,BkgXsecWt in tqdm(zip(BkgFiles,BkgXsecWts)):
    Bkgdfi = uproot.open(BkgFile).get(Tree).pandas.df().query(BkgCut+' & '+CommonCut)
    Bkgdf=pd.concat([Bkgdf,Bkgdfi],ignore_index=True, sort=False)


# In[6]:


Sigdf[cat]=1
Bkgdf[cat]=0

#Reweighing
Sigdf[weight]=1
Bkgdf[weight]=1

df_final=pd.concat([Sigdf,Bkgdf],ignore_index=True, sort=False)
from sklearn.model_selection import train_test_split
EB_train, EB_test = train_test_split(df_final, test_size=testsize, random_state=RandomState, shuffle=True)


# In[16]:


import matplotlib.pyplot as plt
fig, axes = plt.subplots(1, len(features), figsize=(len(features)*5, 5))
print("Making feature plots")
for m in tqdm(range(len(features))):
    for i,group_df in EB_train.groupby(cat):
        group_df[features[m-1]].hist(histtype='step', bins=feature_bins[m-1], alpha=0.7,label=label[i], ax=axes[m-1], density=False, ls='-', weights =group_df[weight],linewidth=4)
        #df_new = pd.concat([group_df, df_new],ignore_index=True, sort=False)                                                                                            
    axes[m-1].legend(loc='upper right')
    axes[m-1].set_xlabel(features[m-1])
    axes[m-1].set_yscale("log")
plt.savefig(OutputDirName+"/featureplots.png")


# In[ ]:




