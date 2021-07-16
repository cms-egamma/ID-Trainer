#!/usr/bin/env python
# coding: utf-8

# In[1]:


import matplotlib.pyplot as plt


# In[2]:





# In[3]:


def ptetaplot(ptbins,etabins,data,ax,title):
    etabinstext=[]
    ptbinstext=[]
    for i in range(len(ptbins)):
        if i==len(ptbins)-1:
            ptbinstext.append('overflow')
            continue
        ptbinstext.append(str(ptbins[i])+'-'+str(ptbins[i+1]))
    for i in range(len(etabins)):
        if i==len(etabins)-1:
            etabinstext.append('overflow')
            continue
        etabinstext.append(str(etabins[i])+'-'+str(etabins[i+1]))
    import seaborn as sns
    ptbinstext.reverse()
    import pandas as pd
    df = pd.DataFrame(data=data, columns=etabinstext, index=ptbinstext)
    df=df[::-1].reset_index(drop=True)
    sns.heatmap(df, square=False,ax=ax,cmap="Blues",annot=True,cbar=False)
    ax.set_yticklabels(labels=ptbinstext,va='center')
    ax.set_ylabel("$p_T$ bins(GeV)")
    ax.set_xlabel("$\eta$ bins")
    ax.set_title(title)
    
def ptetaRwtTested(Sigdf,Bkgdf,ptbins,etabins,Wt,NWt,ele_pt='ele_pt',scl_eta='scl_eta',od='.'):
    print("Reweighting Now...")
    Sdata=[]
    Bdata=[]
    Wtdata=[]
    Sigdf[NWt]=1
    Bkgdf[NWt]=1
    for i in range(len(ptbins)):
        Bdatai=[]
        Sdatai=[]
        Wtdatai=[]
        for j in range(len(etabins)):
            if i==(len(ptbins)-1) and j<(len(etabins)-1):
                sel=ele_pt+'>@ptbins[@i] & '+scl_eta+'>@etabins[@j] & '+scl_eta+'<@etabins[@j+1]'
                Bsum=Bkgdf.query(sel)[Wt].sum()
                Bdatai.append(Bsum)
                Ssum=Sigdf.query(sel)[Wt].sum()
                Sdatai.append(Ssum)
                #print("BSum "+str(Bsum))
                #print("SSum "+str(Ssum))
                if Bsum>0 and Ssum>0:
                    #print("Entering1")
                    Wtdatai.append(Ssum/Bsum)
                    Bkgdf.loc[sel,NWt]=Ssum/Bsum
                else:
                    #print("Entering2")
                    Wtdatai.append(1)
                    Bkgdf.loc[sel,NWt]=1
                continue 
            if i<(len(ptbins)-1) and j==(len(etabins)-1):
                sel=ele_pt+'>@ptbins[@i] & '+ele_pt+'<=@ptbins[@i+1] & '+scl_eta+'>@etabins[@j]'
                Bsum=Bkgdf.query(sel)[Wt].sum()
                Bdatai.append(Bsum)
                Ssum=Sigdf.query(sel)[Wt].sum()
                Sdatai.append(Ssum)
                #print("BSum "+str(Bsum))
                #print("SSum "+str(Ssum))
                if Bsum>0 and Ssum>0:
                    #print("Entering1")
                    Wtdatai.append(Ssum/Bsum)
                    Bkgdf.loc[sel,NWt]=Ssum/Bsum
                else:
                    #print("Entering2")
                    Wtdatai.append(1)
                    Bkgdf.loc[sel,NWt]=1
                continue 
            if i==(len(ptbins)-1) and j==(len(etabins)-1):
                sel=ele_pt+'>@ptbins[@i] & '+scl_eta+'>@etabins[@j]'
                Bsum=Bkgdf.query(sel)[Wt].sum()
                Bdatai.append(Bsum)
                Ssum=Sigdf.query(sel)[Wt].sum()
                Sdatai.append(Ssum)
                #print("BSum "+str(Bsum))
                #print("SSum "+str(Ssum))
                if Bsum>0 and Ssum>0:
                    #print("Entering1")
                    Wtdatai.append(Ssum/Bsum)
                    Bkgdf.loc[sel,NWt]=Ssum/Bsum
                else:
                    #print("Entering2")
                    Wtdatai.append(1)
                    Bkgdf.loc[sel,NWt]=1
                continue 
            sel=ele_pt+'>@ptbins[@i] & '+ele_pt+'<=@ptbins[@i+1] & scl_eta>@etabins[@j] & '+scl_eta+'<@etabins[@j+1]'
            Bsum=Bkgdf.query(sel)[Wt].sum()
            Bdatai.append(Bsum)
            Ssum=Sigdf.query(sel)[Wt].sum()
            Sdatai.append(Ssum)
            #print("BSum "+str(Bsum))
            #print("SSum "+str(Ssum))
            if Bsum>0 and Ssum>0:
                #print("Entering1")
                Wtdatai.append(Ssum/Bsum)
                Bkgdf.loc[sel,NWt]=Ssum/Bsum
            else:
                #print("Entering2")
                Wtdatai.append(1)
                Bkgdf.loc[sel,NWt]=1
        Bdata.append(Bdatai)
        Sdata.append(Sdatai)
        Wtdata.append(Wtdatai)
    Sigdf[NWt]=Sigdf[Wt]
    Bkgdf[NWt]*=Bkgdf[Wt]
    BdataWtd=[]
    for Wtdatal, Bdatal in zip(Wtdata,Bdata):
        BdataWtd.append([a * b for a, b in zip(Wtdatal, Bdatal)])
    fig, axes = plt.subplots(1,4,figsize=(20,5))
    ptetaplot(ptbins,etabins,Sdata,axes[0],"Signal Bins")
    ptetaplot(ptbins,etabins,Bdata,axes[1],"Background Bins")
    ptetaplot(ptbins,etabins,BdataWtd,axes[2],"Background Bins Reweighted")
    ptetaplot(ptbins,etabins,Wtdata,axes[3],"Background Bins per event weight")
    plt.savefig(od+"/ReweightingPlot.pdf")
    plt.savefig(od+"/ReweightingPlot.pdf")
    return Sigdf[NWt],Bkgdf[NWt]


# In[4]:





# In[5]:


def dataptetastrip(data1):
    data=data1
    for ptlist in data:
        ptlist[-2]=ptlist[-2]+ptlist[-1]
        ptlist.pop(-1)
    data[-2] = [sum(i) for i in zip(data[-2], data[-1])]
    data.pop(-1)
    return data


def df_pteta_rwt(Mdf,
                 label,
                 returnOnlyPosWeights=0, 
                 ptw = [10,30,40,50,200,10000], 
                 etaw = [-1.5,-1.0,1.0,1.5], 
                 eta='', 
                 pt='',
                 SumWeightCol="wt",
                 NewWeightCol="NewWt",target=1,cand=0):
    #Mdf=Ndf.copy()
    ptwt = [1.0]*len(ptw)
    etawt = [1.0]*len(etaw)
    
    for k in range(len(etaw)):
        if k == len(etaw)-1:
            continue
        for i in range(len(ptw)):
            if i == len(ptw)-1:
                continue

            targetSum = Mdf.loc[(Mdf[pt] <ptw[i+1]) & (Mdf[pt] >ptw[i]) & (Mdf[eta] <etaw[k+1]) & (Mdf[eta] >etaw[k]) &(Mdf[label]==target),SumWeightCol].sum()
            candSum = Mdf.loc[(Mdf[pt] <ptw[i+1]) & (Mdf[pt] >ptw[i]) & (Mdf[eta] <etaw[k+1]) & (Mdf[eta] >etaw[k]) &(Mdf[label]==cand),SumWeightCol].sum()

            #print('Number of xsec events in signal for pt '+str(ptw[i])+' to '+str(ptw[i+1])+ 'before  weighing = '+str(targetSum))
            #print('Number of xsec events in background for pt '+str(ptw[i])+' to '+str(ptw[i+1])+ 'before  weighing = '+str(candSum))

            if candSum>0 and targetSum>0:
                ptwt[i]=candSum/(targetSum)
            else:
                ptwt[i]=0
            Mdf.loc[(Mdf[pt] <ptw[i+1]) & (Mdf[pt] >ptw[i]) 
                    & (Mdf[eta] <etaw[k+1]) & (Mdf[eta] >etaw[k]) 
                    &(Mdf[label]==cand),"rwt"] = 1.0
            Mdf.loc[(Mdf[pt] <ptw[i+1]) & (Mdf[pt] >ptw[i]) 
                    & (Mdf[eta] <etaw[k+1]) & (Mdf[eta] >etaw[k]) 
                    &(Mdf[label]==target),"rwt"] = ptwt[i]

            Mdf.loc[:,NewWeightCol] = Mdf.loc[:,"rwt"]*Mdf.loc[:,SumWeightCol]

    MtargetSum = Mdf.loc[Mdf[label]==target,NewWeightCol].sum()
    McandSum = Mdf.loc[Mdf[label]==cand,NewWeightCol].sum()
    print('Number of events in signal after  weighing = '+str(MtargetSum))
    print('Number of events in background after  weighing = '+str(McandSum))

    if returnOnlyPosWeights==0: return 0
    else:
        return ptwt
    
    
    
def df_pteta_rwt(Mdf,
                 label,
                 returnOnlyPosWeights=0,
                 ptw = [10,30,40,50,200,10000],
                 etaw = [-1.5,-1.0,1.0,1.5],
                 eta='',
                 pt='',
                 SumWeightCol="wt",
                 NewWeightCol="NewWt",cand="",
                 Classes=[""]):
    Mdf["rwt"]=1
    Mdf[NewWeightCol]=1
    ptwt = [1.0]*len(ptw)
    etawt = [1.0]*len(etaw)

    for k in range(len(etaw)):
        if k == len(etaw)-1:
            continue
        for i in range(len(ptw)):
            if i == len(ptw)-1:
                continue
            for target in Classes:
                if target != cand:
                    targetSum = Mdf.loc[(Mdf[pt] <ptw[i+1]) & (Mdf[pt] >ptw[i])
                                        & (Mdf[eta] <etaw[k+1]) & (Mdf[eta] >etaw[k])
                                        &(Mdf[label]==target),SumWeightCol].sum()
                    candSum = Mdf.loc[(Mdf[pt] <ptw[i+1]) & (Mdf[pt] >ptw[i])
                                      & (Mdf[eta] <etaw[k+1]) & (Mdf[eta] >etaw[k])
                                      &(Mdf[label]==cand),SumWeightCol].sum()

                    #print('Number of xsec events in signal for pt '+str(ptw[i])+' to '+str(ptw[i+1])+ 'before  weighing = '+str(targetSum))
                    #print('Number of xsec events in background for pt '+str(ptw[i])+' to '+str(ptw[i+1])+ 'before  weighing = '+str(candSum))

                    if candSum>0 and targetSum>0:
                        ptwt[i]=candSum/(targetSum)
                    else:
                        ptwt[i]=0

                    Mdf.loc[(Mdf[pt] <ptw[i+1]) & (Mdf[pt] >ptw[i])
                            & (Mdf[eta] <etaw[k+1]) & (Mdf[eta] >etaw[k])
                            &(Mdf[label]==cand),"rwt"] = 1.0
                    Mdf.loc[(Mdf[pt] <ptw[i+1]) & (Mdf[pt] >ptw[i])
                            & (Mdf[eta] <etaw[k+1]) & (Mdf[eta] >etaw[k])
                            &(Mdf[label]==target),"rwt"] = ptwt[i]

    Mdf.loc[:,NewWeightCol] = Mdf.loc[:,"rwt"]*Mdf.loc[:,SumWeightCol]

    for justclass in Classes:
        Sum = Mdf.loc[Mdf[label]==justclass,NewWeightCol].sum()
        print(f'Number of events in {justclass} after  weighing = '+str(Sum))

    return Mdf[NewWeightCol]
