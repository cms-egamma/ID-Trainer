# In this file you can specify the training configuration

import numpy as np

OutputDirName = 'Output' #All plots, models, config file will be stored here

RandomState=42 # Choose the same number everytime for reproducibility

Debug=True # If True, only a 10% of events/objects are used for either Signal or background

#Files, Cuts and XsecWts should have the same number of elements
SigFiles = ['/eos/user/a/akapoor/SWAN_projects/DYJets_incl_MLL-50_TuneCP5_14TeV_NEv_3943691.root',
            '/eos/user/a/akapoor/SWAN_projects/TauGun_Pt-15to500_14TeV_Run3Summer19MiniAOD-2021Scenario_106X_mcRun3_2021_NewCode2021.root']
#Cuts to select appropriate signal
SigCuts= ["(matchedToGenEle == 1)","(matchedToGenEle == 1)"]
#Any extra xsec weight : Useful when stitching samples for either signal or background : '1' means no weight
SigXsecWts=[1,1]

#Files, Cuts and XsecWts should have the same number of elements
BkgFiles = ['/eos/user/a/akapoor/SWAN_projects/QCD_Pt_15to7000_TuneCP5_Flat_14TeV_pythia8-Run3Summer19MiniAOD-106X_mcRun3_2023.root']
#Cuts to select appropriate background
BkgCuts= ["(matchedToGenEle == 0)"]
#Any extra xsec weight : Useful when stitching samples for either signal or background : '1' means no weight
BkgXsecWts=[1]

testsize=0.2

CommonCut = "(ele_pt > 10)" #Common cuts for both signal and background

Tree = "ntuplizer/tree" #Location/Name of tree inside Root files

features = ["ele_fbrem", "ele_deltaetain", "ele_deltaphiin", "ele_oldsigmaietaieta"] #Input features to MVA

feature_bins = [np.linspace(-1, 1, 51), np.linspace(0, 0.03, 51), np.linspace(0, 0.15, 51), np.linspace(0, 0.03, 51)] #Binning used only for plotting features (should be in the same order as features), does not affect training

MVAs = "XGB DNN" #MVAs to use separated by space

DefaultMVAs='True' #If this is true default configuration for MVAs are used : probably best when just starting out

SplitTraining='True' #Whether to split training (for example in barrel and endcap)
SplitBoolean='ele_isEB' #Should be an integer, only used in SplitTraining is true

Reweighting = 'pt-etaSig' #Binning scheme
'''
Possible choices :
None : No reweighting
FlatpT : Binned flat in pT (default binning)
Flateta : Binned flat in eta (default binning)
pt-etaSig : To Signal pt-eta spectrum 
pt-etaBkg : To Background pt-eta spectrum
'''
