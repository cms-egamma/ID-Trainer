# In this file you can specify the training configuration

#####################################################################
######Do not touch this
import numpy as np
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Activation
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.layers import Dropout
#####################################################################
####Start here
#####################################################################
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

#####################################################################

testsize=0.2
CommonCut = "(ele_pt > 10)" #Common cuts for both signal and background
Tree = "ntuplizer/tree" #Location/Name of tree inside Root files

features = ["ele_fbrem", "ele_deltaetain", "ele_deltaphiin", "ele_oldsigmaietaieta"] #Input features to MVA

feature_bins = [np.linspace(-1, 1, 51), np.linspace(0, 0.03, 51), np.linspace(0, 0.15, 51), np.linspace(0, 0.03, 51)] #Binning used only for plotting features (should be in the same order as features), does not affect training

#WPs to compare to
OverlayWP=['Fall17isoV1wpLoose','Fall17noIsoV1wpLoose']
OverlayWPColors = ["black","purple"]
#####################################################################

#MVAs to use as a list, e.g : ["XGB","DNN", "Genetic"]
MVAs = ["XGB","DNN"]
MVAColors = ["green","blue"]
######### Grid Search parameters for XGB (will only be used if MVAs contains "XGB"
XGBGridSearch= {'learning_rate':[0.1, 0.01, 0.001]}

######### DNN parameters and model (will only be used if MVAs contains "DNN"
DNNDict={'epochs':5, 'batchsize':100, 'lr':0.001}
modelDNN=Sequential()
modelDNN.add(Dense(2*len(features), kernel_initializer='glorot_normal', activation='relu', input_dim=len(features)))
modelDNN.add(Dense(len(features), kernel_initializer='glorot_normal', activation='relu'))
modelDNN.add(Dropout(0.1))
modelDNN.add(Dense(1, kernel_initializer='glorot_uniform', activation='sigmoid'))

#####################################################################

######### Reweighting scheme
#Reweighting = 'pt-etaSig'
'''
Possible choices :
None : No reweighting
FlatpT : Binned flat in pT (default binning)
Flateta : Binned flat in eta (default binning)
pt-etaSig : To Signal pt-eta spectrum 
pt-etaBkg : To Background pt-eta spectrum
'''
