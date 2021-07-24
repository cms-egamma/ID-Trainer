# In this file you can specify the training configuration
#####################################################################
######Do not touch this
import numpy as np
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Activation
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.layers import Dropout
from tensorflow.keras.callbacks import EarlyStopping
#####################################################################
####Start here
#####################################################################
OutputDirName = 'SimpleBinaryClassification_XGBoost' #All plots, models, config file will be stored here
Debug=True # If True, only a small subset of events/objects are used for either Signal or background #Useful for quick debugging

#Branches to read #Should be in the root files #Only the read branches can be later used for any purpose
branches=["scl_eta","ele*","matched*","EleMVACats",'passElectronSelection','Fall*']

SaveDataFrameCSV,loadfromsaved=True,False #If loadfromsaved=True, dataframe stored in OutputDirName will be read

Classes,ClassColors = ['IsolatedSignal','NonIsolated'],['#377eb8', '#ff7f00']

processes=[
    {'Class':'IsolatedSignal','path':'./DY.root',
     #Can be a single root file, a list of root file, or even a folder but in a tuple format (folder,fileextension), like ('./samples','.root')
     'xsecwt': 1, #can be a number or a branch name, like 'weight' #Will go into training
     'selection':'(ele_pt > 5) & (abs(scl_eta) < 1.442) & (abs(scl_eta) < 2.5) & (matchedToGenEle==1)', #selection for background
    },
    {'Class':'NonIsolated','path':'./DY.root',
     #Can be a single root file, a list of root file, or even a folder but in a tuple format (folder,fileextension), like ('./samples','.root')
     'xsecwt': 1, #can be a number or a branch name, like 'weight' #Will go into training
     'selection':'(ele_pt > 5) & (abs(scl_eta) < 1.442) & (abs(scl_eta) < 2.5)  & (matchedToGenEle==0)', #selection for background
    },
]

Tree = "ntuplizer/tree"

#MVAs to use as a list of dictionaries
MVAs = [
    #can add as many as you like: For MVAtypes XGB and DNN are keywords, so names can be XGB_new, DNN_old etc. 
    #But keep XGB and DNN in the names (That is how the framework identifies which algo to run
    
    {"MVAtype":"XGB_1", #Keyword to identify MVA method.
     "Color":"green", #Plot color for MVA
     "Label":"XGB try1", # label can be anything (this is how you will identify them on plot legends)
     "features":["ele_fbrem", "ele_deltaetain", "ele_deltaphiin", "ele_oldsigmaietaieta", 
                 "ele_oldhe", "ele_ep", "ele_olde15", "ele_eelepout",
                 "ele_kfchi2", "ele_kfhits", "ele_expected_inner_hits","ele_dr03TkSumPt",
                 "ele_dr03EcalRecHitSumEt","ele_dr03HcalTowerSumEt","ele_gsfchi2","scl_eta","ele_pt",
                 'ele_nbrem','ele_deltaetaseed','ele_hadronicOverEm','ele_olde25max','ele_olde55'],
     "feature_bins":[100 for i in range(22)], #same length as features
     #Binning used only for plotting features (should be in the same order as features), does not affect training
     'Scaler':"MinMaxScaler", #Scaling for features before passing to the model training
     'UseGPU':False, #If you have a GPU card, you can turn on this option (CUDA 10.0, Compute Capability 3.5 required)
     "XGBGridSearch":{'min_child_weight': [5]} #All standard XGB parameters supported
    },
]


#------------------------------------------#------------------------------------------
######################################################################################################
######### Everything below this line is optinal ################################################

#------------------------------------------#Optional parameters below (Can be commented)
'''
#------------------------------------------
OverlayWP=['Fall17isoV2wp90','Fall17isoV2wp80'] # Working Points or flags to comapre to (should be booleans in the trees)
OverlayWPColors = ["black","purple"] #Colors on plots for WPs
#------------------------------------------
##############for 2D pt-eta reweighing
Reweighing = 'True' # This is independent of xsec reweighing (this reweighing will be done after taking into account xsec weight of multiple samples).
##############Even if this is 'False', xsec reweighting will always be carried to the training.
WhichClassToReweightTo="IsolatedSignal" #2D pt-eta spectrum of all other classs will be reweighted to this class
#------------------------------------------
ptbins = [5,10,30,40,50,80,100,5000]
etabins = [-1.6,-1.2,-0.8,-0.5,0.0,0.5,0.8,1.2,1.6]
ptwtvar='ele_pt'
etawtvar='scl_eta'
############# pt and eta bins of interest and branch names to read
############# (will be used for robustness studies and will also be used for 2D pt-eta reweighing) if the reweighing option is True
#------------------------------------------
SigEffWPs=["95%","98%"] # Example for 80% and 90% Signal Efficiency Working Points
############## To print thresholds of mva scores for corresponding signal efficiency
#------------------------------------------
RandomState=42
############### Choose the same number everytime for reproducibility
#------------------------------------------
MVAlogplot=False
############### If true, MVA outputs are plotted in log scale
#------------------------------------------
Multicore=False ### This is not very well tested!! Be careful!!
############### If True all CPU cores available are used XGB
#------------------------------------------
testsize=0.2
############### (0.2 means 20%) (How much data to use for testing)
#------------------------------------------
flatten=False
############## For NanoAOD and other un-flattened trees, you can switch on this option to flatten branches with variable length for each event
############## (Event level -> Object level).
############## You can't flatten branches which have different length for the same events. For example: It is not possible to flatten electron and muon branches both at the same time, since each event could have different electrons vs muons. Branches that have only one value for each event, line Nelectrons, can certainly be read along with unflattened branches.
#------------------------------------------
'''
