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
OutputDirName = 'TwoClassOutput' #All plots, models, config file will be stored here

Debug=False # If True, only a small subset of events/objects are used for either Signal or background #Useful for quick debugging

#Branches to read #Should be in the root files #Only the read branches can be later used for any purpose
branches=["ele_*","scl_eta","rho","matchedToGenEle","Fall17isoV2wp80", "Fall17isoV2wp90"]
#branches=["Electron_*"]
#Possible examples
# ["Electron_*","matchingflag",]
# ["Electron_pt", "Electron_deltaEtaSC", "Electron_r9","Electron_eta"]
# You need to read branches to use them anywhere

##### For NanoAOD and other un-flattened trees, you can switch on this option to flatten branches with variable length for each event (Event level -> Object level)
flatten=False
#You can't flatten branches which have different length for the same events. For example: It is not possible to flatten electron and muon branches both at the same time, since each event could have different electrons vs muons. Branches that have only one value for each event, line Nelectrons, can certainly be read along with unflattened branches.

##### If True, this will save the dataframe as a csv and the next time you run the same training with different parameters, it will be much faster
SaveDataFrameCSV=True
##### If branches and files are same a "previous" (not this one) training and SaveDataFrameCSV was True, you can switch on loadfromsaved and it will be much quicker to run the this time
loadfromsaved=False

#Common cuts for all samples (Would generally correspond to the training region)
CommonCut = "(ele_pt>10) & (abs(scl_eta)<0.8)" 
#This is barrel and pt>10 GeV
#endcap would be
#(ele_pt > 10) & (abs(scl_eta)>1.566)

#pt and eta bins of interest -------------------------------------------------------------------
#will be used for robustness studies and will also be used for 2D pt-eta reweighing if the reweighing option is specified 
ptbins = [10,30,40,50,100,5000] 
etabins = [-0.8,-0.5,0.0,0.5,0.8]
ptwtvar='ele_pt'
etawtvar='scl_eta'
#ptbins = [10,30,40,50,100,5000] #ptbins for reweighting
#etabins = [-1.6,-1.0,0.0,1.0,1.6] #etabins for reweighting
#ptwtvar='Electron_pt'
#etawtvar='Electron_eta'
##pt and eta bins of interest -------------------------------------------------------------------

#Reweighting scheme -------------------------------------------------------------------
Reweighing = 'True' # This is independent of xsec reweighing (this reweighing will be done after taking into account xsec weight of multiple samples). Even if this is 'False', xsec reweighting will always be done.
WhichClassToReweightTo="Signal" #2D pt-eta spectrum of all other classs will be reweighted to this class
#will only be used if Reweighing = 'True'
#Reweighting scheme -------------------------------------------------------------------

Classes = ['Background','Signal'] 
ClassColors = ['#377eb8', '#ff7f00', 'red'] # To use in plots
#dictionary of processes
processes=[
    {
        'Class':'Background',
        'path':'samples/QCD.root',#path of root file
        'xsecwt': 1, #xsec wt if any, if none then it can be 1
        'CommonSelection':CommonCut, #Common selection for all classes
        'selection':"(matchedToGenEle == 0)", #selection for this class
    },

    
    {
        'Class':'Background',
        'path':'samples/TTBar.root',#path of root file
        'xsecwt': 1, #xsec wt if any, if none then it can be 1
        'CommonSelection':CommonCut, #Common selection for all classes
        'selection':"(matchedToGenEle == 0)", #selection for this class
    },

    {
        'Class':'Signal',
        'path':'samples/DY.root',#path of root file
        'xsecwt': 1, #xsec wt if any, if none then it can be 1
        'CommonSelection':CommonCut, #Common selection for all classes
        'selection':"(matchedToGenEle == 1)", #selection for this class
    }
]
#####################################################################

testsize=0.2 #(0.2 means 20%) (How much data to use for testing)

def modfiydf(df):#Do not remove this function, even if empty
    #Can be used to add new branches (The pandas dataframe style)
    
    ############ Write you modifications inside this block #######
    #example:
    #df["Electron_SCeta"]=df["Electron_deltaEtaSC"] + df["Electron_eta"]
    
    ####################################################
    
    return 0

Tree = "ntuplizer/tree"

#MVAs to use as a list of dictionaries
MVAs = [
    #can add as many as you like: Assuming you have enough CPU/GPU power to process all 
    #For MVAtypes XGB and DNN are keywords, so names can be XGB_new, DNN_old etc.
    #But keep XGB and DNN in the names (That is how the framework identifies which algo to run
    #Only shown DNN here. XGB examples can be found in other sample configs.
    
    {"MVAtype":"DNN_1",
     "Color":"blue", #Plot color for MVA
     "Label":"DNN masscut", # label can be anything (this is how you will identify them on plot legends)
     "features":["ele_oldsigmaietaieta", "ele_oldsigmaiphiiphi", "ele_oldcircularity", "ele_oldr9", "ele_scletawidth", 
                 "ele_sclphiwidth", "ele_oldhe", "ele_kfhits", "ele_kfchi2", "ele_gsfchi2", "ele_fbrem", "ele_gsfhits", 
                 "ele_expected_inner_hits", "ele_conversionVertexFitProbability", "ele_ep", "ele_eelepout", "ele_IoEmIop", 
                 "ele_deltaetain", "ele_deltaphiin", "ele_deltaetaseed", "rho", "ele_hcalPFClusterIso", "ele_ecalPFClusterIso","ele_dr03TkSumPt"], #Input features #Should be branchs in your dataframe
     "feature_bins":[100, 100, 100, 100,  np.linspace(0,0.05,1000), 100, 100, 100, 100, 100, 100, 100, 100, np.linspace(0.0,0.1,1000), 100, 100, np.linspace(-0.2,0.2,1000), 100, 100, 100, 100, np.linspace(0, 50, 100), np.linspace(0, 30, 100), np.linspace(0, 50, 100)], #same length as features
     #Binning used only for plotting features (should be in the same order as features), does not affect training
     'Scaler':"MinMaxScaler",
     "DNNDict":{'epochs':100, 'batchsize':100, 'lr':0.001, 
                #The other parameters which are not here, can be modified in Trainer script
                'model': Sequential([Dense(48, kernel_initializer='glorot_normal', activation='relu'),
                                     Dense(96, activation="relu"),
                                     Dense(48, activation="relu"),
                                     Dropout(0.1),
                                     Dense(len(Classes),activation="softmax")]),
                'compile':{'loss':'binary_crossentropy','optimizer':Adam(lr=0.001), 'metrics':['accuracy']},
                'earlyStopping': EarlyStopping(monitor='val_loss', mode='min', verbose=1, patience=10)
                #check the modelDNN1 function above, you can also create your own
               }
     },

    
    {"MVAtype":"DNN_2",
     "Color":"blue", #Plot color for MVA
     "Label":"DNN try2", # label can be anything (this is how you will identify them on plot legends)
     "features":["ele_oldsigmaietaieta", "ele_oldsigmaiphiiphi", "ele_oldcircularity", "ele_oldr9", "ele_scletawidth", 
                 "ele_sclphiwidth", "ele_oldhe", "ele_kfhits", "ele_kfchi2", "ele_gsfchi2", "ele_fbrem", "ele_gsfhits"],
     #Input features #Should be branchs in your dataframe
     "feature_bins":[100, 100, 100, 100,  np.linspace(0,0.05,1000), 100, 100, 100, 100, 100, 100, 100],#same length as features
     #Binning used only for plotting features (should be in the same order as features), does not affect training
     'Scaler':"MinMaxScaler",
     "DNNDict":{'epochs':100, 'batchsize':100, 'lr':0.001, 
                #The other parameters which are not here, can be modified in Trainer script
                'model': Sequential([Dense(48, kernel_initializer='glorot_normal', activation='relu'),
                                     Dense(48, activation="relu"),
                                     Dropout(0.1),
                                     Dense(len(Classes),activation="softmax")]),
                'compile':{'loss':'binary_crossentropy','optimizer':Adam(lr=0.001), 'metrics':['accuracy']},
                'earlyStopping': EarlyStopping(monitor='val_loss', mode='min', verbose=1, patience=10)
               }
     },
]
################################

#binning for feature_bins can also be like this
# np.linspace(lower boundary, upper boundary, totalbins+1)
# example: np.linspace(0,20,21) 
# 20 bins from 0 to 20
#when not sure about the binning, you can just specify numbers, which will then correspond to total bins
#You can even specify lists like [10,20,30,100]

################################

#Working Point Flags to compare to (Should be in your ntuple and should also be read in branches)
OverlayWP=['Fall17isoV2wp80', 'Fall17isoV2wp90']
OverlayWPColors = ["black","purple"] #Colors on plots for WPs

#To print thresholds of mva scores for corresponding signal efficiency
SigEffWPs=["80%","90%"] # Example for 80% and 90% Signal Efficiency Working Points
######### 


#####Optional Features (Can be commented)
RandomState=42 # Choose the same number everytime for reproducibility
MVAlogplot=False #If true, MVA outputs are plotted in log scale
Multicore=True #If True all CPU cores available are used XGB 
