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
Debug=True # If True, only a 10% of events/objects are used for either Signal or background

#Files, Cuts and XsecWts should have the same number of elements
SigFiles = [
    #File 1
    '/eos/user/a/akapoor/SWAN_projects/DYJets_incl_MLL-50_TuneCP5_14TeV_NEv_3943691.root',
    #File 2
    '/eos/user/a/akapoor/SWAN_projects/TauGun_Pt-15to500_14TeV_Run3Summer19MiniAOD-2021Scenario_106X_mcRun3_2021_NewCode2021.root'] #Add as many files you like

#Cuts to select appropriate signal
SigCuts= [
    #For File 1
    "(matchedToGenEle == 1)",
    #For File 2
    "(matchedToGenEle == 1)"] #Cuts same as number of files (Kept like this because it maybe different for different files)

#Any extra xsec weight : Useful when stitching samples for either signal or background : '1' means no weight
SigXsecWts=[
    #For File 1
    1,
    #For File 2
    1] #Weights same as number of files (Kept like this because it maybe different for different files)

#Files, Cuts and XsecWts should have the same number of elements
BkgFiles = [
    #File 1
    '/eos/user/a/akapoor/SWAN_projects/QCD_Pt_15to7000_TuneCP5_Flat_14TeV_pythia8-Run3Summer19MiniAOD-106X_mcRun3_2023.root'] #Add as many files you like 

#Cuts to select appropriate background
BkgCuts= [
    #For File 1 
    "(matchedToGenEle == 0)"]#Cuts same as number of files (Kept like this because it maybe different for different files)

#Any extra xsec weight : Useful when stitching samples for either signal or background : '1' means no weight
BkgXsecWts=[
    #For File 1 
    1] #Weights same as number of files (Kept like this because it maybe different for different files) 

#####################################################################

testsize=0.2 #(0.2 means 20%)

#Common cuts for both signal and background (Would generally correspond to the training region)
CommonCut = "(ele_pt > 10) & (abs(scl_eta)>1.566)" 
#This is endcap and pt>10 GeV
#barrel would be
#(ele_pt > 10) & (abs(scl_eta)<1.442)

Tree = "ntuplizer/tree" #Location/Name of tree inside Root files

################################

#MVAs to use as a list, e.g : ["XGB","DNN", "Genetic"]
MVAs = ["XGB_1","XGB_2","DNN_1"] 
#XGB and DNN are keywords so names can be XGB_new, DNN_old etc. But keep XGB and DNN in the names (That is how the framework identifies which algo to run

MVAColors = ["green","blue","red"] #Plot colors for MVAs

MVALabels = {"XGB_1" : "XGB masscut",
             "XGB_2" : "XGB low",
             "DNN_1" : "DNN HHWW"
            } #These labels can be anything (this is how you will identify them on plot legends)

################################
features = {
            "XGB_1":["ele_fbrem", "ele_deltaetain", "ele_deltaphiin", "ele_oldsigmaietaieta"],
            "XGB_2":["ele_deltaetain", "ele_deltaphiin", "ele_oldsigmaietaieta"],
            "DNN_1":["ele_fbrem", "ele_deltaetain", "ele_deltaphiin", "ele_oldsigmaietaieta"],
           } #Input features to MVA #Should be in your ntuples

feature_bins = {
                "XGB_1":[np.linspace(-1, 1, 51), np.linspace(0, 0.03, 51), np.linspace(0, 0.15, 51), 30],
                "XGB_2":[np.linspace(0, 0.03, 51), np.linspace(0, 0.15, 51), 30],
                "DNN_1":[np.linspace(-1, 1, 51), np.linspace(0, 0.03, 51), np.linspace(0, 0.15, 51), 30],
               } #Binning used only for plotting features (should be in the same order as features), does not affect training
#template 
#np.linspace(lower boundary, upper boundary, totalbins+1)

#when not sure about the binning, you can just specify numbers, which will then correspond to total bins
#You can even specify lists like [10,20,30,100]

#EGamma WPs to compare to (Should be in your ntuple)
OverlayWP=['Fall17isoV1wpLoose','Fall17noIsoV1wpLoose']
OverlayWPColors = ["black","purple"] #Colors on plots for WPs
#####################################################################

######### Grid Search parameters for XGB (will only be used if MVAs contains "XGB"
XGBGridSearch= {
                "XGB_1": {'learning_rate':[0.1, 0.01, 0.001]},
                "XGB_2": {'gamma':[0.5, 1],'learning_rate':[0.1, 0.01]},
               }
#
#To choose just one value for a parameter you can just specify value but in a list 
#Like "XGB_1":{'gamma':[0.5],'learning_rate':[0.1, 0.01]} 
#Here gamma is fixed but learning_rate will be varied

#The above are just one/two paramter grids
#All other parameters are XGB default values
#But you give any number you want:
#example:
#XGBGridSearch= {'learning_rate':[0.1, 0.01, 0.001],      
#                'min_child_weight': [1, 5, 10],
#                'gamma': [0.5, 1, 1.5, 2, 5],
#                'subsample': [0.6, 0.8, 1.0],
#                'colsample_bytree': [0.6, 0.8, 1.0],
#                'max_depth': [3, 4, 5]}
#Just rememeber the larger the grid the more time optimization takes

######### DNN parameters and model (will only be used if MVAs contains "DNN"

#Example for DNN_1
modelDNN_DNN_1=Sequential()
modelDNN_DNN_1.add(Dense(2*len(features["DNN_1"]), kernel_initializer='glorot_normal', activation='relu', input_dim=len(features["DNN_1"])))
modelDNN_DNN_1.add(Dense(len(features["DNN_1"]), kernel_initializer='glorot_normal', activation='relu'))
modelDNN_DNN_1.add(Dropout(0.1))
modelDNN_DNN_1.add(Dense(1, kernel_initializer='glorot_uniform', activation='sigmoid'))
DNNDict={
         "DNN_1":{'epochs':10, 'batchsize':100, 'lr':0.001, 'model':modelDNN_DNN_1}
        }


#####################################################################

SigEffWPs=["80%","90%"] # Example for 80% and 90% Signal Efficiency Working Points

######### Reweighting scheme #Feature not available but planned
#Reweighting = 'pt-etaSig'
'''
Possible choices :
None : No reweighting
FlatpT : Binned flat in pT (default binning)
Flateta : Binned flat in eta (default binning)
pt-etaSig : To Signal pt-eta spectrum 
pt-etaBkg : To Background pt-eta spectrum
'''

#####Optional Features
#SaveDataFrameCSV=False #True will save the final dataframe with all features and MAV predictions
#RandomState=42 # Choose the same number everytime for reproducibility
#MVAlogplot=False #If true, MVA outputs are plotted in log scale
#Multicore=True #If True all CPU cores available are used XGB 
