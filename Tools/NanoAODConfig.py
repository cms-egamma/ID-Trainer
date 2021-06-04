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

branches=["Electron_*"]

#Files, Cuts and XsecWts should have the same number of elements
SigFiles = [
    #File 1
    'DYJetsToLL_M-50_v7_ElePromptGenMatched.root']
    #File 2
    #'/eos/user/a/akapoor/SWAN_projects/TauGun_Pt-15to500_14TeV_Run3Summer19MiniAOD-2021Scenario_106X_mcRun3_2021_NewCode2021.root'] #Add as many files you like

#Cuts to select appropriate signal
SigCuts= [
    #For File 1
    "(Electron_promptgenmatched == 1)"]
    #For File 2
    #"(matchedToGenEle == 1)"] #Cuts same as number of files (Kept like this because it maybe different for different files)

#Any extra xsec weight : Useful when stitching samples for either signal or background : '1' means no weight
SigXsecWts=[
    #For File 1
    1]
    #For File 2
    #1] #Weights same as number of files (Kept like this because it maybe different for different files)

#Files, Cuts and XsecWts should have the same number of elements
BkgFiles = [
    #File 1
    'DYJetsToLL_M-50_v7_ElePromptGenMatched.root'] #Add as many files you like 

#Cuts to select appropriate background
BkgCuts= [
    #For File 1 
    "(Electron_promptgenmatched == 0)"]#Cuts same as number of files (Kept like this because it maybe different for different files)

#Any extra xsec weight : Useful when stitching samples for either signal or background : '1' means no weight
BkgXsecWts=[
    #For File 1 
    2] #Weights same as number of files (Kept like this because it maybe different for different files) 

#####################################################################

testsize=0.2 #(0.2 means 20%)

#Common cuts for both signal and background (Would generally correspond to the training region)
CommonCut = "(Electron_pt>10)" 
#This is barrel and pt>10 GeV
#endcap would be
#(ele_pt > 10) & (abs(scl_eta)<1.566)

Tree = "Events" #Location/Name of tree inside Root files

################################

#MVAs to use as a list, e.g : ["XGB","DNN", "Genetic"]
MVAs = ["XGB_1"] 
#XGB and DNN are keywords so names can be XGB_new, DNN_old etc. But keep XGB and DNN in the names (That is how the framework identifies which algo to run

MVAColors = ["green"] #Plot colors for MVAs

MVALabels = {"XGB_1" : "XGB masscut"
            } #These labels can be anything (this is how you will identify them on plot legends)

################################
features = {
            "XGB_1":["Electron_pt", "Electron_deltaEtaSC", "Electron_r9"]
           } #Input features to MVA #Should be in your ntuples

feature_bins = {
                "XGB_1":[100, 100, 100]
               } #Binning used only for plotting features (should be in the same order as features), does not affect training
#template 
#np.linspace(lower boundary, upper boundary, totalbins+1)

#when not sure about the binning, you can just specify numbers, which will then correspond to total bins
#You can even specify lists like [10,20,30,100]

#EGamma WPs to compare to (Should be in your ntuple)
OverlayWP=["Electron_mvaFall17V2noIso_WP90"]
OverlayWPColors = ["black","purple"] #Colors on plots for WPs
#####################################################################

######### Grid Search parameters for XGB (will only be used if MVAs contains "XGB"
XGBGridSearch= {
                "XGB_1": {'learning_rate':[0.1, 0.01, 0.001]}
               }

Scaler = {"XGB_1":"MinMaxScaler"}
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



#####################################################################

SigEffWPs=["80%","90%"] # Example for 80% and 90% Signal Efficiency Working Points

######### Reweighting scheme #Feature not available but planned
Reweighing = 'Nothing'
ptbins = [10,30,40,50,100,5000] 
etabins = [-1.6,-1.0,1.0,1.2,1.6]
ptwtvar='Electron_pt'
etawtvar='Electron_eta'

'''
Possible choices :
Nothing : No reweighting
ptetaSig : To Signal pt-eta spectrum 
ptetaBkg : To Background pt-eta spectrum
'''

#####Optional Features
#SaveDataFrameCSV=False #True will save the final dataframe with all features and MAV predictions
#RandomState=42 # Choose the same number everytime for reproducibility
#MVAlogplot=False #If true, MVA outputs are plotted in log scale
#Multicore=True #If True all CPU cores available are used XGB 
