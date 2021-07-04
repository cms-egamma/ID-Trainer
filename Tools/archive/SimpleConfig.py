# In this file you can specify the training configuration

OutputDirName = 'Output'
Debug=True

SigFiles = ['/eos/cms/store/group/phys_egamma/electron_ntuple.root']
SigCuts= ["(matchedToGenEle == 1)"]
SigXsecWts=[1]

BkgFiles = ['/eos/cms/store/group/phys_egamma/electron_ntuple.root']
BkgCuts= ["(matchedToGenEle == 0)"]
BkgXsecWts=[1]

testsize=0.2 #(0.2 means 20%)
CommonCut = "(ele_pt > 10) & (abs(scl_eta)<1.442)" 
Tree = "ntuplizer/tree" #Location/Name of tree inside Root files

#-------------------------------
MVAs = ["XGB_1"] 
MVAColors = ["green"]

MVALabels = {"XGB_1" : "my first try",}

features = {"XGB_1":["ele_fbrem", "ele_deltaetain", "ele_deltaphiin", "ele_oldsigmaietaieta"]}
feature_bins = {"XGB_1":[30, 30, 30, 30]} 

XGBGridSearch= {"XGB_1": {'learning_rate':[0.1, 0.01, 0.001]}}

OverlayWP=['Fall17isoV1wpLoose','Fall17noIsoV1wpLoose']
OverlayWPColors = ["black","purple"] #Colors on plots for WPs

Scaler = {"XGB_1":"MinMaxScaler"}
#-------------------------------

SigEffWPs=["80%","90%"]

Reweighing = 'ptetaSig'
ptbins = [10,30,40,50,100,5000] 
etabins = [-1.6,-1.0,1.0,1.2,1.6]
