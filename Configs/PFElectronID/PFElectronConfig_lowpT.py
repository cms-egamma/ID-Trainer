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
OutputDirName = 'PFElectronConfig_lowpT' #All plots, models, config file will be stored here

Debug=False # If True, only a small subset of events/objects are used for either Signal or background #Useful for quick debugging

#Branches to read #Should be in the root files #Only the read branches can be later used for any purpose
branches=["scl_eta",
          "ele_pt",
          "matchedToGenEle",
          "matchedToGenPhoton",
          "matchedToGenTauJet",
          "matchedToHadron",
          "ele_convDist",
          "ele_convDcot",
          "EleMVACats",
          "ele_fbrem","ele_deltaetain", "ele_deltaphiin", "ele_oldsigmaietaieta", 
          "ele_oldhe", "ele_ep", "ele_olde15", "ele_eelepout",
          "ele_kfchi2", "ele_kfhits", "ele_expected_inner_hits","ele_dr03TkSumPt",
          "ele_dr03EcalRecHitSumEt","ele_dr03HcalTowerSumEt",
          "ele_ecalPFClusterIso","ele_hcalPFClusterIso",
          "ele_gsfchi2",
          'ele_conversionVertexFitProbability',
          "ele_nbrem",'ele_deltaetaseed','ele_hadronicOverEm','ele_olde25max','ele_olde55','passElectronSelection']
#branches=["Electron_*"]
#Possible examples
# ["Electron_*","matchingflag",]
# ["Electron_pt", "Electron_deltaEtaSC", "Electron_r9","Electron_eta"]
# You need to read branches to use them anywhere

##### If True, this will save the dataframe as a csv and the next time you run the same training with different parameters, it will be much faster
SaveDataFrameCSV=True
##### If branches and files are same a "previous" (not this one) training and SaveDataFrameCSV was True, you can switch on loadfromsaved and it will be much quicker to run the this time
loadfromsaved=False

#pt and eta bins of interest -------------------------------------------------------------------
#will be used for robustness studies and will also be used for 2D pt-eta reweighing if the reweighing option is True
ptbins = [5,7,8,10]
etabins = [-2.5,-2.2,-1.8,-1.6,-1.2,-0.8,-0.5,0.0,0.5,0.8,1.2,1.6,1.8,2.2,2.5]
ptwtvar='ele_pt'
etawtvar='scl_eta'
##pt and eta bins of interest -------------------------------------------------------------------

#Reweighting scheme -------------------------------------------------------------------
Reweighing = 'True' # This is independent of xsec reweighing (this reweighing will be done after taking into account xsec weight of multiple samples). Even if this is 'False', xsec reweighting will always be done.
WhichClassToReweightTo="IsolatedSignal" #2D pt-eta spectrum of all other classs will be reweighted to this class
#will only be used if Reweighing = 'True'
#Reweighting scheme -------------------------------------------------------------------

Classes = ['IsolatedSignal','NonIsolatedSignal','NonIsolatedBackground','FromHadronicTaus','FromPhotons'] 
ClassColors = ['#377eb8', '#ff7f00', '#4daf4a','#f781bf', '#a65628']

#dictionary of processes
CommonSel='(ele_pt < 10) & (abs(scl_eta) < 2.5)'

PromptSel='((matchedToGenEle == 1) | (matchedToGenEle == 2)) & (matchedToGenPhoton==0)'
bHadSel='(matchedToGenEle != 1) & (matchedToGenEle != 2) &  (matchedToHadron==3) & (matchedToGenTauJet==0) & (matchedToGenPhoton==0) & (index%10==0)'
QCDSel='(matchedToGenEle ==0) &  (matchedToHadron!=3) & (matchedToGenTauJet==0) & (matchedToGenPhoton==0)'
hadtauSel='(matchedToGenEle == 0) & (matchedToGenTauJet==1) & (matchedToGenPhoton==0)'
PhoSel='(matchedToGenEle != 1) & (matchedToGenEle != 2) &  (matchedToHadron==0) & (matchedToGenTauJet==0) & (matchedToGenPhoton==1)'

loc='/scratch/PFNtuples_July_correct/'
import os
if 'cern.ch' in os.uname()[1]: loc='/eos/cms/store/group/phys_egamma/akapoor/ntuple_ForPFID_July_Correct/ntuple_PFID_July_correct/'

processes=[
    {
        'Class':'IsolatedSignal',
        'path':[loc+'mc/DYJets_incl_MLL-50_TuneCP5_14TeV-madgraphMLM-pythia8/crab_DYJets_incl_MLL-50_TuneCP5_14TeV-madgraphMLM-pythia8_July2021newflaaddedclusterisog_ondisknow/210711_205559/0000/output_1.root',
                loc+'mc/DYJets_incl_MLL-50_TuneCP5_14TeV-madgraphMLM-pythia8/crab_DYJets_incl_MLL-50_TuneCP5_14TeV-madgraphMLM-pythia8_July2021newflaaddedclusterisog_ondisknow/210711_205559/0000/output_2.root',
                loc+'mc/DYJets_incl_MLL-50_TuneCP5_14TeV-madgraphMLM-pythia8/crab_DYJets_incl_MLL-50_TuneCP5_14TeV-madgraphMLM-pythia8_July2021newflaaddedclusterisog_ondisknow/210711_205559/0000/output_3.root',
                loc+'mc/DYJets_incl_MLL-50_TuneCP5_14TeV-madgraphMLM-pythia8/crab_DYJets_incl_MLL-50_TuneCP5_14TeV-madgraphMLM-pythia8_July2021newflaaddedclusterisog_ondisknow/210711_205559/0000/output_4.root',
                loc+'mc/DYJets_incl_MLL-50_TuneCP5_14TeV-madgraphMLM-pythia8/crab_DYJets_incl_MLL-50_TuneCP5_14TeV-madgraphMLM-pythia8_July2021newflaaddedclusterisog_ondisknow/210711_205559/0000/output_5.root',
                loc+'mc/DYJets_incl_MLL-50_TuneCP5_14TeV-madgraphMLM-pythia8/crab_DYJets_incl_MLL-50_TuneCP5_14TeV-madgraphMLM-pythia8_July2021newflaaddedclusterisog_ondisknow/210711_205559/0000/output_6.root',
                loc+'mc/DYToEE_M-50_NNPDF31_TuneCP5_14TeV-powheg-pythia8/crab_DYToEE_M-50_NNPDF31_TuneCP5_14TeV-powheg-pythia8_July2021newflaaddedclusterisog/210711_205609/0000/output_1.root',
                loc+'mc/DYToEE_M-50_NNPDF31_TuneCP5_14TeV-powheg-pythia8/crab_DYToEE_M-50_NNPDF31_TuneCP5_14TeV-powheg-pythia8_July2021newflaaddedclusterisog/210711_205609/0000/output_2.root',
                loc+'mc/DYToEE_M-50_NNPDF31_TuneCP5_14TeV-powheg-pythia8/crab_DYToEE_M-50_NNPDF31_TuneCP5_14TeV-powheg-pythia8_July2021newflaaddedclusterisog/210711_205609/0000/output_3.root',
                loc+'mc/DYToEE_M-50_NNPDF31_TuneCP5_14TeV-powheg-pythia8/crab_DYToEE_M-50_NNPDF31_TuneCP5_14TeV-powheg-pythia8_July2021newflaaddedclusterisog/210711_205609/0000/output_4.root',
                loc+'mc/DYToEE_M-50_NNPDF31_TuneCP5_14TeV-powheg-pythia8/crab_DYToEE_M-50_NNPDF31_TuneCP5_14TeV-powheg-pythia8_July2021newflaaddedclusterisog/210711_205609/0000/output_5.root',
                loc+'mc/ZprimeToTT_M3000_W30_TuneCP5_14TeV-madgraphMLM-pythia8/crab_ZprimeToTT_M3000_W30_TuneCP5_14TeV-madgraphMLM-pythia8_July2021newflaaddedclusterisog/210711_205507/0000/output_1.root',
                loc+'mc/ZprimeToTT_M4000_W40_TuneCP5_14TeV-madgraphMLM-pythia8/crab_ZprimeToTT_M4000_W40_TuneCP5_14TeV-madgraphMLM-pythia8_July2021newflaaddedclusterisog/210711_205513/0000/output_1.root',],
        'xsecwt': 1, #xsec wt if any, if none then it can be 1
        'selection':CommonSel+' & '+PromptSel, #selection for background
    },

    {
        'Class':'NonIsolatedSignal',
        'path':[loc+'mc/QCD_Pt_15to20_bcToE_TuneCP5_13TeV_pythia8/crab_QCD_Pt_15to20_bcToE_TuneCP5_13TeV_pythia8_July2021newflaaddedclusterisog/210711_210451/0000/output_1.root',
                loc+'mc/QCD_Pt_15to20_bcToE_TuneCP5_13TeV_pythia8/crab_QCD_Pt_15to20_bcToE_TuneCP5_13TeV_pythia8_July2021newflaaddedclusterisog/210711_210451/0000/output_2.root',
                loc+'mc/QCD_Pt_15to20_bcToE_TuneCP5_13TeV_pythia8/crab_QCD_Pt_15to20_bcToE_TuneCP5_13TeV_pythia8_July2021newflaaddedclusterisog/210711_210451/0000/output_3.root',
                loc+'mc/QCD_Pt_15to20_bcToE_TuneCP5_13TeV_pythia8/crab_QCD_Pt_15to20_bcToE_TuneCP5_13TeV_pythia8_July2021newflaaddedclusterisog/210711_210451/0000/output_4.root',
                loc+'mc/QCD_Pt_15to20_bcToE_TuneCP5_13TeV_pythia8/crab_QCD_Pt_15to20_bcToE_TuneCP5_13TeV_pythia8_July2021newflaaddedclusterisog/210711_210451/0000/output_5.root',
                loc+'mc/QCD_Pt_15to20_bcToE_TuneCP5_13TeV_pythia8/crab_QCD_Pt_15to20_bcToE_TuneCP5_13TeV_pythia8_July2021newflaaddedclusterisog/210711_210451/0000/output_6.root',
                loc+'mc/QCD_Pt_170to250_bcToE_TuneCP5_13TeV_pythia8/crab_QCD_Pt_170to250_bcToE_TuneCP5_13TeV_pythia8_July2021newflaaddedclusterisog/210711_210457/0000/output_1.root',
                loc+'mc/QCD_Pt_170to250_bcToE_TuneCP5_13TeV_pythia8/crab_QCD_Pt_170to250_bcToE_TuneCP5_13TeV_pythia8_July2021newflaaddedclusterisog/210711_210457/0000/output_10.root',
                loc+'mc/QCD_Pt_170to250_bcToE_TuneCP5_13TeV_pythia8/crab_QCD_Pt_170to250_bcToE_TuneCP5_13TeV_pythia8_July2021newflaaddedclusterisog/210711_210457/0000/output_11.root',
                loc+'mc/QCD_Pt_170to250_bcToE_TuneCP5_13TeV_pythia8/crab_QCD_Pt_170to250_bcToE_TuneCP5_13TeV_pythia8_July2021newflaaddedclusterisog/210711_210457/0000/output_12.root',
                loc+'mc/QCD_Pt_170to250_bcToE_TuneCP5_13TeV_pythia8/crab_QCD_Pt_170to250_bcToE_TuneCP5_13TeV_pythia8_July2021newflaaddedclusterisog/210711_210457/0000/output_13.root',
                loc+'mc/QCD_Pt_170to250_bcToE_TuneCP5_13TeV_pythia8/crab_QCD_Pt_170to250_bcToE_TuneCP5_13TeV_pythia8_July2021newflaaddedclusterisog/210711_210457/0000/output_14.root',
                loc+'mc/QCD_Pt_170to250_bcToE_TuneCP5_13TeV_pythia8/crab_QCD_Pt_170to250_bcToE_TuneCP5_13TeV_pythia8_July2021newflaaddedclusterisog/210711_210457/0000/output_2.root',
                loc+'mc/QCD_Pt_170to250_bcToE_TuneCP5_13TeV_pythia8/crab_QCD_Pt_170to250_bcToE_TuneCP5_13TeV_pythia8_July2021newflaaddedclusterisog/210711_210457/0000/output_3.root',
                loc+'mc/QCD_Pt_170to250_bcToE_TuneCP5_13TeV_pythia8/crab_QCD_Pt_170to250_bcToE_TuneCP5_13TeV_pythia8_July2021newflaaddedclusterisog/210711_210457/0000/output_4.root',
                loc+'mc/QCD_Pt_170to250_bcToE_TuneCP5_13TeV_pythia8/crab_QCD_Pt_170to250_bcToE_TuneCP5_13TeV_pythia8_July2021newflaaddedclusterisog/210711_210457/0000/output_5.root',
                loc+'mc/QCD_Pt_170to250_bcToE_TuneCP5_13TeV_pythia8/crab_QCD_Pt_170to250_bcToE_TuneCP5_13TeV_pythia8_July2021newflaaddedclusterisog/210711_210457/0000/output_6.root',
                loc+'mc/QCD_Pt_170to250_bcToE_TuneCP5_13TeV_pythia8/crab_QCD_Pt_170to250_bcToE_TuneCP5_13TeV_pythia8_July2021newflaaddedclusterisog/210711_210457/0000/output_7.root',
                loc+'mc/QCD_Pt_170to250_bcToE_TuneCP5_13TeV_pythia8/crab_QCD_Pt_170to250_bcToE_TuneCP5_13TeV_pythia8_July2021newflaaddedclusterisog/210711_210457/0000/output_8.root',
                loc+'mc/QCD_Pt_170to250_bcToE_TuneCP5_13TeV_pythia8/crab_QCD_Pt_170to250_bcToE_TuneCP5_13TeV_pythia8_July2021newflaaddedclusterisog/210711_210457/0000/output_9.root',
                loc+'mc/QCD_Pt_20to30_bcToE_TuneCP5_13TeV_pythia8/crab_QCD_Pt_20to30_bcToE_TuneCP5_13TeV_pythia8_July2021newflaaddedclusterisog/210711_210503/0000/output_1.root',
                loc+'mc/QCD_Pt_20to30_bcToE_TuneCP5_13TeV_pythia8/crab_QCD_Pt_20to30_bcToE_TuneCP5_13TeV_pythia8_July2021newflaaddedclusterisog/210711_210503/0000/output_10.root',
                loc+'mc/QCD_Pt_20to30_bcToE_TuneCP5_13TeV_pythia8/crab_QCD_Pt_20to30_bcToE_TuneCP5_13TeV_pythia8_July2021newflaaddedclusterisog/210711_210503/0000/output_2.root',
                loc+'mc/QCD_Pt_20to30_bcToE_TuneCP5_13TeV_pythia8/crab_QCD_Pt_20to30_bcToE_TuneCP5_13TeV_pythia8_July2021newflaaddedclusterisog/210711_210503/0000/output_3.root',
                loc+'mc/QCD_Pt_20to30_bcToE_TuneCP5_13TeV_pythia8/crab_QCD_Pt_20to30_bcToE_TuneCP5_13TeV_pythia8_July2021newflaaddedclusterisog/210711_210503/0000/output_4.root',
                loc+'mc/QCD_Pt_20to30_bcToE_TuneCP5_13TeV_pythia8/crab_QCD_Pt_20to30_bcToE_TuneCP5_13TeV_pythia8_July2021newflaaddedclusterisog/210711_210503/0000/output_5.root',
                loc+'mc/QCD_Pt_20to30_bcToE_TuneCP5_13TeV_pythia8/crab_QCD_Pt_20to30_bcToE_TuneCP5_13TeV_pythia8_July2021newflaaddedclusterisog/210711_210503/0000/output_6.root',
                loc+'mc/QCD_Pt_20to30_bcToE_TuneCP5_13TeV_pythia8/crab_QCD_Pt_20to30_bcToE_TuneCP5_13TeV_pythia8_July2021newflaaddedclusterisog/210711_210503/0000/output_7.root',
                loc+'mc/QCD_Pt_20to30_bcToE_TuneCP5_13TeV_pythia8/crab_QCD_Pt_20to30_bcToE_TuneCP5_13TeV_pythia8_July2021newflaaddedclusterisog/210711_210503/0000/output_8.root',
                loc+'mc/QCD_Pt_20to30_bcToE_TuneCP5_13TeV_pythia8/crab_QCD_Pt_20to30_bcToE_TuneCP5_13TeV_pythia8_July2021newflaaddedclusterisog/210711_210503/0000/output_9.root',
                loc+'mc/QCD_Pt_250toInf_bcToE_TuneCP5_13TeV_pythia8/crab_QCD_Pt_250toInf_bcToE_TuneCP5_13TeV_pythia8_July2021newflaaddedclusterisog/210711_210509/0000/output_1.root',
                loc+'mc/QCD_Pt_250toInf_bcToE_TuneCP5_13TeV_pythia8/crab_QCD_Pt_250toInf_bcToE_TuneCP5_13TeV_pythia8_July2021newflaaddedclusterisog/210711_210509/0000/output_10.root',
                loc+'mc/QCD_Pt_250toInf_bcToE_TuneCP5_13TeV_pythia8/crab_QCD_Pt_250toInf_bcToE_TuneCP5_13TeV_pythia8_July2021newflaaddedclusterisog/210711_210509/0000/output_11.root',
                loc+'mc/QCD_Pt_250toInf_bcToE_TuneCP5_13TeV_pythia8/crab_QCD_Pt_250toInf_bcToE_TuneCP5_13TeV_pythia8_July2021newflaaddedclusterisog/210711_210509/0000/output_12.root',
                loc+'mc/QCD_Pt_250toInf_bcToE_TuneCP5_13TeV_pythia8/crab_QCD_Pt_250toInf_bcToE_TuneCP5_13TeV_pythia8_July2021newflaaddedclusterisog/210711_210509/0000/output_13.root',
                loc+'mc/QCD_Pt_250toInf_bcToE_TuneCP5_13TeV_pythia8/crab_QCD_Pt_250toInf_bcToE_TuneCP5_13TeV_pythia8_July2021newflaaddedclusterisog/210711_210509/0000/output_14.root',
                loc+'mc/QCD_Pt_250toInf_bcToE_TuneCP5_13TeV_pythia8/crab_QCD_Pt_250toInf_bcToE_TuneCP5_13TeV_pythia8_July2021newflaaddedclusterisog/210711_210509/0000/output_15.root',
                loc+'mc/QCD_Pt_250toInf_bcToE_TuneCP5_13TeV_pythia8/crab_QCD_Pt_250toInf_bcToE_TuneCP5_13TeV_pythia8_July2021newflaaddedclusterisog/210711_210509/0000/output_16.root',
                loc+'mc/QCD_Pt_250toInf_bcToE_TuneCP5_13TeV_pythia8/crab_QCD_Pt_250toInf_bcToE_TuneCP5_13TeV_pythia8_July2021newflaaddedclusterisog/210711_210509/0000/output_17.root',
                loc+'mc/QCD_Pt_250toInf_bcToE_TuneCP5_13TeV_pythia8/crab_QCD_Pt_250toInf_bcToE_TuneCP5_13TeV_pythia8_July2021newflaaddedclusterisog/210711_210509/0000/output_18.root',
                loc+'mc/QCD_Pt_250toInf_bcToE_TuneCP5_13TeV_pythia8/crab_QCD_Pt_250toInf_bcToE_TuneCP5_13TeV_pythia8_July2021newflaaddedclusterisog/210711_210509/0000/output_2.root',
                loc+'mc/QCD_Pt_250toInf_bcToE_TuneCP5_13TeV_pythia8/crab_QCD_Pt_250toInf_bcToE_TuneCP5_13TeV_pythia8_July2021newflaaddedclusterisog/210711_210509/0000/output_3.root',
                loc+'mc/QCD_Pt_250toInf_bcToE_TuneCP5_13TeV_pythia8/crab_QCD_Pt_250toInf_bcToE_TuneCP5_13TeV_pythia8_July2021newflaaddedclusterisog/210711_210509/0000/output_4.root',
                loc+'mc/QCD_Pt_250toInf_bcToE_TuneCP5_13TeV_pythia8/crab_QCD_Pt_250toInf_bcToE_TuneCP5_13TeV_pythia8_July2021newflaaddedclusterisog/210711_210509/0000/output_5.root',
                loc+'mc/QCD_Pt_250toInf_bcToE_TuneCP5_13TeV_pythia8/crab_QCD_Pt_250toInf_bcToE_TuneCP5_13TeV_pythia8_July2021newflaaddedclusterisog/210711_210509/0000/output_6.root',
                loc+'mc/QCD_Pt_250toInf_bcToE_TuneCP5_13TeV_pythia8/crab_QCD_Pt_250toInf_bcToE_TuneCP5_13TeV_pythia8_July2021newflaaddedclusterisog/210711_210509/0000/output_7.root',
                loc+'mc/QCD_Pt_250toInf_bcToE_TuneCP5_13TeV_pythia8/crab_QCD_Pt_250toInf_bcToE_TuneCP5_13TeV_pythia8_July2021newflaaddedclusterisog/210711_210509/0000/output_8.root',
                loc+'mc/QCD_Pt_250toInf_bcToE_TuneCP5_13TeV_pythia8/crab_QCD_Pt_250toInf_bcToE_TuneCP5_13TeV_pythia8_July2021newflaaddedclusterisog/210711_210509/0000/output_9.root',
                loc+'mc/QCD_Pt_30to80_bcToE_TuneCP5_13TeV_pythia8/crab_QCD_Pt_30to80_bcToE_TuneCP5_13TeV_pythia8_July2021newflaaddedclusterisog/210711_210515/0000/output_1.root',
                loc+'mc/QCD_Pt_30to80_bcToE_TuneCP5_13TeV_pythia8/crab_QCD_Pt_30to80_bcToE_TuneCP5_13TeV_pythia8_July2021newflaaddedclusterisog/210711_210515/0000/output_10.root',
                loc+'mc/QCD_Pt_30to80_bcToE_TuneCP5_13TeV_pythia8/crab_QCD_Pt_30to80_bcToE_TuneCP5_13TeV_pythia8_July2021newflaaddedclusterisog/210711_210515/0000/output_11.root',
                loc+'mc/QCD_Pt_30to80_bcToE_TuneCP5_13TeV_pythia8/crab_QCD_Pt_30to80_bcToE_TuneCP5_13TeV_pythia8_July2021newflaaddedclusterisog/210711_210515/0000/output_12.root',
                loc+'mc/QCD_Pt_30to80_bcToE_TuneCP5_13TeV_pythia8/crab_QCD_Pt_30to80_bcToE_TuneCP5_13TeV_pythia8_July2021newflaaddedclusterisog/210711_210515/0000/output_13.root',
                loc+'mc/QCD_Pt_30to80_bcToE_TuneCP5_13TeV_pythia8/crab_QCD_Pt_30to80_bcToE_TuneCP5_13TeV_pythia8_July2021newflaaddedclusterisog/210711_210515/0000/output_14.root',
                loc+'mc/QCD_Pt_30to80_bcToE_TuneCP5_13TeV_pythia8/crab_QCD_Pt_30to80_bcToE_TuneCP5_13TeV_pythia8_July2021newflaaddedclusterisog/210711_210515/0000/output_15.root',
                loc+'mc/QCD_Pt_30to80_bcToE_TuneCP5_13TeV_pythia8/crab_QCD_Pt_30to80_bcToE_TuneCP5_13TeV_pythia8_July2021newflaaddedclusterisog/210711_210515/0000/output_16.root',
                loc+'mc/QCD_Pt_30to80_bcToE_TuneCP5_13TeV_pythia8/crab_QCD_Pt_30to80_bcToE_TuneCP5_13TeV_pythia8_July2021newflaaddedclusterisog/210711_210515/0000/output_17.root',
                loc+'mc/QCD_Pt_30to80_bcToE_TuneCP5_13TeV_pythia8/crab_QCD_Pt_30to80_bcToE_TuneCP5_13TeV_pythia8_July2021newflaaddedclusterisog/210711_210515/0000/output_2.root',
                loc+'mc/QCD_Pt_30to80_bcToE_TuneCP5_13TeV_pythia8/crab_QCD_Pt_30to80_bcToE_TuneCP5_13TeV_pythia8_July2021newflaaddedclusterisog/210711_210515/0000/output_3.root',
                loc+'mc/QCD_Pt_30to80_bcToE_TuneCP5_13TeV_pythia8/crab_QCD_Pt_30to80_bcToE_TuneCP5_13TeV_pythia8_July2021newflaaddedclusterisog/210711_210515/0000/output_4.root',
                loc+'mc/QCD_Pt_30to80_bcToE_TuneCP5_13TeV_pythia8/crab_QCD_Pt_30to80_bcToE_TuneCP5_13TeV_pythia8_July2021newflaaddedclusterisog/210711_210515/0000/output_5.root',
                loc+'mc/QCD_Pt_30to80_bcToE_TuneCP5_13TeV_pythia8/crab_QCD_Pt_30to80_bcToE_TuneCP5_13TeV_pythia8_July2021newflaaddedclusterisog/210711_210515/0000/output_6.root',
                loc+'mc/QCD_Pt_30to80_bcToE_TuneCP5_13TeV_pythia8/crab_QCD_Pt_30to80_bcToE_TuneCP5_13TeV_pythia8_July2021newflaaddedclusterisog/210711_210515/0000/output_7.root',
                loc+'mc/QCD_Pt_30to80_bcToE_TuneCP5_13TeV_pythia8/crab_QCD_Pt_30to80_bcToE_TuneCP5_13TeV_pythia8_July2021newflaaddedclusterisog/210711_210515/0000/output_8.root',
                loc+'mc/QCD_Pt_30to80_bcToE_TuneCP5_13TeV_pythia8/crab_QCD_Pt_30to80_bcToE_TuneCP5_13TeV_pythia8_July2021newflaaddedclusterisog/210711_210515/0000/output_9.root',
                loc+'mc/QCD_Pt_80to170_bcToE_TuneCP5_13TeV_pythia8/crab_QCD_Pt_80to170_bcToE_TuneCP5_13TeV_pythia8_July2021newflaaddedclusterisog/210711_210521/0000/output_1.root',
                loc+'mc/QCD_Pt_80to170_bcToE_TuneCP5_13TeV_pythia8/crab_QCD_Pt_80to170_bcToE_TuneCP5_13TeV_pythia8_July2021newflaaddedclusterisog/210711_210521/0000/output_10.root',
                loc+'mc/QCD_Pt_80to170_bcToE_TuneCP5_13TeV_pythia8/crab_QCD_Pt_80to170_bcToE_TuneCP5_13TeV_pythia8_July2021newflaaddedclusterisog/210711_210521/0000/output_11.root',
                loc+'mc/QCD_Pt_80to170_bcToE_TuneCP5_13TeV_pythia8/crab_QCD_Pt_80to170_bcToE_TuneCP5_13TeV_pythia8_July2021newflaaddedclusterisog/210711_210521/0000/output_12.root',
                loc+'mc/QCD_Pt_80to170_bcToE_TuneCP5_13TeV_pythia8/crab_QCD_Pt_80to170_bcToE_TuneCP5_13TeV_pythia8_July2021newflaaddedclusterisog/210711_210521/0000/output_13.root',
                loc+'mc/QCD_Pt_80to170_bcToE_TuneCP5_13TeV_pythia8/crab_QCD_Pt_80to170_bcToE_TuneCP5_13TeV_pythia8_July2021newflaaddedclusterisog/210711_210521/0000/output_14.root',
                loc+'mc/QCD_Pt_80to170_bcToE_TuneCP5_13TeV_pythia8/crab_QCD_Pt_80to170_bcToE_TuneCP5_13TeV_pythia8_July2021newflaaddedclusterisog/210711_210521/0000/output_15.root',
                loc+'mc/QCD_Pt_80to170_bcToE_TuneCP5_13TeV_pythia8/crab_QCD_Pt_80to170_bcToE_TuneCP5_13TeV_pythia8_July2021newflaaddedclusterisog/210711_210521/0000/output_16.root',
                loc+'mc/QCD_Pt_80to170_bcToE_TuneCP5_13TeV_pythia8/crab_QCD_Pt_80to170_bcToE_TuneCP5_13TeV_pythia8_July2021newflaaddedclusterisog/210711_210521/0000/output_17.root',
                loc+'mc/QCD_Pt_80to170_bcToE_TuneCP5_13TeV_pythia8/crab_QCD_Pt_80to170_bcToE_TuneCP5_13TeV_pythia8_July2021newflaaddedclusterisog/210711_210521/0000/output_18.root',
                loc+'mc/QCD_Pt_80to170_bcToE_TuneCP5_13TeV_pythia8/crab_QCD_Pt_80to170_bcToE_TuneCP5_13TeV_pythia8_July2021newflaaddedclusterisog/210711_210521/0000/output_19.root',
                loc+'mc/QCD_Pt_80to170_bcToE_TuneCP5_13TeV_pythia8/crab_QCD_Pt_80to170_bcToE_TuneCP5_13TeV_pythia8_July2021newflaaddedclusterisog/210711_210521/0000/output_2.root',
                loc+'mc/QCD_Pt_80to170_bcToE_TuneCP5_13TeV_pythia8/crab_QCD_Pt_80to170_bcToE_TuneCP5_13TeV_pythia8_July2021newflaaddedclusterisog/210711_210521/0000/output_20.root',
                loc+'mc/QCD_Pt_80to170_bcToE_TuneCP5_13TeV_pythia8/crab_QCD_Pt_80to170_bcToE_TuneCP5_13TeV_pythia8_July2021newflaaddedclusterisog/210711_210521/0000/output_3.root',
                loc+'mc/QCD_Pt_80to170_bcToE_TuneCP5_13TeV_pythia8/crab_QCD_Pt_80to170_bcToE_TuneCP5_13TeV_pythia8_July2021newflaaddedclusterisog/210711_210521/0000/output_4.root',
                loc+'mc/QCD_Pt_80to170_bcToE_TuneCP5_13TeV_pythia8/crab_QCD_Pt_80to170_bcToE_TuneCP5_13TeV_pythia8_July2021newflaaddedclusterisog/210711_210521/0000/output_5.root',
                loc+'mc/QCD_Pt_80to170_bcToE_TuneCP5_13TeV_pythia8/crab_QCD_Pt_80to170_bcToE_TuneCP5_13TeV_pythia8_July2021newflaaddedclusterisog/210711_210521/0000/output_6.root',
                loc+'mc/QCD_Pt_80to170_bcToE_TuneCP5_13TeV_pythia8/crab_QCD_Pt_80to170_bcToE_TuneCP5_13TeV_pythia8_July2021newflaaddedclusterisog/210711_210521/0000/output_7.root',
                loc+'mc/QCD_Pt_80to170_bcToE_TuneCP5_13TeV_pythia8/crab_QCD_Pt_80to170_bcToE_TuneCP5_13TeV_pythia8_July2021newflaaddedclusterisog/210711_210521/0000/output_8.root',
                loc+'mc/QCD_Pt_80to170_bcToE_TuneCP5_13TeV_pythia8/crab_QCD_Pt_80to170_bcToE_TuneCP5_13TeV_pythia8_July2021newflaaddedclusterisog/210711_210521/0000/output_9.root',],
        'xsecwt': 1, #xsec wt if any, if none then it can be 1
        'selection':CommonSel+' & '+bHadSel, #selection for background
    },

    
    {
        'Class':'NonIsolatedBackground',
        'path':[loc+'mc/QCD_Pt-120to170_EMEnriched_TuneCP5_14TeV_pythia8/crab_QCD_Pt-120to170_EMEnriched_TuneCP5_14TeV_pythia8_July2021newflaaddedclusterisog/210711_205519/0000/output_1.root',
                loc+'mc/QCD_Pt-170to300_EMEnriched_TuneCP5_14TeV_pythia8/crab_QCD_Pt-170to300_EMEnriched_TuneCP5_14TeV_pythia8/210711_205641/0000/output_1.root',
                loc+'mc/QCD_Pt-30to50_EMEnriched_TuneCP5_14TeV_pythia8/crab_QCD_Pt-30to50_EMEnriched_TuneCP5_14TeV_pythia8_July2021newflaaddedclusterisog_ondisknow/210711_205537/0000/output_1.root',
                loc+'mc/QCD_Pt-50to80_EMEnriched_TuneCP5_14TeV_pythia8/crab_QCD_Pt-50to80_EMEnriched_TuneCP5_14TeV_pythia8_July2021newflaaddedclusterisog_ondisk/210711_205546/0000/output_1.root',
                loc+'mc/QCD_Pt-50to80_EMEnriched_TuneCP5_14TeV_pythia8/crab_QCD_Pt-50to80_EMEnriched_TuneCP5_14TeV_pythia8_July2021newflaaddedclusterisog_ondisk/210711_205546/0000/output_10.root',
                loc+'mc/QCD_Pt-50to80_EMEnriched_TuneCP5_14TeV_pythia8/crab_QCD_Pt-50to80_EMEnriched_TuneCP5_14TeV_pythia8_July2021newflaaddedclusterisog_ondisk/210711_205546/0000/output_11.root',
                loc+'mc/QCD_Pt-50to80_EMEnriched_TuneCP5_14TeV_pythia8/crab_QCD_Pt-50to80_EMEnriched_TuneCP5_14TeV_pythia8_July2021newflaaddedclusterisog_ondisk/210711_205546/0000/output_12.root',
                loc+'mc/QCD_Pt-50to80_EMEnriched_TuneCP5_14TeV_pythia8/crab_QCD_Pt-50to80_EMEnriched_TuneCP5_14TeV_pythia8_July2021newflaaddedclusterisog_ondisk/210711_205546/0000/output_13.root',
                loc+'mc/QCD_Pt-50to80_EMEnriched_TuneCP5_14TeV_pythia8/crab_QCD_Pt-50to80_EMEnriched_TuneCP5_14TeV_pythia8_July2021newflaaddedclusterisog_ondisk/210711_205546/0000/output_2.root',
                loc+'mc/QCD_Pt-50to80_EMEnriched_TuneCP5_14TeV_pythia8/crab_QCD_Pt-50to80_EMEnriched_TuneCP5_14TeV_pythia8_July2021newflaaddedclusterisog_ondisk/210711_205546/0000/output_3.root',
                loc+'mc/QCD_Pt-50to80_EMEnriched_TuneCP5_14TeV_pythia8/crab_QCD_Pt-50to80_EMEnriched_TuneCP5_14TeV_pythia8_July2021newflaaddedclusterisog_ondisk/210711_205546/0000/output_4.root',
                loc+'mc/QCD_Pt-50to80_EMEnriched_TuneCP5_14TeV_pythia8/crab_QCD_Pt-50to80_EMEnriched_TuneCP5_14TeV_pythia8_July2021newflaaddedclusterisog_ondisk/210711_205546/0000/output_5.root',
                loc+'mc/QCD_Pt-50to80_EMEnriched_TuneCP5_14TeV_pythia8/crab_QCD_Pt-50to80_EMEnriched_TuneCP5_14TeV_pythia8_July2021newflaaddedclusterisog_ondisk/210711_205546/0000/output_6.root',
                loc+'mc/QCD_Pt-50to80_EMEnriched_TuneCP5_14TeV_pythia8/crab_QCD_Pt-50to80_EMEnriched_TuneCP5_14TeV_pythia8_July2021newflaaddedclusterisog_ondisk/210711_205546/0000/output_7.root',
                loc+'mc/QCD_Pt-50to80_EMEnriched_TuneCP5_14TeV_pythia8/crab_QCD_Pt-50to80_EMEnriched_TuneCP5_14TeV_pythia8_July2021newflaaddedclusterisog_ondisk/210711_205546/0000/output_8.root',
                loc+'mc/QCD_Pt-50to80_EMEnriched_TuneCP5_14TeV_pythia8/crab_QCD_Pt-50to80_EMEnriched_TuneCP5_14TeV_pythia8_July2021newflaaddedclusterisog_ondisk/210711_205546/0000/output_9.root',
                loc+'mc/QCD_Pt-80to120_EMEnriched_TuneCP5_14TeV_pythia8/crab_QCD_Pt-80to120_EMEnriched_TuneCP5_14TeV_pythia8_July2021newflaaddedclusterisog/210711_205553/0000/output_1.root'],
        'xsecwt': 1, #xsec wt if any, if none then it can be 1
        'selection':CommonSel+' & '+QCDSel, #selection for background
    },

    
    {
        'Class':'FromHadronicTaus',
        'path':[loc+'mc/TauGun_Pt-15to500_14TeV-pythia8/crab_TauGun_Pt-15to500_14TeV-pythia8_July2021newflaaddedclusterisog/210711_205615/0000/output_1.root',
                loc+'mc/TauGun_Pt-15to500_14TeV-pythia8/crab_TauGun_Pt-15to500_14TeV-pythia8_July2021newflaaddedclusterisog/210711_205615/0000/output_2.root'],
        'xsecwt': 1, #xsec wt if any, if none then it can be 1
        'selection':CommonSel+' & '+hadtauSel, #selection for background
    },

    {
        'Class':'FromPhotons',
        'path':[loc+'mc/GJet_Pt-20to40_DoubleEMEnriched_MGG-80toInf_TuneCP5_14TeV_Pythia8/crab_GJet_Pt-20to40_DoubleEMEnriched_MGG-80toInf_TuneCP5_14TeV_Pythia8_July2021newflaaddedclusterisog/210711_205621/0000/output_1.root',
                loc+'mc/GJet_Pt-20to40_DoubleEMEnriched_MGG-80toInf_TuneCP5_14TeV_Pythia8/crab_GJet_Pt-20to40_DoubleEMEnriched_MGG-80toInf_TuneCP5_14TeV_Pythia8_July2021newflaaddedclusterisog/210711_205621/0000/output_10.root',
                loc+'mc/GJet_Pt-20to40_DoubleEMEnriched_MGG-80toInf_TuneCP5_14TeV_Pythia8/crab_GJet_Pt-20to40_DoubleEMEnriched_MGG-80toInf_TuneCP5_14TeV_Pythia8_July2021newflaaddedclusterisog/210711_205621/0000/output_11.root',
                loc+'mc/GJet_Pt-20to40_DoubleEMEnriched_MGG-80toInf_TuneCP5_14TeV_Pythia8/crab_GJet_Pt-20to40_DoubleEMEnriched_MGG-80toInf_TuneCP5_14TeV_Pythia8_July2021newflaaddedclusterisog/210711_205621/0000/output_12.root',
                loc+'mc/GJet_Pt-20to40_DoubleEMEnriched_MGG-80toInf_TuneCP5_14TeV_Pythia8/crab_GJet_Pt-20to40_DoubleEMEnriched_MGG-80toInf_TuneCP5_14TeV_Pythia8_July2021newflaaddedclusterisog/210711_205621/0000/output_13.root',
                loc+'mc/GJet_Pt-20to40_DoubleEMEnriched_MGG-80toInf_TuneCP5_14TeV_Pythia8/crab_GJet_Pt-20to40_DoubleEMEnriched_MGG-80toInf_TuneCP5_14TeV_Pythia8_July2021newflaaddedclusterisog/210711_205621/0000/output_14.root',
                loc+'mc/GJet_Pt-20to40_DoubleEMEnriched_MGG-80toInf_TuneCP5_14TeV_Pythia8/crab_GJet_Pt-20to40_DoubleEMEnriched_MGG-80toInf_TuneCP5_14TeV_Pythia8_July2021newflaaddedclusterisog/210711_205621/0000/output_15.root',
                loc+'mc/GJet_Pt-20to40_DoubleEMEnriched_MGG-80toInf_TuneCP5_14TeV_Pythia8/crab_GJet_Pt-20to40_DoubleEMEnriched_MGG-80toInf_TuneCP5_14TeV_Pythia8_July2021newflaaddedclusterisog/210711_205621/0000/output_16.root',
                loc+'mc/GJet_Pt-20to40_DoubleEMEnriched_MGG-80toInf_TuneCP5_14TeV_Pythia8/crab_GJet_Pt-20to40_DoubleEMEnriched_MGG-80toInf_TuneCP5_14TeV_Pythia8_July2021newflaaddedclusterisog/210711_205621/0000/output_17.root',
                loc+'mc/GJet_Pt-20to40_DoubleEMEnriched_MGG-80toInf_TuneCP5_14TeV_Pythia8/crab_GJet_Pt-20to40_DoubleEMEnriched_MGG-80toInf_TuneCP5_14TeV_Pythia8_July2021newflaaddedclusterisog/210711_205621/0000/output_18.root',
                loc+'mc/GJet_Pt-20to40_DoubleEMEnriched_MGG-80toInf_TuneCP5_14TeV_Pythia8/crab_GJet_Pt-20to40_DoubleEMEnriched_MGG-80toInf_TuneCP5_14TeV_Pythia8_July2021newflaaddedclusterisog/210711_205621/0000/output_19.root',
                loc+'mc/GJet_Pt-20to40_DoubleEMEnriched_MGG-80toInf_TuneCP5_14TeV_Pythia8/crab_GJet_Pt-20to40_DoubleEMEnriched_MGG-80toInf_TuneCP5_14TeV_Pythia8_July2021newflaaddedclusterisog/210711_205621/0000/output_2.root',
                loc+'mc/GJet_Pt-20to40_DoubleEMEnriched_MGG-80toInf_TuneCP5_14TeV_Pythia8/crab_GJet_Pt-20to40_DoubleEMEnriched_MGG-80toInf_TuneCP5_14TeV_Pythia8_July2021newflaaddedclusterisog/210711_205621/0000/output_20.root',
                loc+'mc/GJet_Pt-20to40_DoubleEMEnriched_MGG-80toInf_TuneCP5_14TeV_Pythia8/crab_GJet_Pt-20to40_DoubleEMEnriched_MGG-80toInf_TuneCP5_14TeV_Pythia8_July2021newflaaddedclusterisog/210711_205621/0000/output_21.root',
                loc+'mc/GJet_Pt-20to40_DoubleEMEnriched_MGG-80toInf_TuneCP5_14TeV_Pythia8/crab_GJet_Pt-20to40_DoubleEMEnriched_MGG-80toInf_TuneCP5_14TeV_Pythia8_July2021newflaaddedclusterisog/210711_205621/0000/output_3.root',
                loc+'mc/GJet_Pt-20to40_DoubleEMEnriched_MGG-80toInf_TuneCP5_14TeV_Pythia8/crab_GJet_Pt-20to40_DoubleEMEnriched_MGG-80toInf_TuneCP5_14TeV_Pythia8_July2021newflaaddedclusterisog/210711_205621/0000/output_4.root',
                loc+'mc/GJet_Pt-20to40_DoubleEMEnriched_MGG-80toInf_TuneCP5_14TeV_Pythia8/crab_GJet_Pt-20to40_DoubleEMEnriched_MGG-80toInf_TuneCP5_14TeV_Pythia8_July2021newflaaddedclusterisog/210711_205621/0000/output_5.root',
                loc+'mc/GJet_Pt-20to40_DoubleEMEnriched_MGG-80toInf_TuneCP5_14TeV_Pythia8/crab_GJet_Pt-20to40_DoubleEMEnriched_MGG-80toInf_TuneCP5_14TeV_Pythia8_July2021newflaaddedclusterisog/210711_205621/0000/output_6.root',
                loc+'mc/GJet_Pt-20to40_DoubleEMEnriched_MGG-80toInf_TuneCP5_14TeV_Pythia8/crab_GJet_Pt-20to40_DoubleEMEnriched_MGG-80toInf_TuneCP5_14TeV_Pythia8_July2021newflaaddedclusterisog/210711_205621/0000/output_7.root',
                loc+'mc/GJet_Pt-20to40_DoubleEMEnriched_MGG-80toInf_TuneCP5_14TeV_Pythia8/crab_GJet_Pt-20to40_DoubleEMEnriched_MGG-80toInf_TuneCP5_14TeV_Pythia8_July2021newflaaddedclusterisog/210711_205621/0000/output_8.root',
                loc+'mc/GJet_Pt-20to40_DoubleEMEnriched_MGG-80toInf_TuneCP5_14TeV_Pythia8/crab_GJet_Pt-20to40_DoubleEMEnriched_MGG-80toInf_TuneCP5_14TeV_Pythia8_July2021newflaaddedclusterisog/210711_205621/0000/output_9.root',
                loc+'mc/GJet_Pt-20toInf_DoubleEMEnriched_MGG-40to80_TuneCP5_14TeV_Pythia8/crab_GJet_Pt-20toInf_DoubleEMEnriched_MGG-40to80_TuneCP5_14TeV_Pythia8_July2021newflaaddedclusterisog/210711_205628/0000/output_1.root',
                loc+'mc/GJet_Pt-20toInf_DoubleEMEnriched_MGG-40to80_TuneCP5_14TeV_Pythia8/crab_GJet_Pt-20toInf_DoubleEMEnriched_MGG-40to80_TuneCP5_14TeV_Pythia8_July2021newflaaddedclusterisog/210711_205628/0000/output_10.root',
                loc+'mc/GJet_Pt-20toInf_DoubleEMEnriched_MGG-40to80_TuneCP5_14TeV_Pythia8/crab_GJet_Pt-20toInf_DoubleEMEnriched_MGG-40to80_TuneCP5_14TeV_Pythia8_July2021newflaaddedclusterisog/210711_205628/0000/output_11.root',
                loc+'mc/GJet_Pt-20toInf_DoubleEMEnriched_MGG-40to80_TuneCP5_14TeV_Pythia8/crab_GJet_Pt-20toInf_DoubleEMEnriched_MGG-40to80_TuneCP5_14TeV_Pythia8_July2021newflaaddedclusterisog/210711_205628/0000/output_12.root',
                loc+'mc/GJet_Pt-20toInf_DoubleEMEnriched_MGG-40to80_TuneCP5_14TeV_Pythia8/crab_GJet_Pt-20toInf_DoubleEMEnriched_MGG-40to80_TuneCP5_14TeV_Pythia8_July2021newflaaddedclusterisog/210711_205628/0000/output_13.root',
                loc+'mc/GJet_Pt-20toInf_DoubleEMEnriched_MGG-40to80_TuneCP5_14TeV_Pythia8/crab_GJet_Pt-20toInf_DoubleEMEnriched_MGG-40to80_TuneCP5_14TeV_Pythia8_July2021newflaaddedclusterisog/210711_205628/0000/output_2.root',
                loc+'mc/GJet_Pt-20toInf_DoubleEMEnriched_MGG-40to80_TuneCP5_14TeV_Pythia8/crab_GJet_Pt-20toInf_DoubleEMEnriched_MGG-40to80_TuneCP5_14TeV_Pythia8_July2021newflaaddedclusterisog/210711_205628/0000/output_3.root',
                loc+'mc/GJet_Pt-20toInf_DoubleEMEnriched_MGG-40to80_TuneCP5_14TeV_Pythia8/crab_GJet_Pt-20toInf_DoubleEMEnriched_MGG-40to80_TuneCP5_14TeV_Pythia8_July2021newflaaddedclusterisog/210711_205628/0000/output_4.root',
                loc+'mc/GJet_Pt-20toInf_DoubleEMEnriched_MGG-40to80_TuneCP5_14TeV_Pythia8/crab_GJet_Pt-20toInf_DoubleEMEnriched_MGG-40to80_TuneCP5_14TeV_Pythia8_July2021newflaaddedclusterisog/210711_205628/0000/output_5.root',
                loc+'mc/GJet_Pt-20toInf_DoubleEMEnriched_MGG-40to80_TuneCP5_14TeV_Pythia8/crab_GJet_Pt-20toInf_DoubleEMEnriched_MGG-40to80_TuneCP5_14TeV_Pythia8_July2021newflaaddedclusterisog/210711_205628/0000/output_6.root',
                loc+'mc/GJet_Pt-20toInf_DoubleEMEnriched_MGG-40to80_TuneCP5_14TeV_Pythia8/crab_GJet_Pt-20toInf_DoubleEMEnriched_MGG-40to80_TuneCP5_14TeV_Pythia8_July2021newflaaddedclusterisog/210711_205628/0000/output_7.root',
                loc+'mc/GJet_Pt-20toInf_DoubleEMEnriched_MGG-40to80_TuneCP5_14TeV_Pythia8/crab_GJet_Pt-20toInf_DoubleEMEnriched_MGG-40to80_TuneCP5_14TeV_Pythia8_July2021newflaaddedclusterisog/210711_205628/0000/output_8.root',
                loc+'mc/GJet_Pt-20toInf_DoubleEMEnriched_MGG-40to80_TuneCP5_14TeV_Pythia8/crab_GJet_Pt-20toInf_DoubleEMEnriched_MGG-40to80_TuneCP5_14TeV_Pythia8_July2021newflaaddedclusterisog/210711_205628/0000/output_9.root',
                loc+'mc/GJet_Pt-40toInf_DoubleEMEnriched_MGG-80toInf_TuneCP5_14TeV_Pythia8/crab_GJet_Pt-40toInf_DoubleEMEnriched_MGG-80toInf_TuneCP5_14TeV_Pythia8_July2021newflaaddedclusterisog/210711_205634/0000/output_1.root',
                loc+'mc/GJet_Pt-40toInf_DoubleEMEnriched_MGG-80toInf_TuneCP5_14TeV_Pythia8/crab_GJet_Pt-40toInf_DoubleEMEnriched_MGG-80toInf_TuneCP5_14TeV_Pythia8_July2021newflaaddedclusterisog/210711_205634/0000/output_10.root',
                loc+'mc/GJet_Pt-40toInf_DoubleEMEnriched_MGG-80toInf_TuneCP5_14TeV_Pythia8/crab_GJet_Pt-40toInf_DoubleEMEnriched_MGG-80toInf_TuneCP5_14TeV_Pythia8_July2021newflaaddedclusterisog/210711_205634/0000/output_11.root',
                loc+'mc/GJet_Pt-40toInf_DoubleEMEnriched_MGG-80toInf_TuneCP5_14TeV_Pythia8/crab_GJet_Pt-40toInf_DoubleEMEnriched_MGG-80toInf_TuneCP5_14TeV_Pythia8_July2021newflaaddedclusterisog/210711_205634/0000/output_12.root',
                loc+'mc/GJet_Pt-40toInf_DoubleEMEnriched_MGG-80toInf_TuneCP5_14TeV_Pythia8/crab_GJet_Pt-40toInf_DoubleEMEnriched_MGG-80toInf_TuneCP5_14TeV_Pythia8_July2021newflaaddedclusterisog/210711_205634/0000/output_13.root',
                loc+'mc/GJet_Pt-40toInf_DoubleEMEnriched_MGG-80toInf_TuneCP5_14TeV_Pythia8/crab_GJet_Pt-40toInf_DoubleEMEnriched_MGG-80toInf_TuneCP5_14TeV_Pythia8_July2021newflaaddedclusterisog/210711_205634/0000/output_2.root',
                loc+'mc/GJet_Pt-40toInf_DoubleEMEnriched_MGG-80toInf_TuneCP5_14TeV_Pythia8/crab_GJet_Pt-40toInf_DoubleEMEnriched_MGG-80toInf_TuneCP5_14TeV_Pythia8_July2021newflaaddedclusterisog/210711_205634/0000/output_3.root',
                loc+'mc/GJet_Pt-40toInf_DoubleEMEnriched_MGG-80toInf_TuneCP5_14TeV_Pythia8/crab_GJet_Pt-40toInf_DoubleEMEnriched_MGG-80toInf_TuneCP5_14TeV_Pythia8_July2021newflaaddedclusterisog/210711_205634/0000/output_4.root',
                loc+'mc/GJet_Pt-40toInf_DoubleEMEnriched_MGG-80toInf_TuneCP5_14TeV_Pythia8/crab_GJet_Pt-40toInf_DoubleEMEnriched_MGG-80toInf_TuneCP5_14TeV_Pythia8_July2021newflaaddedclusterisog/210711_205634/0000/output_5.root',
                loc+'mc/GJet_Pt-40toInf_DoubleEMEnriched_MGG-80toInf_TuneCP5_14TeV_Pythia8/crab_GJet_Pt-40toInf_DoubleEMEnriched_MGG-80toInf_TuneCP5_14TeV_Pythia8_July2021newflaaddedclusterisog/210711_205634/0000/output_6.root',
                loc+'mc/GJet_Pt-40toInf_DoubleEMEnriched_MGG-80toInf_TuneCP5_14TeV_Pythia8/crab_GJet_Pt-40toInf_DoubleEMEnriched_MGG-80toInf_TuneCP5_14TeV_Pythia8_July2021newflaaddedclusterisog/210711_205634/0000/output_7.root',
                loc+'mc/GJet_Pt-40toInf_DoubleEMEnriched_MGG-80toInf_TuneCP5_14TeV_Pythia8/crab_GJet_Pt-40toInf_DoubleEMEnriched_MGG-80toInf_TuneCP5_14TeV_Pythia8_July2021newflaaddedclusterisog/210711_205634/0000/output_8.root',
                loc+'mc/GJet_Pt-40toInf_DoubleEMEnriched_MGG-80toInf_TuneCP5_14TeV_Pythia8/crab_GJet_Pt-40toInf_DoubleEMEnriched_MGG-80toInf_TuneCP5_14TeV_Pythia8_July2021newflaaddedclusterisog/210711_205634/0000/output_9.root',],
        'xsecwt': 1, #xsec wt if any, if none then it can be 1
        'selection':CommonSel+' & '+PhoSel, #selection for background
    },
]

#####################################################################
'''
def modfiydf(df):#Do not remove this function, even if empty
    #Can be used to add new branches (The pandas dataframe style)
    
    ############ Write you modifications inside this block #######
    #example:
    #df["Electron_SCeta"]=df["Electron_deltaEtaSC"] + df["Electron_eta"]
    
    ####################################################
    
    return 0
'''
#####################################################################

Tree = "ntuplizer/tree"

#MVAs to use as a list of dictionaries
MVAs = [
    #can add as many as you like: For MVAtypes XGB and DNN are keywords, so names can be XGB_new, DNN_old etc. 
    #But keep XGB and DNN in the names (That is how the framework identifies which algo to run
    
    {"MVAtype":"DNN_rechitiso_2drwt",
     "Color":"blue", #Plot color for MVA
     "Label":"DNN_rechitiso_2drwt", # label can be anything (this is how you will identify them on plot legends)
     "features":["ele_fbrem", "ele_deltaetain", "ele_deltaphiin", "ele_oldsigmaietaieta", 
                 "ele_oldhe", "ele_ep", "ele_olde15", "ele_eelepout",
                 "ele_kfchi2", "ele_kfhits", "ele_expected_inner_hits","ele_dr03TkSumPt",
                 "ele_dr03EcalRecHitSumEt","ele_dr03HcalTowerSumEt","ele_gsfchi2",#"scl_eta","ele_pt",
                 #"ele_ecalPFClusterIso","ele_hcalPFClusterIso",
                 #'ele_conversionVertexFitProbability',
                 'ele_nbrem','ele_deltaetaseed','ele_hadronicOverEm','ele_olde25max','ele_olde55'],#Input features #Should be branchs in your dataframe
     "feature_bins":[100 for i in range(20)],#same length as features
     #Binning used only for plotting features (should be in the same order as features), does not affect training
     'Scaler':"MinMaxScaler",
     "DNNDict":{'epochs':1000, 'batchsize':5000, 'lr':0.001, 
                #The other parameters which are not here, can be modified in Trainer script
                'model': Sequential([Dense(24, kernel_initializer='glorot_normal', activation='relu'),
                                     Dense(48, activation="relu"),
                                     Dense(24, activation="relu"),
                                     Dropout(0.1),
                                     Dense(len(Classes),activation="softmax")]),
                'compile':{'loss':'categorical_crossentropy','optimizer':Adam(lr=0.001), 'metrics':['accuracy']},
                'earlyStopping': EarlyStopping(monitor='val_loss', mode='min', verbose=1, patience=10)
                #check the modelDNN1 function above, you can also create your own
               }
    },

    {"MVAtype":"DNN_rechitiso_2drwt_withconvvars",
     "Color":"blue", #Plot color for MVA
     "Label":"DNN_rechitiso_2drwt_withconvvars", # label can be anything (this is how you will identify them on plot legends)
     "features":["ele_fbrem", "ele_deltaetain", "ele_deltaphiin", "ele_oldsigmaietaieta", 
                 "ele_oldhe", "ele_ep", "ele_olde15", "ele_eelepout",
                 "ele_kfchi2", "ele_kfhits", "ele_expected_inner_hits","ele_dr03TkSumPt",
                 "ele_dr03EcalRecHitSumEt","ele_dr03HcalTowerSumEt","ele_gsfchi2",#"scl_eta","ele_pt",
                 #"ele_ecalPFClusterIso","ele_hcalPFClusterIso",
                 #'ele_conversionVertexFitProbability',
                 "ele_convDist","ele_convDcot",
                 'ele_nbrem','ele_deltaetaseed','ele_hadronicOverEm','ele_olde25max','ele_olde55'],#Input features #Should be branchs in your dataframe
     "feature_bins":[100 for i in range(22)],#same length as features
     #Binning used only for plotting features (should be in the same order as features), does not affect training
     'Scaler':"MinMaxScaler",
     "DNNDict":{'epochs':1000, 'batchsize':5000, 'lr':0.001, 
                #The other parameters which are not here, can be modified in Trainer script
                'model': Sequential([Dense(24, kernel_initializer='glorot_normal', activation='relu'),
                                     Dense(48, activation="relu"),
                                     Dense(24, activation="relu"),
                                     Dropout(0.1),
                                     Dense(len(Classes),activation="softmax")]),
                'compile':{'loss':'categorical_crossentropy','optimizer':Adam(lr=0.001), 'metrics':['accuracy']},
                'earlyStopping': EarlyStopping(monitor='val_loss', mode='min', verbose=1, patience=10)
                #check the modelDNN1 function above, you can also create your own
               }
    },
    
    {"MVAtype":"DNN_rechitiso_2drwt_withpteta",
     "Color":"green", #Plot color for MVA
     "Label":"DNN_rechitiso_2drwt_withpteta", # label can be anything (this is how you will identify them on plot legends)
     "features":["ele_fbrem", "ele_deltaetain", "ele_deltaphiin", "ele_oldsigmaietaieta", 
                 "ele_oldhe", "ele_ep", "ele_olde15", "ele_eelepout",
                 "ele_kfchi2", "ele_kfhits", "ele_expected_inner_hits","ele_dr03TkSumPt",
                 "ele_dr03EcalRecHitSumEt","ele_dr03HcalTowerSumEt","ele_gsfchi2","scl_eta","ele_pt",
                 #"ele_ecalPFClusterIso","ele_hcalPFClusterIso",
                 #'ele_conversionVertexFitProbability',
                 'ele_nbrem','ele_deltaetaseed','ele_hadronicOverEm','ele_olde25max','ele_olde55'],#Input features #Should be branchs in your dataframe
     "feature_bins":[100 for i in range(22)],#same length as features
     #Binning used only for plotting features (should be in the same order as features), does not affect training
     'Scaler':"MinMaxScaler",
     "DNNDict":{'epochs':1000, 'batchsize':5000, 'lr':0.001, 
                #The other parameters which are not here, can be modified in Trainer script
                'model': Sequential([Dense(24, kernel_initializer='glorot_normal', activation='relu'),
                                     Dense(48, activation="relu"),
                                     Dense(24, activation="relu"),
                                     Dropout(0.1),
                                     Dense(len(Classes),activation="softmax")]),
                'compile':{'loss':'categorical_crossentropy','optimizer':Adam(lr=0.001), 'metrics':['accuracy']},
                'earlyStopping': EarlyStopping(monitor='val_loss', mode='min', verbose=1, patience=10)
                #check the modelDNN1 function above, you can also create your own
               }
    },

    
    {"MVAtype":"DNN_rechitandclusteriso_2drwt_withpteta",
     "Color":"green", #Plot color for MVA
     "Label":"DNN_rechitandclusteriso_2drwt_withpteta", # label can be anything (this is how you will identify them on plot legends)
     "features":["ele_fbrem", "ele_deltaetain", "ele_deltaphiin", "ele_oldsigmaietaieta", 
                 "ele_oldhe", "ele_ep", "ele_olde15", "ele_eelepout",
                 "ele_kfchi2", "ele_kfhits", "ele_expected_inner_hits","ele_dr03TkSumPt",
                 "ele_dr03EcalRecHitSumEt","ele_dr03HcalTowerSumEt","ele_gsfchi2","scl_eta","ele_pt",
                 "ele_ecalPFClusterIso","ele_hcalPFClusterIso",
                 #'ele_conversionVertexFitProbability',
                 'ele_nbrem','ele_deltaetaseed','ele_hadronicOverEm','ele_olde25max','ele_olde55'],#Input features #Should be branchs in your dataframe
     "feature_bins":[100 for i in range(24)],#same length as features
     #Binning used only for plotting features (should be in the same order as features), does not affect training
     'Scaler':"MinMaxScaler",
     "DNNDict":{'epochs':1000, 'batchsize':5000, 'lr':0.001, 
                #The other parameters which are not here, can be modified in Trainer script
                'model': Sequential([Dense(24, kernel_initializer='glorot_normal', activation='relu'),
                                     Dense(48, activation="relu"),
                                     Dense(24, activation="relu"),
                                     Dropout(0.1),
                                     Dense(len(Classes),activation="softmax")]),
                'compile':{'loss':'categorical_crossentropy','optimizer':Adam(lr=0.001), 'metrics':['accuracy']},
                'earlyStopping': EarlyStopping(monitor='val_loss', mode='min', verbose=1, patience=10)
                #check the modelDNN1 function above, you can also create your own
               }
    },

        
    {"MVAtype":"DNN_rechitandclusteriso_2drwt",
     "Color":"red", #Plot color for MVA
     "Label":"DNN_rechitandclusteriso_2drwt", # label can be anything (this is how you will identify them on plot legends)
     "features":["ele_fbrem", "ele_deltaetain", "ele_deltaphiin", "ele_oldsigmaietaieta", 
                 "ele_oldhe", "ele_ep", "ele_olde15", "ele_eelepout",
                 "ele_kfchi2", "ele_kfhits", "ele_expected_inner_hits","ele_dr03TkSumPt",
                 "ele_dr03EcalRecHitSumEt","ele_dr03HcalTowerSumEt","ele_gsfchi2",#"scl_eta","ele_pt",
                 "ele_ecalPFClusterIso","ele_hcalPFClusterIso",
                 #'ele_conversionVertexFitProbability',
                 'ele_nbrem','ele_deltaetaseed','ele_hadronicOverEm','ele_olde25max','ele_olde55'],#Input features #Should be branchs in your dataframe
     "feature_bins":[100 for i in range(22)],#same length as features
     #Binning used only for plotting features (should be in the same order as features), does not affect training
     'Scaler':"MinMaxScaler",
     "DNNDict":{'epochs':1000, 'batchsize':5000, 'lr':0.001, 
                #The other parameters which are not here, can be modified in Trainer script
                'model': Sequential([Dense(24, kernel_initializer='glorot_normal', activation='relu'),
                                     Dense(48, activation="relu"),
                                     Dense(24, activation="relu"),
                                     Dropout(0.1),
                                     Dense(len(Classes),activation="softmax")]),
                'compile':{'loss':'categorical_crossentropy','optimizer':Adam(lr=0.001), 'metrics':['accuracy']},
                'earlyStopping': EarlyStopping(monitor='val_loss', mode='min', verbose=1, patience=10)
                #check the modelDNN1 function above, you can also create your own
               }
    },

    
    {"MVAtype":"DNN_clusteriso_2drwt",
     "Color":"black", #Plot color for MVA
     "Label":"DNN_clusteriso_2drwt", # label can be anything (this is how you will identify them on plot legends)
     "features":["ele_fbrem", "ele_deltaetain", "ele_deltaphiin", "ele_oldsigmaietaieta", 
                 "ele_oldhe", "ele_ep", "ele_olde15", "ele_eelepout",
                 "ele_kfchi2", "ele_kfhits", "ele_expected_inner_hits","ele_dr03TkSumPt",
                 #"ele_dr03EcalRecHitSumEt","ele_dr03HcalTowerSumEt",
                 "ele_gsfchi2",
                 #"scl_eta","ele_pt",
                 "ele_ecalPFClusterIso","ele_hcalPFClusterIso",
                 #'ele_conversionVertexFitProbability',
                 'ele_nbrem','ele_deltaetaseed','ele_hadronicOverEm','ele_olde25max','ele_olde55'],#Input features #Should be branchs in your dataframe
     "feature_bins":[100 for i in range(20)],#same length as features
     #Binning used only for plotting features (should be in the same order as features), does not affect training
     'Scaler':"MinMaxScaler",
     "DNNDict":{'epochs':1000, 'batchsize':5000, 'lr':0.001, 
                #The other parameters which are not here, can be modified in Trainer script
                'model': Sequential([Dense(24, kernel_initializer='glorot_normal', activation='relu'),
                                     Dense(48, activation="relu"),
                                     Dense(24, activation="relu"),
                                     Dropout(0.1),
                                     Dense(len(Classes),activation="softmax")]),
                'compile':{'loss':'categorical_crossentropy','optimizer':Adam(lr=0.001), 'metrics':['accuracy']},
                'earlyStopping': EarlyStopping(monitor='val_loss', mode='min', verbose=1, patience=10)
                #check the modelDNN1 function above, you can also create your own
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
OverlayWP=['passElectronSelection']
OverlayWPColors = ["black"] #Colors on plots for WPs

#To print thresholds of mva scores for corresponding signal efficiency
SigEffWPs=["95%","98%"] # Example for 80% and 90% Signal Efficiency Working Points
######### 


#####Optional Features

RandomState=42
#Choose the same number everytime for reproducibility

#MVAlogplot=False
#If true, MVA outputs are plotted in log scale

#Multicore=False
#If True all CPU cores available are used XGB 

testsize=0.2
#(0.2 means 20%) (How much data to use for testing)

#flatten=False
#For NanoAOD and other un-flattened trees, you can switch on this option to flatten branches with variable length for each event (Event level -> Object level)
#You can't flatten branches which have different length for the same events. For example: It is not possible to flatten electron and muon branches both at the same time, since each event could have different electrons vs muons. Branches that have only one value for each event, line Nelectrons, can certainly be read along with unflattened branches.

