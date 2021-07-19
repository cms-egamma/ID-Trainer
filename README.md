# ID-Trainer

> A simple config-based tool for high-energy-physics machine learning tasks.


Currently supports:
* **Binary-classification** (currently using XGBoost and DNN)
Examples: DY vs ttbar, DY prompt vs DY fake
* **Multi-sample classification** (currently using XGBoost and DNN)
Examples: DY vs (ttbar and QCD)
* **Multi-class classification** (currently using XGBoost and DNN)
Examples: DY vs ttbar vs QCD


Salient features:
1) Parallel reading of root files (using DASK)
2) Runs on flat ntuples (even NanoAODs) out of the box
3) Adding multiple MVAs is very trivial (Subject to available computing power)
4) Cross-section and pt-eta reweighting can be handled together
5) Multi-Sample training possible
6) Multi-Class training possible
7) Ability to customize thresholds

### Setting up

#### Clone
```
git clone https://github.com/cms-egamma/ID-Trainer.git
```
#### Setup
In principle, you can set this up on your local computer by installing packages via conda/pip, but when possible please set up a cvmfs release.

#### Run on CPUs only

Use LCG 97python3 and you will have all the dependencies! (Tested at lxplus and SWAN)
`source /cvmfs/sft.cern.ch/lcg/views/LCG_97python3/x86_64-centos7-gcc8-opt/setup.sh`

#### Run on CPUs and GPUs

The code can also transparently use a GPU, if a GPU card is available. The cvmfs release to use in that case is:
`source /cvmfs/sft.cern.ch/lcg/views/LCG_97py3cu10/x86_64-centos7-gcc7-opt/setup.sh`


### Running the trainer

#### Create a config
Create a new python config. Some sample python configs are available in the 'Configs' folder. They cover the most possible examples. All you need to do is to edit the config with the settings for your analysis and then run:

```
python Trainer.py NewTrainConfig #without the .py
```

The Trainer will read the settings from the config file and run training

Projects where the framework has been helpful

1) Run-3 Ele MVA ID
2) Close photon analysis
3) H->eeg analysis

##########################################

### The different parts of the config

#### Imports
This is needed to use numpy and tensorflow. You can leave it as is.
```python
import numpy as np
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Activation
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.layers import Dropout
from tensorflow.keras.callbacks import EarlyStopping
```



#### All the Parameters

| Parameters |Type| Description|
| --------------- | ----------------| ---------------- |
| `OutputDirName` |string| All plots, models, config file will be stored in this directory. This will be automatically created. If it already exists, it will overwrite everything if you run it again with the same `OutputDirName`|
| `branches` |list of strings| Branches to read (Should be in the input root files). Only these branches can be later used for any purpose. The '\*' is useful for selecting pattern-based branches. In principle one can do ``` branches=["*"] ```, but remember that the data loading time increases, if you select more branches|
|`SaveDataFrameCSV`|boolean| If True, this will save the data frame as a parquet file and the next time you run the same training with different parameters, it will be much faster|
|`loadfromsaved`|boolean| If root files and branches are the same as previous training and SaveDataFrameCSV was True, you can assign this as `True`, and data loading time will reduce significantly. Remember that this will use the same output directory as mentioned using `OutputDirName`, so the data frame should be present there|
|`Classes` | list of strings | Two or more classes possible. For two classes the code will do a binary classification. For more than two classes Can be anything but samples will be later loaded under this scheme. |
|`ClassColors`|list of strings|Colors for `Classes` to use in plots. Standard python colors work!|
|`Tree`| string |Location of the tree inside the root file|
|`processes`| list of dictionaries| You can add as many process files as you like and assign them to a specific class. For example WZ.root and TTBar.root could be 'Background' class and DY.root could be 'Signal' or both 'Signal and 'background' can come from the same root file. In fact you can have, as an example: 4 classes and 5 root files. The Trainer will take care of it at the backend. Look at the sample  config below to see how processes are added. It is a list of dictionaries, with one example dictionary looking like this ` {'Class':'IsolatedSignal','path':['./DY.root','./Zee.root'], 'xsecwt': 1, 'selection':'(ele_pt > 5) & (abs(scl_eta) < 1.442) & (abs(scl_eta) < 2.5) & (matchedToGenEle==1)'} ` |
|`MVAs`|list of dictionaries| MVAs to use. You can add as many as you like: MVAtypes XGB and DNN are keywords, so names can be XGB_new, DNN_old etc, but keep XGB and DNN in the names (That is how the framework identifies which algo to run). Look at the sample config below to see how MVAs are added. |

#### Optional Parameters

| Parameters          |Type| Description| Default value|
| --------------- | ----------------| ---------------- | ---------------- |
|`Reweighing`|boolean| This is independent of xsec reweighing (this reweighing will be done after taking into account xsec weight of multiple samples). Even if this is 'False', xsec reweighting will always be done. To switch off xsec reweighting, you can just assign the xsec weight is `1`| False |
|`ptbins`,`etabins`|lists of numbers| $p_T$ and $\eta$ bins of interest (will be used for robustness studies: function coming soon) and will also be used for 2D $p_T$-$\eta$ reweighing if the `Reweighing` option is `True`|Not activated until Reweighing==True |
|`ptwtvar`,`etawtvar`|strings| names of $p_T$ and $\eta$ branches|Not activated until Reweighing==True|
|`WhichClassToReweightTo`|string|  2D $p_T$-$\eta$ spectrum of all other classes will be reweighted to this class|Not activated until Reweighing==True|
|`OverlayWP`|list of strings| Working Point Flags to compare to (Should be in your ntuple and should also be read in branches)|empty list|
|`OverlayWPColors`|list of strings| Working Point Flags colors in plot|empty list|
|`SigEffWPs`| list of strings | To print thresholds of mva scores for corresponding signal efficiency, example `["80%","90%"]` (Add as many as you would like) |empty list|
|`testsize`|float| In fraction, how much data to use for testing (0.3 means 30%)| 0.2|
|`flatten`       |boolean| For NanoAOD and other un-flattened trees, you can assign this as `True` to flatten branches with variable length for each event (Event level -> Object level)| False |
| `Debug`         |boolean| If True, only a small subset of events/objects are used for either Signal or background. Useful for quick debugging of code| False |
|`RandomState`|integer |Choose the same number every time for reproducibility| 42|
|`MVAlogplot`|boolean| If true, MVA outputs are plotted in log scale| False|
|`Multicore`|boolean| If True all CPU cores available are used XGB | True|

#### How to add variables? or modify the ones that are in tree

| Function         |Type| Description| Default value|
| --------------- | ----------------| ---------------- | ---------------- |
|`modifydf`|function| In your config, you can add a function with this exact name `modifydf` which accepts a pandas dataframe and manipulates it and then returns 0. Using this you can add new variables or modify already present variables. Example: `def modifydf(df): df['A']=df[X]+df[Y]; return 0;` This will add a new branch named 'A'.| Not activated until defined|


### A sample config for running XGboost and DNN together

```python


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
OutputDirName = 'SimpleBinaryClassification' #All plots, models, config file will be stored here
Debug=False # If True, only a small subset of events/objects are used for either Signal or background #Useful for quick debugging

#Branches to read #Should be in the root files #Only the read branches can be later used for any purpose
branches=["scl_eta","ele*","matched*","EleMVACats",'passElectronSelection','Fall*']

SaveDataFrameCSV,loadfromsaved=True,False #If loadfromsaved=True, dataframe stored in OutputDirName will be read

Classes,ClassColors = ['IsolatedSignal','NonIsolated'],['#377eb8', '#ff7f00']

processes=[
    {'Class':'IsolatedSignal','path':['./DY.root','./Zee.root'],
     #Can be a single root file, a list of root file, or even a folder but in a tuple format (folder,fileextension), like ('./samples','.root')
     'xsecwt': 1, #can be a number or a branch name, like 'weight' #Will go into training
     'selection':'(ele_pt > 5) & (abs(scl_eta) < 1.442) & (abs(scl_eta) < 2.5) & (matchedToGenEle==1)', #selection for background
    },
    {'Class':'NonIsolated','path':['./QCD.root'],
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
                 "ele_oldhe", "ele_ep", "ele_olde15", "ele_eelepout"],
     "feature_bins":[100 , 100, 100, 100, 100, 100, 100, 100], #same length as features
     #Binning used only for plotting features (should be in the same order as features), does not affect training
     'Scaler':"MinMaxScaler", #Scaling for features before passing to the model training
     'UseGPU':True, #If you have a GPU card, you can turn on this option (CUDA 10.0, Compute Capability 3.5 required)
     "XGBGridSearch":{'min_child_weight': [5], 'max_depth': [2,3,4]} ## multiple values for a parameter will automatically do a grid search
     #All standard XGB parameters supported
    },

     {"MVAtype":"DNN_clusteriso_2drwt",#Keyword to identify MVA method.
     "Color":"black", #Plot color for MVA
     "Label":"DNN_clusteriso_2drwt", # label can be anything (this is how you will identify them on plot legends)
     "features":["ele_fbrem", "ele_deltaetain", "ele_deltaphiin", "ele_oldsigmaietaieta",
                 "ele_oldhe", "ele_ep", "ele_olde15", "ele_eelepout",
                 "ele_kfchi2", "ele_kfhits", "ele_expected_inner_hits","ele_dr03TkSumPt",
                 "ele_dr03EcalRecHitSumEt","ele_dr03HcalTowerSumEt","ele_gsfchi2","scl_eta","ele_pt",
                 'ele_nbrem','ele_deltaetaseed','ele_hadronicOverEm','ele_olde25max','ele_olde55'],
     "feature_bins":[100 for i in range(22)], #same length as features
     'Scaler':"MinMaxScaler", #Scaling for features before passing to the model training
     "DNNDict":{'epochs':10, 'batchsize':5000,
                'model': Sequential([Dense(24, kernel_initializer='glorot_normal', activation='relu'),
                                     Dense(48, activation="relu"),
                                     Dense(24, activation="relu"),
                                     Dropout(0.1),
                                     Dense(len(Classes),activation="softmax")]),
                'compile':{'loss':'categorical_crossentropy','optimizer':Adam(lr=0.001), 'metrics':['accuracy']},
                'earlyStopping': EarlyStopping(monitor='val_loss', mode='min', verbose=1, patience=10)
               }
    },
]

```