# ID-Trainer

> A simple config-based tool for ID-Training as well as other high-energy-physics machine learning tasks.

Currently supports:
1) Binary-classification: XGBoost and DNN
Examples: DY vs ttbar, DY prompt vs DY fake
3) Multi-sample classification: XGBoost and DNN 
Examples: DY vs (ttbar and QCD)
5) Multi-class classification: DNN (experimental feature)
Examples: DY vs ttbar vs QCD




### Setting in up

#### Clone
```
git clone https://github.com/cms-egamma/ID-Trainer.git
```
#### Setup
In principle, you can set this up on your local computer by installing packages via conda/pip, but if it is possible to set up a cvmfs release, you can do that using LCG 97python3 and you will have all the dependencies! (Tested at lxplus and SWAN)

`source /cvmfs/sft.cern.ch/lcg/views/LCG_97python3/x86_64-centos7-gcc8-opt/setup.sh`

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
|`Reweighing`|boolean| This is independent of xsec reweighing (this reweighing will be done after taking into account xsec weight of multiple samples). Even if this is 'False', xsec reweighting will always be done. To switch off xsec reweighting, you can just assign the xsec weight is `1`|
|`ptbins`,`etabins`|lists of numbers| $p_T$ and $\eta$ bins of interest (will be used for robustness studies: function coming soon) and will also be used for 2D $p_T$-$\eta$ reweighing if the `Reweighing` option is `True`|
|`ptwtvar`,`etawtvar`|strings| names of $p_T$ and $\eta$ branches|
|`Classes` | list of strings | Two or more classes possible. For two classes the code will do a binary classification. For more than two classes Can be anything but samples will be later loaded under this scheme. |
|`WhichClassToReweightTo`|string|  2D $p_T$-$\eta$ spectrum of all other classes will be reweighted to this class|
|`ClassColors`|list of strings|Colors for `Classes` to use in plots. Standard python colors work!|
|`Tree`| string |Location of the tree inside the root file|
|`OverlayWP`|list of strings| Working Point Flags to compare to (Should be in your ntuple and should also be read in branches)|
|`OverlayWPColors`|list of strings| Working Point Flags colors in plot|
|`SigEffWPs`| list of strings | To print thresholds of mva scores for corresponding signal efficiency, example `["80%","90%"]` (Add as many as you would like) Currently not supported for multi-class classification but fully supported for Binary-classification |
|`processes`| list of dictionaries| You can add as many process files as you like and assign them to a specific class. For example WZ.root and TTBar.root could be 'Background' class and DY.root could be 'Signal' or both 'Signal and 'background' can come from the same root file. In fact you can have, as an example: 4 classes and 5 root files. The Trainer will take care of it at the backend. Look at the same config below to see how processes are added. |
|`MVAs`|list of dictionaries| MVAs to use. You can add as many as you like: MVAtypes XGB and DNN are keywords, so names can be XGB_new, DNN_old etc, but keep XGB and DNN in the names (That is how the framework identifies which algo to run). Look at the same config below to see how MVAs are added. |

#### Optional Parameters

| Parameters          |Type| Description| Default value|
| --------------- | ----------------| ---------------- | ---------------- |
|`testsize`|float| In fraction, how much data to use for testing (0.3 means 30%)| 0.2
|`flatten`       |boolean| For NanoAOD and other un-flattened trees, you can assign this as `True` to flatten branches with variable length for each event (Event level -> Object level)| False |
| `Debug`         |boolean| If True, only a small subset of events/objects are used for either Signal or background. Useful for quick debugging of code| False |
|`RandomState`|integer |Choose the same number every time for reproducibility| 42|
|`MVAlogplot`|boolean| If true, MVA outputs are plotted in log scale| False|
|`Multicore`|boolean| If True all CPU cores available are used XGB | True|

#### A sample config

```python
OutputDirName = 'TwoClassOutput'
branches=["ele_*","scl_eta","rho","matchedToGenEle","Fall17isoV2wp80", "Fall17isoV2wp90"]
SaveDataFrameCSV,loadfromsaved=True,False

ptbins = [10,30,40,50,100,5000]
etabins = [-1.6,-1.2,-0.8,-0.5,0.0,0.5,0.8,1.2,1.6]
ptwtvar,etawtvar='ele_pt','scl_eta'
Reweighing,WhichClassToReweightTo = 'True','Signal'
Classes,ClassColors = ['Background','Signal'],['#377eb8', '#ff7f00']

processes=[
    {'Class':'Background','path':'./QCD.root',
     'xsecwt': 1, #xsec wt if any. Can be a branch name or number
     'selection':"(ele_pt>10) & (abs(scl_eta)<1.566) & (matchedToGenEle == 0)", #selection for background
    },
    
    {'Class':'Background','path':'./GJet.root',
     'xsecwt': "weight", #xsec wt if any. Can be a branch name or number
     'selection':"(ele_pt>10) & (abs(scl_eta)<1.566) & (matchedToGenEle == 0)", #selection for background
    },

    {'Class':'Signal','path':'./DY.root', 
     'xsecwt': 1, #xsec wt if any. Can be a branch name or number
     'selection':"(ele_pt>10) & (abs(scl_eta)<1.566) & (matchedToGenEle == 1)", #selection for signal
    }
]

Tree = "ntuplizer/tree"
OverlayWP,OverlayWPColors =['Fall17isoV2wp80', 'Fall17isoV2wp90'],["black","purple"]
SigEffWPs=["80%","90%"]

MVAs = [
    
    {"MVAtype":"XGB_1",
     "Color":"green", #Plot color for MVA
     "Label":"XGB try 1", # label can be anything (this is how you will identify them on plot legends)
     "features":["ele_oldsigmaietaieta", "ele_oldsigmaiphiiphi", "ele_oldcircularity", "ele_oldr9"],
     "feature_bins":[100, 100, np.linspace(0.0,0.1,1001), 100], #only for plotting features (should be in the same order as features)
     'Scaler':"MinMaxScaler",
     "XGBGridSearch":{'min_child_weight': [5,2,1], 'gamma': [0.4,0.1], 'subsample': [0.6], 'colsample_bytree': [1.0], 'max_depth': [4,3]} #All standard XGB parameters supported
     #More than one element for any parameters will result in a grid search. example 'gamma': [0.4,0.1] has two grid poits in gamma. To not a grid search in this parameter, just do something like 'gamma': [0.4], which will fix the value at 0.4.
    },

    {"MVAtype":"DNN_1",
     "Color":"blue", #Plot color for MVA
     "Label":"DNN try 1", # label can be anything (this is how you will identify them on plot legends)
     "features":["ele_oldsigmaietaieta", "ele_oldsigmaiphiiphi", "ele_oldcircularity", "ele_oldr9"], #Input features #Should be branchs in your dataframe
     "feature_bins":[100, 100, np.linspace(0.0,0.1,1001), 100], #only for plotting features (should be in the same order as features)
     'Scaler':"MinMaxScaler",
     "DNNDict":{'epochs':100, 'batchsize':100, 'lr':0.001,
                #The other parameters which are not here, can be modified in Trainer script
                'model': Sequential([Dense(48, kernel_initializer='glorot_normal', activation='relu'),
                                     Dense(12, activation="relu"),
                                     Dropout(0.1),
                                     Dense(len(Classes),activation="softmax")]),
                'compile':{'loss':'binary_crossentropy','optimizer':Adam(lr=0.001), 'metrics':['accuracy']},
                'earlyStopping': EarlyStopping(monitor='val_loss', mode='min', verbose=1, patience=10)
               }
    },
]


# ID-Trainer

> A simple config based tool for ID-Training as well as other high-energy-physics machine learning tasks.

Currently supports:
1) Binary-classification : XGBoost and DNN
Examples: DY vs ttbar, DY prompt vs DY fake
3) Multi-sample classification : XGBoost and DNN 
Examples: DY vs (ttbar and QCD)
5) Multi-class classification : DNN (experimental feature)
Examples: DY vs ttbar vs QCD




### Setting in up

#### Clone
```
git clone https://github.com/cms-egamma/ID-Trainer.git
```
#### Setup
In principle you can set this up on you local computer by installing packages via conda/pip, but if it is possible to setup a cvmfs release, you can do that using LCG 97python3 and you will have all the dependencies! (Tested at lxplus and SWAN)

`source /cvmfs/sft.cern.ch/lcg/views/LCG_97python3/x86_64-centos7-gcc8-opt/setup.sh`

### Running the trainer
#### Create a config
Create a new python config. Some sample python configs are available in the 'Configs' folder. They cover most possible examples. All you need to do is to edit the config with the settings for your analysis and then run:

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

| Parameters          |Type| Description|
| --------------- | ----------------| ---------------- |
| `OutputDirName` |string| All plots, models, config file will be stored in this directory. This will be automatically created. If it already exists, it will overwrite everything if you run it again with the same `OutputDirName`|
| `branches`      |list of strings| Branches to read (Should be in the input root files). Only these branches can be later used for any purpose. The '\*' is useful for selecting pattern based branches. In principle one can do ``` branches=["*"] ```, but remember that the data loading time increases, if you select more branches|
|`SaveDataFrameCSV`|boolean| If True, this will save the dataframe as a parquet file and the next time you run the same training with different parameters, it will be much faster|
|`loadfromsaved`|boolean| If root files and branches are same as a previous training and SaveDataFrameCSV was True, you can assign this as `True` and data loading time will reduce significantly. Remember that this will use the same output directory as mentioned using `OutputDirName`, so the dataframe should be present there|
|`Reweighing`|boolean| This is independent of xsec reweighing (this reweighing will be done after taking into account xsec weight of multiple samples). Even if this is 'False', xsec reweighting will always be done. To switch off xsec reweighting, you can just assign the xsec weight is `1`|
|`ptbins`,`etabins`|lists of numbers| $p_T$ and $\eta$ bins of interest (will be used for robustness studies: function coming soon) and will also be used for 2D $p_T$-$\eta$ reweighing if the `Reweighing` option is `True`|
|`ptwtvar`,`etawtvar`|strings| names of $p_T$ and $\eta$ branches|
|`Classes` | list of strings | Two or more classes possible. For two classes the code will do a binary classification. For more than two classes Can be anything but samples will be later loaded under this scheme. |
|`WhichClassToReweightTo`|string|  2D $p_T$-$\eta$ spectrum of all other classs will be reweighted to this class|
|`ClassColors`|list of strings|Colors for `Classes` to use in plots. Standard python colors work!|
|`Tree`| string |Location of tree inside the root file|
|`OverlayWP`|list of strings| Working Point Flags to compare to (Should be in your ntuple and should also be read in branches)
|`OverlayWPColors`|list of strings| Working Point Flags colors in plot|
|`SigEffWPs`| list of strings | To print thresholds of mva scores for corresponding signal efficiency, example `["80%","90%"]` (Add as many as you would like) Currently not supported for multi-class classification but fully supported for Binary-classification |
|`processes`| list of dictionaries| You can add as many process files as you like and assign them to a specific class. For example WZ.root and TTBar.root could be 'Background' class and DY.root could be 'Signal' or both 'Signal and 'background' can come from the same root file. Infact you can have, as an example: 4 classes and 5 root files. The Trainer will take care of it at the backend|
|`MVAs`|list of dictionaries| MVAs to use. You can add as many as you like: MVAtypes XGB and DNN are keywords, so names can be XGB_new, DNN_old etc, but keep XGB and DNN in the names (That is how the framework identifies which algo to run)

```python
OutputDirName = 'TwoClassOutput'
branches=["ele_*","scl_eta","rho","matchedToGenEle","Fall17isoV2wp80", "Fall17isoV2wp90"]
SaveDataFrameCSV=True
loadfromsaved=False

ptbins = [10,30,40,50,100,5000]
etabins = [-1.6,-1.2,-0.8,-0.5,0.0,0.5,0.8,1.2,1.6]
ptwtvar='ele_pt'
etawtvar='scl_eta'
Reweighing = 'True'
WhichClassToReweightTo="Signal"
Classes = ['Background','Signal']
ClassColors = ['#377eb8', '#ff7f00']

processes=[
    {'Class':'Background','path':'./QCD.root',
     'xsecwt': 1, #xsec wt if any. Can be a branch name or number
     'CommonSelection':"(ele_pt>10) & (abs(scl_eta)<1.566)",
     'selection':"(ele_pt>10) & (abs(scl_eta)<1.566) & (matchedToGenEle == 0)", #selection for background
    },
    
    {'Class':'Background','path':'./GJet.root',
     'xsecwt': 1, #xsec wt if any, if none then it can be 1
     'CommonSelection':"(ele_pt>10) & (abs(scl_eta)<1.566)",
     'selection':"(ele_pt>10) & (abs(scl_eta)<1.566) & (matchedToGenEle == 0)", #selection for background
    },

    {'Class':'Signal','path':'./DY.root', 
     'xsecwt': 1, #xsec wt if any, if none then it can be 1
        'CommonSelection':"(ele_pt>10) & (abs(scl_eta)<1.566)",
        'selection':"(ele_pt>10) & (abs(scl_eta)<1.566) & (matchedToGenEle == 1)", #selection for signal
    }
]

Tree = "ntuplizer/tree"
OverlayWP=['Fall17isoV2wp80', 'Fall17isoV2wp90']
OverlayWPColors = ["black","purple"]
SigEffWPs=["80%","90%"]

MVAs = [
    
    {"MVAtype":"XGB_1",
     "Color":"green", #Plot color for MVA
     "Label":"XGB try 1", # label can be anything (this is how you will identify them on plot legends)
     "features":["ele_oldsigmaietaieta", "ele_oldsigmaiphiiphi", "ele_oldcircularity", "ele_oldr9"],
     "feature_bins":[100, 100, np.linspace(0.0,0.1,1001), 100], #only for plotting features (should be in the same order as features)
     'Scaler':"MinMaxScaler",
     "XGBGridSearch":{'min_child_weight': [5,2,1], 'gamma': [0.4,0.1], 'subsample': [0.6], 'colsample_bytree': [1.0], 'max_depth': [4,3]} #All standard XGB parameters supported
     #More than one element for any parameters will result in a grid search. example 'gamma': [0.4,0.1] has two grid poits in gamma. To not a grid search in this parameter, just do something like 'gamma': [0.4], which will fix the value at 0.4.
    },

    {"MVAtype":"DNN_1",
     "Color":"blue", #Plot color for MVA
     "Label":"DNN try 1", # label can be anything (this is how you will identify them on plot legends)
     "features":["ele_oldsigmaietaieta", "ele_oldsigmaiphiiphi", "ele_oldcircularity", "ele_oldr9"], #Input features #Should be branchs in your dataframe
     "feature_bins":[100, 100, np.linspace(0.0,0.1,1001), 100], #only for plotting features (should be in the same order as features)
     'Scaler':"MinMaxScaler",
     "DNNDict":{'epochs':100, 'batchsize':100, 'lr':0.001,
                #The other parameters which are not here, can be modified in Trainer script
                'model': Sequential([Dense(48, kernel_initializer='glorot_normal', activation='relu'),
                                     Dense(12, activation="relu"),
                                     Dropout(0.1),
                                     Dense(len(Classes),activation="softmax")]),
                'compile':{'loss':'binary_crossentropy','optimizer':Adam(lr=0.001), 'metrics':['accuracy']},
                'earlyStopping': EarlyStopping(monitor='val_loss', mode='min', verbose=1, patience=10)
               }
    },
]


```
#### Optional Parameters

| Parameters          |Type| Description| Default value|
| --------------- | ----------------| ---------------- | ---------------- |
|`testsize`|float| In fraction, how much data to use for testing (0.3 means 30%)| 0.2
|`flatten`       |boolean| For NanoAOD and other un-flattened trees, you can assign this as `True` to flatten branches with variable length for each event (Event level -> Object level)| False |
| `Debug`         |boolean| If True, only a small subset of events/objects are used for either Signal or background. Useful for quick debugging of code| False |
|`RandomState`|integer |Choose the same number everytime for reproducibility| 42|
|`MVAlogplot`|boolean| If true, MVA outputs are plotted in log scale| False|
|`Multicore`|boolean| If True all CPU cores available are used XGB | True|