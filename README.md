# ID-Trainer

A simple config based tool for ID-Training and high-energy-physics machine learning tasks.
Currently supports:
1) Binary-classification
Examples: DY vs ttbar, DY prompt vs DY fake
3) Multi-sample classification
Examples: DY vs (ttbar and QCD)
5) Multi-class classification (experimental)
Examples: DY vs ttbar vs QCD



### Setting in up

#### Clone
```
git clone https://github.com/cms-egamma/ID-Trainer.git
```
#### Setup
In principle you can set this up on you local computer by installing packages via conda/pip, but if it is possible to setup a cvmfs release, you can do that using LCG 97python3! (Tested at lxplus and SWAN)

`source /cvmfs/sft.cern.ch/lcg/views/LCG_97python3/x86_64-centos7-gcc8-opt/setup.sh`

### Running the trainer
#### Create a config
Create a new python config. Some sample python configs are available in the 'Configs' folder. They cover most possible examples.All you need to do is to edit the config with the settings for your analysis and then run

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

```python
import numpy as np
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Activation
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.layers import Dropout
from tensorflow.keras.callbacks import EarlyStopping
```
This is needed to use numpy and tensorflow. You can leave it as is.

```python
OutputDirName = 'TwoClassOutput'
```
All plots, models, config file will be stored in this directory. This will be automatically created. If it already exists, it will overwrite everything if you run it again with the same OutputDirName.

```python
Debug=True
```
If True, only a small subset of events/objects are used for either Signal or background. Useful for quick debugging of code.

```python
branches=["ele_*","scl_eta","rho","matchedToGenEle","Fall17isoV2wp80", "Fall17isoV2wp90"]
```
Branches to read: Should be in the input root files. Only these branches can be later used for any purpose. The '\*' is iseful for pattern based brancges. In the above all branches with 'ele_' as prefix will be read along with other mentioned branches.


