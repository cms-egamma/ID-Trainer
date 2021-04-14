# ID-Trainer

### Clone

```
git clone https://github.com/cms-egamma/ID-Trainer.git

```

### Setup

```
source /cvmfs/sft.cern.ch/lcg/views/LCG_97python3/x86_64-centos7-gcc8-opt/setup.sh

```
### Create a new config (Just copy the default one and start editing on top of it)

```
cp Tools/TrainConfig.py Tools/NewTrainConfig.py

```
### All you need to do is to edit the NewTrainConfig.py with the settings for your analysis and then run 

``` 
python Trainer.py Tools/NewTrainConfig

```

### The Trainer.py will read the settings from the config file and run training

## Suggestion : Do not remove or touch the original Tools/TrainConfig.py (Keep it for reference)


# What can this framework do?

<img src="READMEDocs/SampleROC.png" alt="drawing" width="30%"/>

# Projects using the framework

1) Run-3 Ele MVA ID
2) Close photon analysis
3) H->eeg analysis
