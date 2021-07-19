#!/bin/bash
cd /afs/cern.ch/user/a/akapoor/workspace/2020/IDTRainer/ID-Trainer

#use only one of the source commands

## For GPU
#source /cvmfs/sft.cern.ch/lcg/views/LCG_97py3cu10/x86_64-centos7-gcc7-opt/setup.sh

## For only CPU
source /cvmfs/sft.cern.ch/lcg/views/LCG_97python3/x86_64-centos7-gcc8-opt/setup.sh

python Trainer.py $1
