#!/bin/bash
cd /afs/cern.ch/user/a/akapoor/workspace/2020/IDTRainer/ID-Trainer
source /cvmfs/sft.cern.ch/lcg/views/LCG_97py3cu10/x86_64-centos7-gcc7-opt/setup.sh
#source /cvmfs/sft.cern.ch/lcg/views/dev3cuda/latest/x86_64-centos7-gcc8-opt/setup.sh
python Trainer.py $1
