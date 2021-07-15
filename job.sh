universe = vanilla
+JobFlavour = "workday"
executable            = train.sh
arguments = "PFElectronConfig_EB_fullpT"
log = test.log
output = condor_ouput/outfile.$(Cluster).$(Process).out
error = condor_ouput/errors.$(Cluster).$(Process).err
request_GPUs = 1
request_CPUs = 4
+testJob = True
queue 

#PFElectronConfig_EB_highpT.py  PFElectronConfig_EB_lowpT.py  PFElectronConfig_EE_highpT.py  PFElectronConfig_EE_lowpT.py
