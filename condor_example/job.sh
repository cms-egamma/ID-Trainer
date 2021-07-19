universe = vanilla
+JobFlavour = "workday"
executable            = train.sh
arguments = "Config"
log = test.log
output = condor_ouput/outfile.$(Cluster).$(Process).out
error = condor_ouput/errors.$(Cluster).$(Process).err
request_GPUs = 1
request_CPUs = 4
+testJob = True
queue 
