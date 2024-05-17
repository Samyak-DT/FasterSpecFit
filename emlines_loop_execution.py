from datetime import datetime
import time
from astropy.table import Table
import numpy as np

pRuntime = input("What runtime do you want to run the program on [CPU/GPU]")
if pRuntime == "CPU":
    import emlines_cpu as fit
    paramTxt = "params_cpu.txt"
    timeTxt = "time_cpu.txt"
elif pRuntime == "GPU":
    import emlines_gpu as fit
    paramTxt = "params_gpu.txt"
    timeTxt = "time_gpu.txt"
else: 
    print("This program only supports GPU and CPU runtime for now")
# Reading the Model with all the emlines
print('Reading emission line file.')
t = Table.read('fastspec-emlines.ecsv')
# trim to just the key optical lines
t = t[12:34]

linewaves = t['restwave'].data

#Reading the metafile 
targetids, redshifts, specfiles = np.genfromtxt('fastspec-sample.txt', dtype=None, encoding=None, unpack=True, skip_header=0)
print(targetids, redshifts)
targetids = np.atleast_1d(targetids)
redshifts = np.atleast_1d(redshifts)
specfiles = np.atleast_1d(specfiles)


totaltime = 0
exec_arr = []
#Calculating the time for each file to process
for redshift, specfile in zip(redshifts, specfiles):
    wave, flux, ivar, _ = np.loadtxt(specfile, unpack=True)
    start_time = time.time()
    bestamps, bestvshifts, bestsigmas = fit.emlines([wave, flux, ivar], linewaves, redshift)
    end_time = time.time()
    exec_time = end_time - start_time 
    exec_arr.append(exec_time)
    print("Execution time: " , exec_time , "seconds")
    totaltime = totaltime + exec_time
np.savetxt(timeTxt, exec_arr)
    
print("Total Time:" , totaltime)


params = np.column_stack((bestamps, bestvshifts, bestsigmas))
np.savetxt(paramTxt , params, delimiter=" ")
#Line-Spacing for the Next Experiment    

