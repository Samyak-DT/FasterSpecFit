import sys
import time
from pathlib import Path
from importlib import import_module
import timeit

import cProfile as profile

import numpy as np
import pandas as pd
from astropy.table import Table

data_path = Path("./data-5")
#data_path = Path("./data-100")

pRuntime = sys.argv[1]
fit = import_module(f"emlines_{pRuntime}")

# Reading the Model with all the emlines
t = Table.read(data_path / "fastspec-emlines.ecsv")
line_wavelengths = t["restwave"].data

#Reading the metafile
df = pd.read_csv(data_path / "fastspec-sample.txt",
                 delimiter = " ",
                 names = ["targetid", "redshift", "file"],
                 dtype = {
                     "targetid": str,
                     "redshift": np.float64,
                     "file": str
                 })

coldcache = True

NTRIALS = 50
totalTime = 0.

pr = profile.Profile()

with open(f"times-{pRuntime}.txt", "w") as times:
    with open(f"results-{pRuntime}.txt", "w") as results:
        print(f"Runtime: {pRuntime}", file=times)
        print(f"Time of Execution: {time.ctime(time.time())}", file=times)
    
        #Calculating the time for each file to process
        for spectrum in df.itertuples(index=False):
            
            data = pd.read_csv(data_path / spectrum.file,
                               delimiter = " ",
                               names = ["wavelength", "flux", "ivar", "xxx"])
            
            if coldcache:
                for i in range(10):
                    # run once to warm up the cache and do any initial
                    # driver loading
                    fit.emlines(data["wavelength"].values,
                                data["flux"].values,
                                data["ivar"].values,
                                spectrum.redshift,
                                line_wavelengths)
                coldcache = False

            pr.enable()
            fit.emlines(data["wavelength"].values,
                        data["flux"].values,
                        data["ivar"].values,
                        spectrum.redshift,
                        line_wavelengths)
            pr.disable()
            
            fitted_amplitudes, fitted_vshift, fitted_sigma, objval = \
                fit.emlines(data["wavelength"].values,
                            data["flux"].values,
                            data["ivar"].values,
                            spectrum.redshift,
                            line_wavelengths)
            print(fitted_amplitudes, fitted_vshift, fitted_sigma, objval, file=results)


pr.print_stats(2)
