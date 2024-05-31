import sys
from pathlib import Path
from importlib import import_module

import cProfile as profile
import pstats

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

NTRIALS = 10

pr = profile.Profile()

    
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
    for i in range(NTRIALS):
        fit.emlines(data["wavelength"].values,
                    data["flux"].values,
                    data["ivar"].values,
                    spectrum.redshift,
                    line_wavelengths)
    pr.disable()

st = pstats.Stats(pr).strip_dirs().sort_stats("cumulative")
st.print_stats()
st.print_callees()

#pr.print_stats(2)
