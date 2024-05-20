# FasterSpecFit

## Table of Contents 
- [About](#about)
- [Getting Started](#getting_started)
- [Usage](#Usage)

## About  <a name = "about"></a>

This branch includes a number of different implementations of emission
line fitting for FastSpecFit.  The implementations do not yet support
all features needed by the library but do illustrate a variety of approaches
to accelerating the fitting computation.

The list of implementations includes:

* emlines_cpu.py -- the current approach: estimate contribution of
  each peak at an array of equally-spaced points, then use trapezoidal
  rebinning to estimate the average flux in each observed bin.

* emlines_gpu.py -- the current approach, ported to the GPU with JAX

* emline_cpu_direct.py -- computes the average flux in each bin directly
  using a Gaussian integral to estimate the contribution of each peak

* emline_gpu_direct.py -- same approach as cpu_direct, but in JAX for
  the GPU

* emlines_sparse.py -- sparse version of the cpu_direct implementation
  that also implements an explicit computation of the Jacobian, rather
  than relying on the optimizer to estimate it.  Sparsity is slightly
  helpful for accelerating the emlines model but is more significant
  for the extremely sparse Jacobian.

* emlines_sparse_multi.py -- sparse cpu_direct computation that
  learns a separate shift and width for each fitted line

* emlines_sparse_custom.py -- improved version of sparse_multi that
  builds and uses a custom linear operator for the Jacobian
  
All the CPU implementations are accelerated with Numba, while the GPU
implementations use JAX.

## Getting Started <a name = "getting_started"></a>

These instructions will help get you a copy of the project up and running on your local machine for development and testing purposes.

### Prerequisites 

* Python 3.11.x or later (I've been using 3.12.2 successfully)
* NumPy
* SciPy
* AstroPy

For CPU:
* Numba 0.59.1 or later

For GPU:
* JAX 0.4.26 or later
* JAXopt 0.8.3 or later

Note that JAXopt will at some point be merged into the Optax library.

## Usage <a name = "Usage"></a>

> `python emlines_loop_execution impl`

where `impl` is the suffix of the implementation; e.g., to use
`emlines_sparse.py`, `impl` should be `sparse`.

The script expects to find its data files (spectra, list of spectral
lines, etc.) in a data subdirectory, whose name can be set near the
top of the code. Output is written to files `results-impl.py`, while
timings are written to `times-impl.py`.  Timings use timeit and are
averaged over tens of runs for accuracy.