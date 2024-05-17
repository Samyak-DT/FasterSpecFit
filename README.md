# FasterSpecFit

## Table of Contents 
- [About](#about)
- [Getting Started](#getting_started)
- [Usage](#Usage)

## About  <a name = "about"></a>

FasterSpecFit is an alternative to FastSpecFit used in DESI for non-negative least squares fitting (NNLS) and non-linear fitting of spectral data. We chose the Google JAX as our programming language as Google JAX is being well-maintained by the official team and since it already has a library JAXopt which already contains optimizers that already does NNLS and non-linear line fitting and is being updated by the same developers  as JAX. FasterSpecFit takes in data of an spectra and then uses the emission lines given to create a model of the spectra. FasterSpecFit can make the model using both CPU and GPU and can also do benchmarking tests on how fast the runtimes are on GPU vs CPU and how closely does the two models optimize parameters. 

## Getting Started <a name = "getting_started"></a>

These instructions will help get you a copy of the project up and running on your local machine for development and testing purposes.

### Prerequisites 

#### Python

The program is written in Python programming language. It is one of the standard languages used in academic  Specifically, this program was developed and was written in Python version 3.11.8

```
pip install python
```

#### Google JAX 

Google JAX is the main library used for line-fiting model of GPU version of FastSpecFit. This program was developed in the JAX version 0.4.26 

```
pip install jax
```

#### JAXopt 

JAXopt is the library based on Google JAX with the hardware-accelerated, differentiable optimizers for GPUs and TPUs. This program was  developed in the JAXopt version 0.8.3

```
pip install jaxopt
```

#### Numpy 

Numpy is the library used mainly for the CPU version of the FasterSpecFit and for looping and benchmarking tests in FasterSpecFit. 

```
pip install numpy
```

#### SciPy

Scipy is the library that provides the optimiser for the CPU version of FasterSpecFit. 

```
pip install scipy
```

#### AstroPy
Astropy is used to access a "fasterspec.ecsv file which gives necessary emission lines for the model. 
```
pip install astropy
```

## Usage <a name = "Usage"></a>

In order for input, this program takes two types of .txt files. 

One is an metadata txt file containing all the filenames of the spectra and their redshifts. The first columns needs to be the list of targetIDs we're optimizing for, the second column is the redshifts of the target IDs and the third column needs to be the filenames of the spectra of the targetIDs. 

For the actual spectra files themselves, they are also in .txt format. In the txt file, the first column is the actual length of spectrum values that has been observed for a spectra, the second column is the flux values (the amount of light) that has been seen at the particular spectral values and the third is the inverse variance of the flux values measured at that spectrum. 


 
