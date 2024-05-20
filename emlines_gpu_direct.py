#
# Multicore implementation of spectral fitting via Gaussian
# integration.
#

import jax
import jax.numpy as np
import jax.scipy.stats.norm as norm

from jaxopt import ScipyBoundedMinimize

jax.config.update("jax_enable_x64", True)


#
# build_emline_model() 
#
# Given a fixed set of spectral lines and known redshift, and estimates
# for the amplitude of each line and a common velocity shift and line width,
# compute the average value observed in each of a series of spectral bins
# whose edges are defined by obs_bin_edges.
#
# INPUTS:
#   line_amplitudes -- amplitudes for each fitted spectral line
#   line_vshift     -- additional velocity shift common to all fitted lines
#   line_sigma      -- width of Gaussian profile common to all fitted lines
#
#   obs_bin_edges     -- edges for observed spectral bins in ascending order
#   log_obs_bin-edges -- natural log of values in obs_bin_edges
#   redshift          -- red shift of observed spectrum
#   line_wavelengths  -- array of wavelengths for all fitted spectral lines
#
# RETURNS:
#   vector of average fluxes in each observed wavelength bin
#
def build_emline_model(line_amplitudes, line_vshift, line_sigma,
                       obs_bin_edges,
                       log_obs_bin_edges,
                       redshift,
                       line_wavelengths):

    C_LIGHT = 299792.458
    
    # line width
    sigma = line_sigma / C_LIGHT

    # redshifted wavelengths of spectral lines and per-line offsets
    # for evaluation of Gaussian integral
    line_shift = 1. + redshift + line_vshift / C_LIGHT
    shifted_lines = line_wavelengths * line_shift
    offset = np.log(shifted_lines) / sigma

    # per-line multiplier A for Gaussian integral
    c = np.sqrt(2 * np.pi) * sigma * np.exp(0.5 * sigma**2)
    A = c * line_amplitudes * shifted_lines

    # x - offset[j] = (log(lambda_j) - mu_i)/sigma - sigma,
    # the argument of the Gaussian integral
    
    x = log_obs_bin_edges / sigma - sigma  
    edge_vals = np.sum(A * norm.cdf(x[:,None] - offset), axis=1)
    model_fluxes = np.diff(edge_vals) / np.diff(obs_bin_edges)
    
    # missing step -- apply the resolution matrix

    return model_fluxes


#
# Objective function for least-squares optimization Build the emline
# model as described above and compute the weighted vector of
# residuals between the modeled fluxes and the observations.
#
# *args contains the fixed arguments to build_emline_model()
#
def _objective_function(parameters,
                        obs_fluxes,
                        obs_weights,
                        *args):
    
    line_amplitudes = parameters[:-2]
    line_vshift     = parameters[-2]
    line_sigma      = parameters[-1]

    model_fluxes = build_emline_model(line_amplitudes,
                                      line_vshift,
                                      line_sigma,
                                      *args)
    
    residuals = obs_weights * (model_fluxes - obs_fluxes)
    return 0.5 * np.sum(residuals**2)


#
# centes_to_edges()
# Convert N bin centers to N+1 bin edges.  Edges are placed
# halfway between centers, with extrapolation at the ends.
#
def centers_to_edges(centers):
  
  #- Interior edges are just points half way between bin centers
  int_edges = 0.5 * (centers[:-1] + centers[1:])
  
  #- edge edges are extrapolation of interior bin sizes
  edge_l = centers[ 0] - (centers[ 1] - int_edges[ 0])
  edge_r = centers[-1] + (centers[-1] - int_edges[-1])
  
  return np.hstack((edge_l, int_edges, edge_r))


#
# emlines()
# Fit a set of noisy flux measurements to an underlying collection of
# spectral lines, assigning an amplitude to each line.
#
# INPUTS:
# obs_wavelengths: vector of wavelengths at which flux was measured
# obs_fluxes:      fluxes at each observed wavelength
# obs_ivar:        1/variance of flux of each observed wavelength
#
# redshift:       redshift of object's spectrum
# line_wavelengths: wavelengths of spectral lines being fit to data
#
# RETURNS:
# fitted line amplitudes, fitted velocity shift, fitted width, objective value
#
def emlines(obs_wavelengths,
            obs_fluxes,
            obs_ivar,
            redshift,
            line_wavelengths):
    
    # statistical weights of observed fluxes
    obs_weights = np.sqrt(obs_ivar)
    obs_bin_edges = centers_to_edges(obs_wavelengths)
    farg = (
        obs_fluxes,
        obs_weights,
        obs_bin_edges,
        np.log(obs_bin_edges),
        redshift,
        line_wavelengths
    )
    
    # initial guesses for parameters
    init_vshift     = np.array(0.)
    init_sigma      = np.array(75.)
    init_amplitudes = np.full_like(line_wavelengths, 1.)
    
    init_vals = np.hstack([init_amplitudes, init_vshift, init_sigma])

    # lower and upper bounds for params
    bounds = np.array([[0.]  * len(line_wavelengths)  + [-100.,   0.],
                       [1e3] * len(line_wavelengths)  + [+100., 500.]])

    # optimize!
    lbfgsb = ScipyBoundedMinimize(fun=_objective_function,
                                  method="L-BFGS-B",
                                  #options={"disp": True},
                                  dtype=np.float64,
                                  jit=True)
    
    lbfgsb_sol = lbfgsb.run(init_vals, bounds, *farg)
    
    # extract solution
    params = lbfgsb_sol.params
    fitted_amplitudes = params[:-2]
    fitted_vshift     = params[-2]
    fitted_sigma      = params[-1]

    objval = lbfgsb_sol.state.fun_val
    
    return fitted_amplitudes, fitted_vshift, fitted_sigma, objval
