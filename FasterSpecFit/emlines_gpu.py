import jax
import jax.numpy as np
from jaxopt import ScipyBoundedMinimize

jax.config.update("jax_enable_x64", True)

C_LIGHT = 299792.458
LOG10 = np.log(10)

def trapz_rebin(x, y, E):

  # Boolean assertions not allowed in Jax traced code
  #assert x[0] <= E[0] and E[-1] <= x[-1], "edges must be within input x range"
  
  P = np.searchsorted(x, E, side="right") - 1
  S = np.diff(y) / np.diff(x)

  def area(xl, xr, j):
    return (y[j] + S[j] * (0.5 * (xl + xr) - x[j])) * (xr - xl)

  r = np.minimum(x[P[:-1] + 1], E[1:])
  resultL = area(E[:-1], r, P[:-1])

  l = x[P[1:]]
  resultR = np.where(l > E[:-1], area(l, E[1:], P[1:]), 0.)

  B = np.searchsorted(E, x, side="right") - 1

  areapts = 0.5 * (y[1:] + y[:-1]) * np.diff(x)
  A = np.where(B[1:] == B[:-1], areapts, 0.)

  resultI = jax.ops.segment_sum(A, B[:-1], num_segments = len(E) - 1)

  result = resultL + resultR + resultI
  return result / np.diff(E)


@jax.jit
def build_emline_model(line_amplitudes, line_vshift, line_sigma,
                       obs_bin_edges,
                       redshift,
                       grid,
                       log_grid,
                       log_line_wavelengths):
    
    # line-width and redshifted wavelengths of spectral lines
    log_sigma = line_sigma / (C_LIGHT * LOG10)
    log_shifted_lines = log_line_wavelengths + \
        np.log10(1. + redshift + line_vshift / C_LIGHT)
    
    # flux contributions of each spectral line at each grid point
    Z = line_amplitudes * \
        np.exp(-0.5/log_sigma**2 * (log_grid[:,None] - log_shifted_lines)**2)
    
    # sum flux at each grid point over all spectral lines
    grid_fluxes = np.sum(Z, axis=1)
    
    model_fluxes = trapz_rebin(grid, grid_fluxes, obs_bin_edges)
    #model_fluxes = np.interp(obs_bin_edges, grid, grid_fluxes)
    
    # missing step -- apply the resolution matrix

    return model_fluxes


def _objective_function(parameters,
                        obs_bin_edges,
                        obs_fluxes,
                        obs_weights,
                        redshift,
                        grid,
                        log_grid,
                        log_line_wavelengths):

    line_amplitudes = parameters[:-2]
    line_vshift     = parameters[-2]
    line_sigma      = parameters[-1]

    model_fluxes = build_emline_model(line_amplitudes, line_vshift, line_sigma,
                                      obs_bin_edges,
                                      redshift,
                                      grid,
                                      log_grid,
                                      log_line_wavelengths)

    residuals = obs_weights * (model_fluxes - obs_fluxes)
    return 0.5 * np.sum(residuals**2)


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
# log_line_wavelengths: log-wavelengths of spectral lines being fit to data
#
# RETURNS:
# fitted line amplitudes, fitted velocity shift, fitted width,
# final objective value
#
def emlines(obs_wavelengths,
            obs_fluxes,
            obs_ivar,
            redshift,
            log_line_wavelengths):
    
    # create the oversampled (constant-velocity) wavelength array
    step_size = 5. / (C_LIGHT * LOG10)
    log_grid = np.arange(np.log10(np.min(obs_wavelengths) - 1),
                         np.log10(np.max(obs_wavelengths) + 1),
                         step_size)
    
    # statistical weights of observed fluxes
    obs_weights = np.sqrt(obs_ivar)
    
    farg = (
        centers_to_edges(obs_wavelengths),
        #obs_wavelengths,
        obs_fluxes,
        obs_weights,
        redshift,
        10**log_grid,
        log_grid,
        log_line_wavelengths
    )
    
    # initial guesses for parameters
    init_vshift     = np.array(0.)
    init_sigma      = np.array(75.)
    init_amplitudes = np.full_like(log_line_wavelengths, 1.)
    
    init_vals = np.hstack([init_amplitudes, init_vshift, init_sigma])
    
    bounds = np.array([[0.]  * len(log_line_wavelengths)  + [-100.,   0.],
                       [1e3] * len(log_line_wavelengths)  + [+100., 500.]])

    #Optimizer and Solutions
    lbfgsb = ScipyBoundedMinimize(fun=_objective_function,
                                  method="L-BFGS-B",
                                  dtype=np.float64,
                                  jit=True,
                                  )#options={"disp": True})
    lbfgsb_sol = lbfgsb.run(init_vals, bounds, *farg)

    params = lbfgsb_sol.params
    fitted_amplitudes = params[:-2]
    fitted_vshift     = params[-2]
    fitted_sigma      = params[-1]

    objval = lbfgsb_sol.state.fun_val
    
    return fitted_amplitudes, fitted_vshift, fitted_sigma, objval
