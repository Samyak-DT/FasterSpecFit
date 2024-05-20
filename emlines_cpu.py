import numpy as np
from scipy.optimize import least_squares
from numba import jit, prange, set_num_threads

# Limit number of parallel CPU cores used
set_num_threads(1)

C_LIGHT = 299792.458
LOG10 = np.log(10)

@jit(nopython=True, fastmath=True, parallel=True, nogil=True)
def trapz_rebin(x, y, edges):

    assert x[0] <= edges[0] and edges[-1] <= x[-1], "edges must be within input x range"

    bins = np.empty(len(edges) - 1)
    
    #j = 0  #- index counter for inputs

    # compute area of each output bin in parallel
    for i in prange(len(bins)):

        #- Seek next sample beyond bin edge
        #while x[j] <= edges[i]:
        #    j += 1

        # j is index counter for inputs
        # for bin i, start with least j s.t. edges[i] < x[j]
        j = np.searchsorted(x, edges[i], side="right")
        
        area = 0.
        
        #- What is the y value where the interpolation crossed the edge?
        yedge = y[j-1] + (edges[i] - x[j-1]) * (y[j] - y[j-1]) / (x[j] - x[j-1])
        
        #- Is this sample inside this bin?
        if x[j] < edges[i+1]:
            area += 0.5 * (y[j] + yedge) * (x[j] - edges[i])

            #- Continue with interior bins
            while x[j+1] < edges[i+1]:
                area += 0.5 * (y[j+1] + y[j]) * (x[j+1] - x[j])
                j += 1
                
            #- Next sample will be outside this bin; handle upper edge
            yedge = y[j] + (edges[i+1] - x[j]) * (y[j+1] - y[j]) / (x[j+1] - x[j])
            area += 0.5 * (yedge + y[j]) * (edges[i+1] - x[j])

        #- Otherwise the samples span over this bin
        else:
            ylo = y[j] + (edges[i]   - x[j]) * (y[j] - y[j-1]) / (x[j] - x[j-1])
            yhi = y[j] + (edges[i+1] - x[j]) * (y[j] - y[j-1]) / (x[j] - x[j-1])
            area += 0.5 * (ylo + yhi) * (edges[i+1] - edges[i])

        bins[i] = area / (edges[i+1] - edges[i])
    
    return bins


@jit(nopython=True, fastmath=True, parallel=True, nogil=True)
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

    # compute total flux contribution of all spectral lines at each grid point
    grid_fluxes = np.empty_like(log_grid)
    for i in prange(len(log_grid)):
        acc = 0.
        for j in range(len(log_shifted_lines)):
            if np.abs(log_grid[i] - log_shifted_lines[j]) < 5. * log_sigma:
                acc += line_amplitudes[j] * np.exp(-0.5/log_sigma**2 * (log_grid[i] - log_shifted_lines[j])**2)
        grid_fluxes[i] = acc

    model_fluxes = trapz_rebin(grid, grid_fluxes, obs_bin_edges)
    
    # missing step -- apply the resolution matrix

    return model_fluxes


@jit(nopython=True, fastmath=True, nogil=True)
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
    
    return residuals


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
# fitted line amplitudes, fitted velocity shift, fitted width
#
def emlines(obs_wavelengths,
            obs_fluxes,
            obs_ivar,
            redshift,
            line_wavelengths):
    
    # create the oversampled (constant-velocity) wavelength array
    step_size = 5. / (C_LIGHT * LOG10)
    log_grid = np.arange(np.log10(obs_wavelengths[ 0] - 1),
                         np.log10(obs_wavelengths[-1] + 1),
                         step_size)
    
    # statistical weights of observed fluxes
    obs_weights = np.sqrt(obs_ivar)
    
    farg = (
        centers_to_edges(obs_wavelengths),
        obs_fluxes,
        obs_weights,
        redshift,
        10**log_grid,
        log_grid,
        np.log10(line_wavelengths)
    )
    
    # initial guesses for parameters
    init_vshift     = np.array(0.)
    init_sigma      = np.array(75.)
    init_amplitudes = np.full_like(line_wavelengths, 1.)
    
    init_vals = np.hstack([init_amplitudes, init_vshift, init_sigma])
    
    # lower and upper bounds for params
    bounds_min = [0.]   * len(line_wavelengths) + [-100.,   0.]
    bounds_max = [1e+3] * len(line_wavelengths) + [+100., 500.]
    
    #Optimizer and Solutions
    fit_info = least_squares(_objective_function,
                             init_vals,
                             bounds=[bounds_min, bounds_max],
                             args=farg,
                             max_nfev=100,
                             xtol=1e-8, method='trf', # verbose=2,
                             x_scale="jac", tr_solver="lsmr",
                             tr_options={"regularize": True})
    
    params = fit_info.x
    fitted_amplitudes = params[:-2]
    fitted_vshift     = params[-2]
    fitted_sigma      = params[-1]
    
    objval = fit_info.cost
        
    return fitted_amplitudes, fitted_vshift, fitted_sigma, objval
