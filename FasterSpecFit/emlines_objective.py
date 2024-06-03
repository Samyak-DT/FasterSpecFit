#
# Calculation of objective function and
# Jacobian for emission line fitting
#

import numpy as np
from math import erf, erfc

from numba import jit

from .params_mapping import ParamsMapping
from .sparse_rep import EMLineJacobian

# Do not bother computing normal PDF/CDF if more than this many 
# standard deviations from mean.
MAX_SDEV = 5.


#
# norm_pdf()
# PDF of standard normal distribution at a point a
#
@jit(nopython=True, fastmath=False, nogil=True)
def norm_pdf(a):

    SQRT_2PI = np.sqrt(2 * np.pi)
    
    return 1/SQRT_2PI * np.exp(-0.5 * a**2)


#
# norm_cdf()
# Approximate the integral of a standard normal PDF from -infty to a.
#
# Optimization (currently disabled because it is not needed): If
# |a| > MAX_SDEV, treat the value as extreme and return 0 or 1 as
# appropriate.
#
@jit(nopython=True, fastmath=False, nogil=True)
def norm_cdf(a):

    SQRT1_2 = 1.0 / np.sqrt(2)
    
    z = np.abs(a)

    #if z > MAX_SDEV: # short-circuit extreme values
    #    if a > 0:
    #        y = 1.
    #    else:
    #        y = 0.
    if z < 1.:
        y = 0.5 + 0.5 * erf(a * SQRT1_2)
    else:
        y = 0.5 * erfc(z * SQRT1_2)
        if a > 0:
            y = 1.0 - y
    
    return y


##############################################################

#
# Objective function for least-squares optimization
#
# Build the emline model as described above and compute the weighted
# vector of residuals between the modeled fluxes and the observations.
#
def EMLine_objective(free_parameters,
                     bin_data,
                     obs_fluxes,
                     obs_weights,
                     redshift,
                     line_wavelengths,
                     resolution_matrices,
                     camerapix,
                     params_mapping):

    #
    # expand free paramters into complete
    # parameter array, handling tied params
    # and doublets
    #
    parameters = params_mapping.mapFreeToFull(free_parameters)
    lineamps, linevshifts, linesigmas = np.array_split(parameters, 3)

    log_obs_bin_edges, ibin_widths = bin_data
    
    model_fluxes = np.empty_like(obs_fluxes, dtype=obs_fluxes.dtype)
    
    for icam, campix in enumerate(camerapix):

        # start and end for obs fluxes of camera icam
        s, e = campix

        # Actual bin widths are in ibw[1..e-s].
        # Setup guarantees that ibw[0] and
        # ibw[e-s+1] are not out of bounds.
        ibw = ibin_widths[s:e+1]
        
        mf = build_emline_model(lineamps, linevshifts, linesigmas,
                                log_obs_bin_edges[s+icam:e+icam+1],
                                ibw,
                                redshift,
                                line_wavelengths)
        
        # convolve model with resolution matrix and store in
        # this camera's subrange of model_fluxes
        resolution_matrices[icam].matvec(mf, model_fluxes[s:e])
        
    return obs_weights * (model_fluxes - obs_fluxes) # residuals



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
#   line_vshifts   -- additional velocity shift for each fitted lines
#   line_sigmas    -- width of Gaussian profile for each fitted lines
#
#   log_obs_bin-edges -- natural log of observed wavelength bin edges
#   ibin_widths       -- one over widths of each observed wavelength bin
#   redshift          -- red shift of observed spectrum
#   line_wavelengths  -- array of wavelengths for all fitted spectral lines
#
# RETURNS:
#   vector of average fluxes in each observed wavelength bin
#
@jit(nopython=True, fastmath=False, nogil=True)
def build_emline_model(line_amplitudes, line_vshifts, line_sigmas,
                       log_obs_bin_edges,
                       ibin_widths,
                       redshift,
                       line_wavelengths):
    
    C_LIGHT = 299792.458
    SQRT_2PI = np.sqrt(2 * np.pi)
    
    nbins = len(log_obs_bin_edges) - 1
    
    # output per-bin fluxes
    # entry i corresponds bin i-1
    # entry 0 is a dummy in case lo == 0
    # last entry is a dummy in case hi == nbins + 1
    model_fluxes = np.zeros(nbins + 1, dtype=line_amplitudes.dtype)

    # temporary buffer for per-line calculations, sized large
    # enough for whatever we may need to compute ([lo - 1 .. hi])
    max_width = int(2*MAX_SDEV*np.max(line_sigmas/C_LIGHT) / \
                    np.min(np.diff(log_obs_bin_edges))) + 4
    edge_vals = np.empty(max_width, dtype=model_fluxes.dtype)
        
    # compute total area of all Gaussians inside each bin.
    # For each Gaussian, we only compute area contributions
    # for bins where it is non-negligible.
    for j in range(len(line_wavelengths)):

        # line width
        sigma = line_sigmas[j] / C_LIGHT
        c = SQRT_2PI * sigma * np.exp(0.5 * sigma**2)
    
        # wavelength shift for spectral lines
        line_shift = 1. + redshift + line_vshifts[j] / C_LIGHT
    
        shifted_line     = line_wavelengths[j] * line_shift
        log_shifted_line = np.log(shifted_line)
        
        # leftmost edge i that needs a value (> 0) for this line 
        lo = np.searchsorted(log_obs_bin_edges,
                             log_shifted_line - MAX_SDEV * sigma,
                             side="left")
        
        # leftmost edge i that does *not* need a value (== 1) for this line
        hi = np.searchsorted(log_obs_bin_edges,
                             log_shifted_line + MAX_SDEV * sigma,
                             side="right")
        
        if hi == lo:  # entire Gaussian is outside bounds of log_obs_bin_edges
            continue
        
        nedges = hi - lo + 2  # compute values at edges [lo - 1 ... hi]
        
        A = c * line_amplitudes[j] * shifted_line
        offset = log_shifted_line / sigma + sigma
        
        # vals[i] --> edge i + lo - 1
        
        edge_vals[0] = 0. # edge lo - 1
        
        for i in range(1, nedges-1):
            
            # x = (log(lambda_j) - mu_i)/sigma - sigma,
            # the argument of the Gaussian integral
            
            x = log_obs_bin_edges[i+lo-1]/sigma - offset
            edge_vals[i] = A * norm_cdf(x)
        
        edge_vals[nedges-1] = A  # edge hi
        
        # convert vals[i] to avg of bin i+lo-1 (last value is garbage)
        # we get values for bins lo-1 to hi-1 inclusive
        for i in range(nedges-1):
            edge_vals[i] = (edge_vals[i+1] - edge_vals[i]) * ibin_widths[i+lo]
            
        # add bin avgs for this peak to the full array
        model_fluxes[lo:hi+1] += edge_vals[:nedges-1] 
        
    # trim off left and right dummy values before returning
    return model_fluxes[1:-1]


###############################################################################

#
# Jacobian of objective function for least-squares EMLine
# optimization. The result of the detailed calculation is converted
# to a sparse matrix, since it is extremely sparse, to speed up
# subsequent matrix-vector multiplies in the optimizer.
#
def EMLine_jacobian(free_parameters,
                    bin_data,
                    obs_fluxes, # not used, but must match objective
                    obs_weights,
                    redshift,
                    line_wavelengths,
                    resolution_matrices,
                    camerapix,
                    params_mapping):
    
    #
    # expand free paramters into complete
    # parameter array, handling tied params
    # and doublets
    #
    
    parameters = params_mapping.mapFreeToFull(free_parameters)
    lineamps, linevshifts, linesigmas = np.array_split(parameters, 3)

    log_obs_bin_edges, ibin_widths = bin_data
    
    J_S = params_mapping.getJacobian(free_parameters)

    jacs = []
    for icam, campix in enumerate(camerapix):
        s, e = campix

        # Actual bin widths are in ibw[1..e-s].
        # Setup guarantees that ibw[0] and
        # ibw[e-s+1] are not out of bounds.
        ibw = ibin_widths[s:e+1]

        idealJac = \
            build_emline_model_jacobian(lineamps, linevshifts, linesigmas,
                                        log_obs_bin_edges[s+icam:e+icam+1],
                                        ibw,
                                        redshift,
                                        line_wavelengths)
        
        # ignore any columns corresponding to fixed parameters
        endpts = idealJac[0]
        endpts[params_mapping.fixedMask(), :] = (0, 0)
        jacs.append( mulWMJ(obs_weights[s:e],
                            resolution_matrices[icam].data,
                            idealJac) )
    
    nBins = np.sum(np.diff(camerapix))
    nFreeParms = len(free_parameters)
    nParms = len(parameters)
    J =  EMLineJacobian((nBins, nFreeParms), nParms,
                        camerapix, jacs, J_S)

    #jacEstimateConditionNumber(J)
    
    return J


#
# for debugging -- compute singular values of Jacobian and estimate
# its condition number.
#
def jacEstimateConditionNumber(J):

    from scipi.sparse.linalg import svds
    
    _, nFreeParms = J[0]
    
    try:
        svs = svds(J, return_singular_vectors=False,
                   k=nFreeParms - 1, which="LM")
        sv0 = svds(J, return_singular_vectors=False,
                   k=1, which="SM")[0]
        cond = svs[-1] / sv0
        print(np.hstack((sv0, svs)))       
        print(f"cond(J) = {cond:.3e}")
        
    except:
        print("Failed to compute Jacobian condition number")

    
#
# build_emline_model_jacobian() 
#
# Compute the Jacobian of the function computed in build_emlines_model().
# Inputs are as for build_emlines_model().
#
# RETURNS:
# Sparse Jacobian as tuple (endpts, dd), where
#  column j has nonzero values in interval [ endpts[j,0] , endpts[j,1] )
#  which are stored in dd[j].
#
@jit(nopython=True, fastmath=False, nogil=True)
def build_emline_model_jacobian(line_amplitudes, line_vshifts, line_sigmas,
                                log_obs_bin_edges,
                                ibin_widths,
                                redshift,
                                line_wavelengths):

    C_LIGHT = 299792.458
    SQRT_2PI = np.sqrt(2*np.pi)

    nbins = len(log_obs_bin_edges) - 1
    
    max_width = int(2*MAX_SDEV*np.max(line_sigmas/C_LIGHT) / \
                    np.min(np.diff(log_obs_bin_edges))) + 4
    
    nlines = len(line_wavelengths)
    dd     = np.empty((3 * nlines, max_width), dtype=line_amplitudes.dtype)
    endpts = np.zeros((nlines, 2), dtype=np.int32)

    starts = endpts[:,0]
    ends   = endpts[:,1]
    
    # compute partial derivatives for avg values of all Gaussians
    # inside each bin. For each Gaussian, we only compute
    # contributions for bins where it is non-negligible.
    for j in range(len(line_wavelengths)):
        
        # line width
        sigma = line_sigmas[j] / C_LIGHT
        c0 = SQRT_2PI * np.exp(0.5 * sigma**2)
        
        # wavelength shift for spectral lines
        line_shift = 1. + redshift + line_vshifts[j] / C_LIGHT
        shifted_line     = line_wavelengths[j] * line_shift
        log_shifted_line = np.log(shifted_line)
        
        # leftmost edge i that needs a value (> 0) for this line 
        lo = np.searchsorted(log_obs_bin_edges,
                             log_shifted_line - MAX_SDEV * sigma,
                             side="left")
        
        # leftmost edge i that does *not* need a value (== 1) for this line
        hi = np.searchsorted(log_obs_bin_edges,
                             log_shifted_line + MAX_SDEV * sigma,
                             side="right")
                    
        if hi == lo:  # Gaussian is entirely outside bounds of log_obs_bin_edges
            continue

        nedges = hi - lo + 2 # compute values at edges [lo - 1 ... hi]
        
        # Compute contribs of each line to each partial derivative in place.
        # No sharing of params between peaks means that we never have to
        # add contributions from two peaks to same line.
        dda_vals = dd[           j]
        ddv_vals = dd[nlines   + j]
        dds_vals = dd[2*nlines + j]
        
        offset = log_shifted_line / sigma + sigma
        
        c = c0 * line_wavelengths[j]
        A = c / C_LIGHT * line_amplitudes[j]
        
        # vals[i] --> edge i + lo - 1
        
        dda_vals[0] = 0. # edge lo - 1
        ddv_vals[0] = 0.
        dds_vals[0] = 0.
        
        for i in range(1, nedges - 1):
            
            # x - offset = (log(lambda_j) - mu_i)/sigma - sigma,
            # the argument of the Gaussian integral
            
            x = log_obs_bin_edges[i+lo-1]/sigma - offset
            pdf = norm_pdf(x)
            cdf = norm_cdf(x)
            
            dda_vals[i] = c * line_shift * sigma * cdf
            ddv_vals[i] = A * (sigma * cdf - pdf)
            dds_vals[i] = A * line_shift * \
                ((1 + sigma**2) * cdf - (x + 2*sigma) * pdf)
            
        dda_vals[nedges - 1] = c * line_shift * sigma     # edge hi
        ddv_vals[nedges - 1] = A * sigma
        dds_vals[nedges - 1] = A * line_shift * (1 + sigma**2)
        
        # convert *_vals[i] to partial derivatives for bin i+lo-1
        # (last value in each array is garbage)
        # we get values for bins lo-1 to hi-1 inclusive
        for i in range(nedges - 1):
            dda_vals[i] = (dda_vals[i+1] - dda_vals[i]) * ibin_widths[i+lo]
            ddv_vals[i] = (ddv_vals[i+1] - ddv_vals[i]) * ibin_widths[i+lo]
            dds_vals[i] = (dds_vals[i+1] - dds_vals[i]) * ibin_widths[i+lo]
        
        # starts[j] is set to first valid bin
        if lo == 0:
            # bin low - 1 is before start of requested bins,
            # and its true left endpt value is unknown
            starts[j] = lo

            # discard bin lo - 1 in dd*_vals
            dda_vals[0:nedges - 1] = dda_vals[1:nedges]
            ddv_vals[0:nedges - 1] = ddv_vals[1:nedges]
            dds_vals[0:nedges - 1] = dds_vals[1:nedges]
        else:
            # bin lo - 1 is valid
            starts[j] = lo - 1
        
        # ends[j] is set one past last valid bin
        if hi >= nbins + 1:
            # bin hi - 1 is one past end of requested bins,
            # and its true right endpt value is unknown
            ends[j] = hi - 1
        else:
            # bin hi - 1 is valid
            ends[j] = hi

    return (mytile(endpts, 3), dd)    


# replacement for np.tile, which is not supported by Numba
@jit(nopython=True, fastmath=False, nogil=True)
def mytile(a, n):
    sz = a.shape[0]
    r = np.empty((n * sz, a.shape[1]), dtype=a.dtype)
    for i in range(n):
        r[i*sz:(i+1)*sz,:] = a
    return r

@jit(nopython=True, fastmath=False, nogil=True)
def mulWJ(w, jac):
    endpts, J = jac
    ncol, _ = J.shape

    for j in range(ncol):
        # boundaries of nonzero entries
        # in jth column of J
        s, e = endpts[j]
        
        if s == e: # no nonzero values in column j
            continue
        
        for i in range(s,e):
            J[j, i - s] *= w[i]

    return jac

#
# mulWMJ()
# Compute the sparse matrix product WMJ, where
#   W is a diagonal matrix (represented by a vector)
#   M is a resolution matrix (the .data field of a
#     ResMatrix, since Numba can't handle classes)
#   J is a column-sparse matrix computed by the ideal
#     Jacobian calculation.
#
# Return the result as a column-sparse matrix in the
# same form as J.
#
# NB: for future, we should allocate enough extra space
# in J to let us create P in place, overwriting J
#
@jit(nopython=True, fastmath=False, nogil=True)
def mulWMJ(w, M, jac):

    endpts, J = jac
    
    nbins, ndiag = M.shape
    ncol, maxColSize = J.shape
    
    hdiag = ndiag//2
        
    P = np.empty((ncol, maxColSize + ndiag - 1), dtype=J.dtype)
    endptsP = np.zeros((ncol,2), dtype=np.int32)
    
    for j in range(ncol):
        # boundaries of nonzero entries
        # in jth column of J
        s, e = endpts[j]
        
        if s == e: # no nonzero values in column j
            continue
            
        # boundaries of entries in jth column of P
        # impacted by matrix multiply
        imin = np.maximum(s - hdiag, 0)
        imax = np.minimum(e + hdiag, nbins) # one past last impacted entry
        
        for i in range(imin, imax):
            
            # boundaries of interval of k where both
            # M[i, k] and J[k, j] are nonzero.
            kmin = np.maximum(i - hdiag,     s)
            kmax = np.minimum(i + hdiag, e - 1)
        
            acc = 0.
            for k in range(kmin, kmax + 1):
                acc += M[i, k - i + hdiag] * J[j, k - s]
            P[j, i - imin] = acc * w[i]

        endptsP[j] = np.array([np.maximum(imin, 0),
                               np.minimum(imax, nbins)])
        
    return (endptsP, P)


###############################################################################

#
# bin_centers_to_edges()
# Convert N bin centers to N+1 bin edges.  Edges are placed
# halfway between centers, with extrapolation at the ends.
#
@jit(nopython=True, fastmath=False, nogil=True)
def prepare_bins(centers, camerapix):

    ncameras = camerapix.shape[0]
    edges = np.empty(len(centers) + ncameras, dtype=centers.dtype)
    
    for icam, campix in enumerate(camerapix):

        s, e = campix
        icenters = centers[s:e]
        
        #- interior edges are just points half way between bin centers
        int_edges = 0.5 * (icenters[:-1] + icenters[1:])
        
        #- exterior edges are extrapolation of interior bin sizes
        edge_l = icenters[ 0] - (icenters[ 1] - int_edges[ 0])
        edge_r = icenters[-1] + (icenters[-1] - int_edges[-1])

        edges[s + icam]              = edge_l
        edges[s + icam + 1:e + icam] = int_edges
        edges[e + icam]              = edge_r

    # dummies before and after widths are needed
    # for corner cases in edge -> bin computation
    ibin_widths = np.hstack((np.array([0.]),
                             1. / np.diff(edges),
                             np.array([0.])))
    
    return (np.log(edges), ibin_widths)
