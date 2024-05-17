
#Necessary Imports

#Constants
C_LIGHT = 299792.458


import jaxopt
import jax
import jax.numpy as jnp
from jax import config
from jaxopt import ScipyBoundedMinimize
config.update("jax_enable_x64" , True)


def emlines( data, linewaves, redshift):

  

  #Making sure all the arrays are CPU/GPU friendly 
  emlinewave = jnp.asarray(data[0])
  emlineflux = jnp.asarray(data[1])
  emlineivar = jnp.asarray(data[2])
  linewaves = jnp.asarray(linewaves)
  nline = len(linewaves)
  
  @jax.jit
  def build_emline_model_gpu(lineamps, linevshifts, linesigmas, linewaves,
                      redshift, log10wave, emlinewave, X):

    # line-width [log-10 Angstrom] and redshifted wavelength [log-10 Angstrom]
    log10sigmas = linesigmas / C_LIGHT / jnp.log(10)
    linezwaves = jnp.log10(linewaves * (1.0 + redshift + linevshifts / C_LIGHT))

    # contributions of each peak at each point
    Y = lineamps * jnp.exp(-0.5 * (X.T - linezwaves)**2 / log10sigmas**2)

    # sum over peaks
    log10model = jnp.sum(Y, axis=1)

    #emlinemodel = trapz_rebin(10**log10wave, log10model, emlinewave)
    emlinemodel = jnp.interp(emlinewave, 10**log10wave, log10model)
    # missing step -- apply the resolution matrix

    return 10**log10wave, log10model, emlinemodel

  def _objective_function_gpu(free_parameters, emlinewave, emlineflux, weights,
                        redshift, log10wave, parameters, Ifree, linewaves, X):
    """The parameters array should only contain free (not tied or fixed) parameters."""

    # Parameters have to be allowed to exceed their bounds in the optimization
    # function, otherwise they get stuck at the boundary.

    #print(free_parameters)
    #parameters = parameters.at[Ifree].set(free_parameters)
    lineamps = free_parameters[:nline]
    linevshifts = jnp.zeros_like(lineamps) + free_parameters[nline]
    linesigmas = jnp.zeros_like(lineamps) + free_parameters[nline+1]

    _, _, emlinemodel = build_emline_model_gpu(lineamps, linevshifts, linesigmas, linewaves,
                                            redshift, log10wave, emlinewave, X)
    
    residuals = weights * (emlineflux - emlinemodel)
    return 0.5 * jnp.sum(residuals**2)

  #Oversampled wavelength arrray
  dlog10wave = .5 / C_LIGHT / jnp.log(10)
  log10wave = jnp.arange(jnp.log10(jnp.min(emlinewave)), jnp.log10(jnp.max(emlinewave)), dlog10wave)
  X = jnp.broadcast_to(log10wave, (nline, len(log10wave)))

  #Statistical Weights
  weights = jnp.sqrt(emlineivar)

  # initial guesses
  linevshifts = jnp.array([0])
  linesigmas = jnp.array([45])
  lineamps = jnp.zeros(nline) + 10.

  #Parameters for the optimizer
  parameters = jnp.hstack([lineamps, linevshifts, linesigmas])
  Ifree = jnp.arange(len(parameters))
  free_parameters = parameters[Ifree]
  farg = emlinewave, emlineflux, weights, redshift, log10wave, parameters , Ifree, linewaves, X
  bounds =  jnp.array([([-jnp.inf] *(nline+2)), ([+jnp.inf] *(nline+2))])
    
  lbgfsb = ScipyBoundedMinimize(fun=_objective_function_gpu, method="L-BFGS-B", dtype=jnp.float64, jit = True)
  lbgfsb_sol = lbgfsb.run(parameters, bounds, *farg)
  params = lbgfsb_sol.params
  bestamps = params[:nline]
  bestvshifts = jnp.zeros_like(lineamps) + params[nline]
  bestsigmas = jnp.zeros_like(lineamps) + params[nline+1]
  #_, _, bestmodel = build_emline_model_gpu(bestamps, bestvshifts, bestsigmas, linewaves, redshift, log10wave, emlinewave, X)
  

  return bestamps, bestvshifts, bestsigmas, #bestmodel