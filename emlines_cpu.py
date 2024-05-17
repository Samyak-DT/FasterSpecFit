import numpy as np
from scipy.optimize import least_squares

C_LIGHT = 299792.458


def emlines(data, linewaves, redshift):

    emlinewave = np.asarray(data[0])
    emlineflux = np.asarray(data[1])
    emlineivar = np.asarray(data[2])
    linewaves = np.asarray(linewaves)
    nline = len(linewaves)

    def build_emline_model(lineamps, linevshifts, linesigmas, linewaves,
                           redshift, log10wave, emlinewave, X):

        #line-width and redshifted wavelength
        log10sigmas = linesigmas / C_LIGHT / np.log(10)
        linezwaves = np.log10(linewaves * (1.0 + redshift + linevshifts / C_LIGHT))

        # contributions of each peak at each point
        Y = lineamps * np.exp(-0.5 * (X.T - linezwaves)**2 / log10sigmas**2)

        # sum over peaks
        log10model = np.sum(Y, axis=1)

        #emlinemodel = trapz_rebin(10**log10wave, log10model, emlinewave)
        emlinemodel = np.interp(emlinewave, 10**log10wave, log10model)
        

        return 10**log10wave, log10model, emlinemodel



    def _objective_function(free_parameters, emlinewave, emlineflux, weights, redshift,
                            log10wave, parameters, linewaves, X):

        nline = len(linewaves)
        lineamps = free_parameters[:nline]
        linevshifts = np.zeros_like(lineamps) + free_parameters[nline]
        linesigmas = np.zeros_like(lineamps) + free_parameters[nline+1]
        _, _, emlinemodel = build_emline_model(lineamps, linevshifts, linesigmas, linewaves,
                            redshift, log10wave, emlinewave, X)
        
        residuals = weights * (emlinemodel - emlineflux)
        
        return residuals

    # create the oversampled (constant-velocity) wavelength array
    dlog10wave = 5. / C_LIGHT / np.log(10)
    log10wave = np.arange(np.log10(np.min(emlinewave)), np.log10(np.max(emlinewave)), dlog10wave)
    X = np.broadcast_to(log10wave, (nline, len(log10wave)))

    #Statistical weights
    weights = np.sqrt(emlineivar)

    #Initial parameters
    linevshift = np.array([0.])
    linesigma = np.array([75.])
    lineamps = np.zeros(nline) + 20.
    parameters = np.hstack([lineamps, linevshift, linesigma])
    farg = emlinewave, emlineflux, weights, redshift, log10wave, parameters , linewaves, X
    bounds_min = np.hstack([[0.] * nline + [-100.] + [0.]]).tolist()
    bounds_max = np.hstack([[1e3] * nline + [+100.] + [500.]]).tolist()
    bounds = [bounds_min, bounds_max]

    print(bounds)

    #Optimizer and Solutions
    #fit_info = least_squares(_objective_function, parameters, args=farg, max_nfev=100, xtol=1e-1, method='lm', verbose=2)
    fit_info = least_squares(_objective_function, parameters, bounds=bounds, args=farg, max_nfev=100,
                             xtol=1e-8, x_scale='jac', 
                             method='trf', tr_solver='lsmr', tr_options={'regularize': True})
    emline_sol = fit_info.x
    
    #bestamps, bestvshifts, bestsigmas = np.array_split(emline_sol, 3)
    bestamps = emline_sol[:nline]
    bestvshifts = np.zeros_like(lineamps) + emline_sol[nline]
    bestsigmas = np.zeros_like(lineamps) + emline_sol[nline+1]

    # Buiding out the model
    #_, _, bestmodel = build_emline_model(bestamps, bestvshifts, bestsigmas, linewaves, redshift, log10wave, emlinewave, X)

    return bestamps, bestvshifts, bestsigmas, #bestmodel
