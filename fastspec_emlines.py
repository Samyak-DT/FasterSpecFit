#!/usr/bin/env python

"""Toy version of fastspec with simplified I/O and no stellar-continuum fitting
which will allow us to benchmark updated and sped-up emission-line fitting
algorithms.

Dependencies:
* astropy
* fitsio
* speclite
* desispec
* desiutil
* desitarget
* desimodel
* fastspecfit

time python code/desihub/FasterSpecFit/FasterSpecFit/fastspec_emlines.py
time python code/desihub/FasterSpecFit/FasterSpecFit/fastspec_emlines.py --fast

"""
import os, time, pdb
import numpy as np

from desiutil.log import get_logger, INFO
log = get_logger(INFO)

from FasterSpecFit import ResMatrix

def read_test_data(datadir='.', ntargets=None):
    """Read the test data.

    """
    import fitsio
    from astropy.table import Table, hstack
    from desispec.io import read_spectra
    from desispec.resolution import Resolution
    from desispec.coaddition import coadd_cameras
    from fastspecfit.continuum import ContinuumTools

    # methods for dealing with the stellar continuum
    CTools = ContinuumTools()
        
    # read the data - fastspecfit.io.DESISpectra().select()
    coaddfile = os.path.join(datadir, 'coadd-test-data2.fits')
    redrockfile = os.path.join(datadir, 'redrock-test-data2.fits')
    #fastfile = os.path.join(datadir, 'fastspec-test-data2.fits')

    spec = read_spectra(coaddfile)

    zb = Table(fitsio.read(redrockfile, ext='REDSHIFTS', columns=['TARGETID', 'Z', 'ZWARN', 'SPECTYPE', 'DELTACHI2']))
    fm = Table(fitsio.read(redrockfile, ext='FIBERMAP', columns=['TARGETID', 'TARGET_RA', 'TARGET_DEC']))
    assert(np.all(zb['TARGETID'] == fm['TARGETID']))
    zb.remove_column('TARGETID')
    meta = hstack((fm, zb))#, tsnr2))    

    if ntargets is not None and ntargets <= len(meta):
        meta = meta[:ntargets]

    # pack into a dictionary - fastspecfit.io.DESISpectra().read_and_unpack()
    
    # Coadd across cameras.
    coadd_spec = coadd_cameras(spec)
    
    # unpack the desispec.spectra.Spectra objects into simple arrays
    cameras = spec.bands                # ['b', 'r', 'z']
    coadd_cameras = coadd_spec.bands[0] # 'brz'
    
    data = []
    for iobj in np.arange(len(meta)):
        specdata = {
            'uniqueid': meta['TARGETID'][iobj], # meta[CTools.uniqueid][iobj],
            'zredrock': meta['Z'][iobj],
            #'photsys': photsys[iobj],
            'cameras': cameras,
            #'dluminosity': dlum[iobj], 'dmodulus': dmod[iobj], 'tuniv': tuniv[iobj],
            'wave': [],
            'flux': [],
            'ivar': [],
            'mask': [],
            'res': [],
            'res_fast': [],
            'linemask': [], 
            'linemask_all': [], 
            'linename': [],
            'linepix': [], 
            'contpix': [],
            'coadd_wave': coadd_spec.wave[coadd_cameras],
            'coadd_flux': coadd_spec.flux[coadd_cameras][iobj, :],
            'coadd_ivar': coadd_spec.ivar[coadd_cameras][iobj, :],
            'coadd_res': Resolution(coadd_spec.resolution_data[coadd_cameras][iobj, :]),
            }

        npixpercamera = []
        for cam in cameras:
            wave = spec.wave[cam]
            flux = spec.flux[cam][iobj, :]
            ivar = spec.ivar[cam][iobj, :]
            mask = spec.mask[cam][iobj, :]
            res_numpy = spec.resolution_data[cam][iobj, :, :]
            res = Resolution(res_numpy)
            res_fast = ResMatrix(res_numpy)
            
            # this is where we would correct for dust
            # ...

            # always mask the first and last pixels
            mask[0] = 1
            mask[-1] = 1

            # In the pipeline, if mask!=0 that does not mean ivar==0, but we
            # want to be more aggressive about masking here.
            ivar[mask != 0] = 0

            specdata['wave'].append(wave)
            specdata['flux'].append(flux)
            specdata['ivar'].append(ivar)
            specdata['mask'].append(mask)
            specdata['res'].append(res)
            specdata['res_fast'].append(res_fast)
            
            npixpercamera.append(len(wave)) # number of pixels in this camera
            
        # Pre-compute some convenience variables for "un-hstacking"
        # an "hstacked" spectrum.
        specdata['npixpercamera'] = npixpercamera

        ncam = len(specdata['cameras'])
        npixpercam = np.hstack([0, npixpercamera])
        specdata['camerapix'] = np.zeros((ncam, 2), np.int16)
        for icam in np.arange(ncam):
            specdata['camerapix'][icam, :] = [np.sum(npixpercam[:icam+1]), np.sum(npixpercam[:icam+2])]

        # build the pixel list of emission lines
        coadd_linemask_dict = CTools.build_linemask(specdata['coadd_wave'], specdata['coadd_flux'],
                                                    specdata['coadd_ivar'], redshift=specdata['zredrock'],
                                                    linetable=CTools.linetable)
        specdata['coadd_linename'] = coadd_linemask_dict['linename']
        specdata['coadd_linepix'] = [np.where(lpix)[0] for lpix in coadd_linemask_dict['linepix']]
        specdata['coadd_contpix'] = [np.where(cpix)[0] for cpix in coadd_linemask_dict['contpix']]
    
        specdata['linesigma_narrow'] = coadd_linemask_dict['linesigma_narrow']
        specdata['linesigma_balmer'] = coadd_linemask_dict['linesigma_balmer']
        specdata['linesigma_uv'] = coadd_linemask_dict['linesigma_uv']
    
        specdata['linesigma_narrow_snr'] = coadd_linemask_dict['linesigma_narrow_snr']
        specdata['linesigma_balmer_snr'] = coadd_linemask_dict['linesigma_balmer_snr']
        specdata['linesigma_uv_snr'] = coadd_linemask_dict['linesigma_uv_snr']

        specdata['smoothsigma'] = coadd_linemask_dict['smoothsigma']
        
        # Map the pixels belonging to individual emission lines and
        # their local continuum back onto the original per-camera
        # spectra. These lists of arrays are used in
        # continuum.ContinnuumTools.smooth_continuum.
        for icam in np.arange(len(specdata['cameras'])):
            #specdata['smoothflux'].append(np.interp(specdata['wave'][icam], specdata['coadd_wave'], coadd_linemask_dict['smoothflux']))
            specdata['linemask'].append(np.interp(specdata['wave'][icam], specdata['coadd_wave'], coadd_linemask_dict['linemask']*1) > 0)
            specdata['linemask_all'].append(np.interp(specdata['wave'][icam], specdata['coadd_wave'], coadd_linemask_dict['linemask_all']*1) > 0)
            _linename, _linenpix, _contpix = [], [], []
            for ipix in np.arange(len(coadd_linemask_dict['linepix'])):
                I = np.interp(specdata['wave'][icam], specdata['coadd_wave'], coadd_linemask_dict['linepix'][ipix]*1) > 0
                J = np.interp(specdata['wave'][icam], specdata['coadd_wave'], coadd_linemask_dict['contpix'][ipix]*1) > 0
                if np.sum(I) > 3 and np.sum(J) > 3:
                    _linename.append(coadd_linemask_dict['linename'][ipix])
                    _linenpix.append(np.where(I)[0])
                    _contpix.append(np.where(J)[0])
            specdata['linename'].append(_linename)
            specdata['linepix'].append(_linenpix)
            specdata['contpix'].append(_contpix)

        specdata.update({'coadd_linemask': coadd_linemask_dict['linemask'],
                         'coadd_linemask_all': coadd_linemask_dict['linemask_all']})

        data.append(specdata)

    return data


def emfit_optimize(emfit, linemodel, emlinewave, emlineflux, weights, redshift,
                   resolution_matrix, resolution_matrix_fast,
                   camerapix, log=None, get_finalamp=False,
                   verbose=False, debug=False, fast=False):

    from scipy.optimize import least_squares

    if fast:
        from FasterSpecFit import EMLine_prepare_bins, EMLine_ParamsMapping
        from FasterSpecFit import EMLine_objective as objective
        from FasterSpecFit import EMLine_jacobian  as jacobian
        from FasterSpecFit import EMLine_find_peak_amplitudes
    else:
        from fastspecfit.emlines import _objective_function as objective

    if log is None:
        from desiutil.log import get_logger, DEBUG
        if verbose:
            log = get_logger(DEBUG)
        else:
            log = get_logger()
    
    parameters, (Ifree, Itied, tiedtoparam, tiedfactor, bounds, doubletindx, doubletpair, \
                 linewaves) = emfit._linemodel_to_parameters(linemodel, emfit.fit_linetable)
    log.debug(f'Optimizing {len(Ifree)} free parameters')
    
    # corner case where all lines are out of the wavelength range, which can
    # happen at high redshift and with the red camera masked, e.g.,
    # iron/main/dark/6642/39633239580608311).
    initial_guesses = parameters[Ifree]
    
    t0 = time.time()

    # The only difference between the old and new emline fitting is in the
    # arguments passed to the least_squares method
    if fast:
        
        bin_data = EMLine_prepare_bins(emlinewave, camerapix)
        
        params_mapping = EMLine_ParamsMapping(parameters, Ifree,
                                              Itied, tiedtoparam, tiedfactor,
                                              doubletindx, doubletpair)
        
        farg = (bin_data, emlineflux, weights, redshift,
                linewaves, tuple(resolution_matrix_fast), camerapix,
                params_mapping)
        
        jac = jacobian
        
        """
        # validate the correctness of the Jacobian
        J1 = jacobian(initial_guesses, *farg).dot(np.eye(len(initial_guesses)))
        J0 = np.empty(J1.shape)
        for i in range(J1.shape[1]):
            ig1 = initial_guesses.copy()
            ig1[i] -= 0.00001
            ig2 = initial_guesses.copy()
            ig2[i] += 0.00001
            vp1 = objective(ig1, *farg)
            vp2 = objective(ig2, *farg)
            J0[:,i] = (vp2 - vp1)/(2*0.00001)
            print(i, np.max(np.abs(J0[:,i] - J1[:,i])))
            #for j in range(J1.shape[0]):
            #    if np.abs(J0[j][i] - J1[j][i]) > 1e-6:
            #        print(j, i, J0[j,i], J1[j,i])
        """
    else:
        
        farg = (emlinewave, emlineflux, weights, redshift, emfit.dlog10wave, 
                resolution_matrix, camerapix, parameters, ) + \
                (Ifree, Itied, tiedtoparam, tiedfactor, doubletindx, 
                 doubletpair, linewaves)
        
        jac = "2-point"
        
    if len(Ifree) == 0:
        fit_info = {'nfev': 0, 'status': 0}
    else:
        
        FIT_TRIES = 2
        for fit_try in range(FIT_TRIES):
            try:
                fit_info = least_squares(objective, initial_guesses, jac=jac, args=farg, max_nfev=5000, 
                                         xtol=1e-10, #ftol=1e-5, #x_scale='jac' gtol=1e-10,
                                         tr_solver='lsmr', tr_options={'maxiter': 1000, 'regularize': True},
                                         method='trf', bounds=tuple(zip(*bounds)),) # verbose=2)
                parameters[Ifree] = fit_info.x
            except:
                if emfit.uniqueid:
                    errmsg = f'Problem in scipy.optimize.least_squares for {emfit.uniqueid}.'
                else:
                    errmsg = 'Problem in scipy.optimize.least_squares.'
                log.critical(errmsg)
                raise RuntimeError(errmsg)
            
            if fit_try < FIT_TRIES - 1:
                # If the narrow-line sigma didn't change by more than ~one km/s from
                # its initial guess, then something has gone awry, so perturb the
                # initial guess by 10% and try again. Examples where this matters:
                #   fuji-sv3-bright-28119-39628390055022140
                #   fuji-sv3-dark-25960-1092092734472204
                S = np.where(emfit.sigma_param_bool[Ifree] * (linemodel['isbroad'][Ifree] == False))[0]
                sig_init = initial_guesses[S]
                sig_final = parameters[Ifree][S]
                G = np.abs(sig_init - sig_final) < 1.
                
                if G.any():
                    log.warning(f'Poor convergence on line-sigma for {emfit.uniqueid}; perturbing initial guess and refitting.')
                    initial_guesses[S[G]] *= 0.9
                else:
                    break
            
    t1 = time.time()
    
    # drop (zero out) any dubious free parameters
    drop_params(parameters, emfit, linemodel, Ifree)
    
    # at this point, parameters contains correct *free* and *fixed* values,
    # but we need to update *tied* values to reflect any changes to free
    # params.  We do *not* apply doublet rules, as other code expects
    # us to return a params array with doublet ratios as ratios, not amplitudes.
    if len(Itied) > 0:
        for I, indx, factor in zip(Itied, tiedtoparam, tiedfactor):
            parameters[I] = parameters[indx] * factor
    
    out_linemodel = linemodel.copy()
    out_linemodel['value'] = parameters.copy() # so we don't munge it below
    out_linemodel.meta['nfev'] = fit_info['nfev']
    out_linemodel.meta['status'] = fit_info['status']
    
    if get_finalamp:
        
        lineamps, linevshifts, linesigmas = np.array_split(parameters, 3) # 3 parameters per line

        # apply doublet rules
        lineamps[doubletindx] *= lineamps[doubletpair]
        
        if fast:

            #
            # calculate the observed maximum amplitude for each
            # fitted spectral line after convolution with the resolution
            # matrix.
            #
            peaks = EMLine_find_peak_amplitudes(parameters,
                                                bin_data,
                                                redshift,
                                                linewaves,
                                                resolution_matrix_fast,
                                                camerapix)
            
            out_linemodel['obsvalue'][:len(lineamps)] = peaks
            
        else:
            from fastspecfit.util import trapz_rebin
            
            # Get the final line-amplitudes, after resampling and convolution (see
            # https://github.com/desihub/fastspecfit/issues/139). Some repeated code
            # from build_emline_model...
            
            lineindxs = np.arange(len(lineamps))

            I = lineamps > 0
            if np.count_nonzero(I) > 0:
                linevshifts = linevshifts[I]
                linesigmas = linesigmas[I]
                lineamps = lineamps[I]
                linewaves = linewaves[I]
                lineindxs = lineindxs[I]
            
                # demand at least 20 km/s for rendering the model
                if np.any(linesigmas) < 20.:
                    linesigmas[linesigmas<20.] = 20.
                
                if camerapix is None:
                    minwave = emlinewave[0][0]-2.
                    maxwave = emlinewave[-1][-1]+2.
                else:
                    minwave = emlinewave[0]-2.
                    maxwave = emlinewave[-1]+2.
            
                _emlinewave = []
                for icam, campix in enumerate(camerapix):
                    _emlinewave.append(emlinewave[campix[0]:campix[1]])

                # line-width [log-10 Angstrom] and redshifted wavelength [log-10 Angstrom]
                C_LIGHT = 299792.458
                log10sigmas = linesigmas / C_LIGHT / np.log(10)                
                linezwaves = np.log10(linewaves * (1.0 + redshift + linevshifts / C_LIGHT))
            
                for lineindx, lineamp, linezwave, log10sigma in zip(lineindxs, lineamps, linezwaves, log10sigmas):
                    log10wave = np.arange(linezwave - (5 * log10sigma), linezwave + (5 * log10sigma), emfit.dlog10wave)
                    log10wave = np.hstack((np.log10(minwave), log10wave, np.log10(maxwave)))
                    log10model = lineamp * np.exp(-0.5 * (log10wave-linezwave)**2 / log10sigma**2)
                    # Determine which camera we're on and then resample and
                    # convolve with the resolution matrix.
                    icam = np.argmin([np.abs((np.max(emwave)-np.min(emwave))/2+np.min(emwave)-10**linezwave) for emwave in _emlinewave])
                    model_resamp = trapz_rebin(10**log10wave, log10model, _emlinewave[icam])
                                        
                    model_convol = resolution_matrix[icam].dot(model_resamp)
                    out_linemodel['obsvalue'][lineindx] = np.max(model_convol)

        """
        peaks0 = out_linemodel['obsvalue'][lineindxs].data
        peaks1 = peaks[lineindxs]

        for p0, p1 in zip(peaks0, peaks1):
            if np.abs(p1 - p0) / np.maximum(p0, p1) > 1e-2:
                print(p0, p1)

        """
        
        """
        import matplotlib.pyplot as plt
        bestfit = emfit.bestfit(out_linemodel, redshift, emlinewave, resolution_matrix, camerapix)
        plt.clf()
        plt.plot(emlinewave, emlineflux, label='data', color='gray', lw=4)
        plt.plot(emlinewave, bestfit, label='bestfit', ls='--', lw=3, alpha=0.7, color='k')
        plt.legend()
        plt.savefig('fit.png')
        """
            
    return out_linemodel, (t1 - t0)


#
# drop_parameters()
# Drop dubious free parameters after fitting
#
def drop_params(parameters, emfit, linemodel, Ifree):
    
    # Conditions for dropping a parameter (all parameters, not just those
    # being fitted):
    # --negative amplitude or sigma
    # --parameter at its default value (fit failed, right??)
    # --parameter within 0.1% of its bounds
    lineamps, linevshifts, linesigmas = np.array_split(parameters, 3) # 3 parameters per line
    notfixed = np.logical_not(linemodel['fixed'])

    # drop any negative amplitude or sigma parameter that is not fixed 
    drop1 = np.hstack((lineamps < 0, np.zeros(len(linevshifts), bool), linesigmas <= 0)) * notfixed
    
    # Require equality, not np.isclose, because the optimization can be very
    # small (<1e-6) but still significant, especially for the doublet
    # ratios. If linesigma is dropped this way, make sure the corresponding
    # line-amplitude is dropped, too (see MgII 2796 on
    # sv1-bright-17680-39627622543528153).
    drop2 = np.zeros(len(parameters), bool)
        
    # if any amplitude is zero, drop the corresponding sigma and vshift
    amp_param_bool = emfit.amp_param_bool[Ifree]
    I = np.where(parameters[Ifree][amp_param_bool] == 0.)[0]
    if len(I) > 0:
        _Ifree = np.zeros(len(parameters), bool)
        _Ifree[Ifree] = True
        for pp in linemodel[Ifree][amp_param_bool][I]['param_name']:
            J = np.where(_Ifree * (linemodel['param_name'] == pp.replace('_amp', '_sigma')))[0]
            drop2[J] = True
            K = np.where(_Ifree * (linemodel['param_name'] == pp.replace('_amp', '_vshift')))[0]
            drop2[K] = True
            #print(pp, J, K, np.sum(drop2))

    # drop amplitudes for any lines tied to a line with a dropped sigma
    sigmadropped = np.where(emfit.sigma_param_bool * drop2)[0]
    for lineindx, dropline in zip(sigmadropped, linemodel[sigmadropped]['linename']):
        # Check whether lines are tied to this line. If so, find the
        # corresponding amplitude and drop that, too.
        T = linemodel['tiedtoparam'] == lineindx
        for tiedline in set(linemodel['linename'][T]):
            drop2[linemodel['param_name'] == f'{tiedline}_amp'] = True
        drop2[linemodel['param_name'] == f'{dropline}_amp'] = True

    # drop amplitudes for any lines tied to a line with a dropped vshift
    vshiftdropped = np.where(emfit.vshift_param_bool * drop2)[0]
    for lineindx, dropline in zip(vshiftdropped, linemodel[vshiftdropped]['linename']):
        # Check whether lines are tied to this line. If so, find the
        # corresponding amplitude and drop that, too.
        T = linemodel['tiedtoparam'] == lineindx
        for tiedline in set(linemodel['linename'][T]):
            drop2[linemodel['param_name'] == f'{tiedline}_amp'] = True
        drop2[linemodel['param_name'] == f'{dropline}_amp'] = True

    # drop any non-fixed parameters outside their bounds
    # It's OK for parameters to be *at* their bounds.
    drop3 = np.zeros(len(parameters), bool)
    drop3[Ifree] = np.logical_or(parameters[Ifree] < linemodel['bounds'][Ifree, 0], 
                                 parameters[Ifree] > linemodel['bounds'][Ifree, 1])
    drop3 *= notfixed
    
    log.debug(f'Dropping {np.sum(drop1)} negative-amplitude lines.') # linewidth can't be negative
    log.debug(f'Dropping {np.sum(drop2)} sigma,vshift parameters of zero-amplitude lines.')
    log.debug(f'Dropping {np.sum(drop3)} parameters which are out-of-bounds.')
    Idrop = np.where(np.logical_or.reduce((drop1, drop2, drop3)))[0]
    
    if len(Idrop) > 0:
        log.debug(f'  Dropping {len(Idrop)} unique parameters.')
        parameters[Idrop] = 0.0

    # Now loop back through and drop Broad balmer lines that:
    #   (1) are narrower than their narrow-line counterparts;
    #   (2) have a narrow line whose amplitude is smaller than that of the broad line
    #      --> Deprecated! main-dark-32303-39628176678192981 is an example
    #          of an object where there's a broad H-alpha line but no other
    #          forbidden lines!

        
def fit_emlines(datadir='.', fast=False, ntargets=None):
    """Use the current (main) version of the emission-line fitting code.

    """
    from astropy.table import Table, vstack
    from fastspecfit.emlines import EMFitTools

    t0 = time.time()    
    specdata = read_test_data(datadir=datadir, ntargets=ntargets)
    t1 = time.time()
    log.info(f'Gathering the data took {t1-t0:.2f} seconds.')

    # loop on each spectrum
    for iobj, data in enumerate(specdata):
        EMFit = EMFitTools(uniqueid=data['uniqueid'])

        # Combine all three cameras; we will unpack them to build the
        # best-fitting model (per-camera) below.
        redshift = data['zredrock']
        emlinewave = data['wave']
        emlineflux = data['flux'] # we already subtracted the continuum for this test
        emlinewave = np.hstack(emlinewave)
        emlineflux = np.hstack(emlineflux)
        
        # Build all the emission-line models for this object.
        linemodel_broad, linemodel_nobroad = EMFit.build_linemodels(
            redshift, wavelims=(np.min(emlinewave)+5, np.max(emlinewave)-5),
            verbose=False, strict_broadmodel=True)

        # Get initial guesses on the parameters and populate the two "initial"
        # linemodels; the "final" linemodels will be initialized with the
        # best-fitting parameters from the initial round of fitting.
        initial_guesses, param_bounds = EMFit.initial_guesses_and_bounds(
            data, emlinewave, emlineflux, log=log)
        
        EMFit.populate_linemodel(linemodel_nobroad, initial_guesses, param_bounds, log=log)
        EMFit.populate_linemodel(linemodel_broad, initial_guesses, param_bounds, log=log)
        
        emlineivar = np.hstack(data['ivar'])
        camerapix = data['camerapix']
        resolution_matrix = data['res']
        resolution_matrix_fast = data['res_fast']
        
        weights = np.sqrt(emlineivar)
        
        # run once without timing to ensure that JIT happens

        if fast and iobj == 0:
            emfit_optimize(EMFit, linemodel_nobroad, emlinewave, emlineflux, weights, redshift,
                           resolution_matrix, resolution_matrix_fast, camerapix, log=log, debug=False, get_finalamp=True,
                           fast=fast)
            
        fit_nobroad, t_elapsed = emfit_optimize(EMFit, linemodel_nobroad, emlinewave, emlineflux, weights, redshift,
                                                resolution_matrix, resolution_matrix_fast, camerapix, log=log, debug=False, get_finalamp=True,
                                                fast=fast)
        
        model_nobroad = EMFit.bestfit(fit_nobroad, redshift, emlinewave, resolution_matrix, camerapix)
        chi2_nobroad, ndof_nobroad, nfree_nobroad = EMFit.chi2(fit_nobroad, emlinewave, emlineflux, emlineivar, model_nobroad, return_dof=True)
        log.info(f'{iobj}: line-fitting with no broad lines and {nfree_nobroad} free parameters took {t_elapsed:.3f} seconds '
                 f'[niter={fit_nobroad.meta["nfev"]}, rchi2={chi2_nobroad:.4f}].')


        # Now try adding broad Balmer and helium lines and see if we improve the
        # chi2.
        if True:
            # Gather the pixels around the broad Balmer lines and the corresponding
            # linemodel table.
            balmer_pix, balmer_linemodel_broad, balmer_linemodel_nobroad = [], [], []
            for icam in np.arange(len(data['cameras'])):
                pixoffset = int(np.sum(data['npixpercamera'][:icam]))
                if len(data['linename'][icam]) > 0:
                    I = (linemodel_nobroad['isbalmer'] * (linemodel_nobroad['ishelium'] == False) *
                         linemodel_nobroad['isbroad'] * np.isin(linemodel_nobroad['linename'], data['linename'][icam]))
                    _balmer_linemodel_broad = linemodel_broad[I]
                    _balmer_linemodel_nobroad = linemodel_nobroad[I]
                    balmer_linemodel_broad.append(_balmer_linemodel_broad)
                    balmer_linemodel_nobroad.append(_balmer_linemodel_nobroad)
                    if len(_balmer_linemodel_broad) > 0: # use balmer_linemodel_broad not balmer_linemodel_nobroad
                        I = np.where(np.isin(data['linename'][icam], _balmer_linemodel_broad['linename']))[0]
                        for ii in I:
                            #print(data['linename'][icam][ii])
                            balmer_pix.append(data['linepix'][icam][ii] + pixoffset)
                            
            if len(balmer_pix) > 0:
                fit_broad, t_elapsed = emfit_optimize(EMFit, linemodel_broad, emlinewave, emlineflux, weights, redshift,
                                                      resolution_matrix, resolution_matrix_fast, camerapix, log=log, debug=False, get_finalamp=True,
                                                      fast=fast)
        
                model_broad = EMFit.bestfit(fit_broad, redshift, emlinewave, resolution_matrix, camerapix)
                chi2_broad, ndof_broad, nfree_broad = EMFit.chi2(fit_broad, emlinewave, emlineflux, emlineivar, model_broad, return_dof=True)
                log.info(f'{iobj}: line-fitting with broad lines and {nfree_broad} free parameters took {t_elapsed:.3f} seconds '
                         f'[niter={fit_broad.meta["nfev"]}, rchi2={chi2_broad:.4f}].')
    
                # compute delta-chi2 around just the Balmer lines
                balmer_pix = np.hstack(balmer_pix)
                balmer_linemodel_broad = vstack(balmer_linemodel_broad)
    
                balmer_nfree_broad = (np.count_nonzero((balmer_linemodel_broad['fixed'] == False) *
                                                       (balmer_linemodel_broad['tiedtoparam'] == -1)))
                balmer_ndof_broad = np.count_nonzero(emlineivar[balmer_pix] > 0) - balmer_nfree_broad
    
                balmer_linemodel_nobroad = vstack(balmer_linemodel_nobroad)
                balmer_nfree_nobroad = (np.count_nonzero((balmer_linemodel_nobroad['fixed'] == False) *
                                                         (balmer_linemodel_nobroad['tiedtoparam'] == -1)))
                balmer_ndof_nobroad = np.count_nonzero(emlineivar[balmer_pix] > 0) - balmer_nfree_nobroad
    
                linechi2_balmer_broad = np.sum(emlineivar[balmer_pix] * (emlineflux[balmer_pix] - model_broad[balmer_pix])**2)
                linechi2_balmer_nobroad = np.sum(emlineivar[balmer_pix] * (emlineflux[balmer_pix] - model_nobroad[balmer_pix])**2)
                delta_linechi2_balmer = linechi2_balmer_nobroad - linechi2_balmer_broad
                delta_linendof_balmer = balmer_ndof_nobroad - balmer_ndof_broad
    
                # Choose broad-line model only if:
                # --delta-chi2 > delta-ndof
                # --broad_sigma < narrow_sigma
                # --broad_sigma < 250
    
                dchi2test = delta_linechi2_balmer > delta_linendof_balmer
                Hanarrow = fit_broad['param_name'] == 'halpha_sigma' # Balmer lines are tied to H-alpha even if out of range
                Habroad = fit_broad['param_name'] == 'halpha_broad_sigma'
                Bbroad = fit_broad['isbalmer'] * fit_broad['isbroad'] * (fit_broad['fixed'] == False) * EMFit.amp_balmer_bool
                broadsnr = fit_broad[Bbroad]['obsvalue'].data * np.sqrt(fit_broad[Bbroad]['civar'].data)
    
                sigtest1 = fit_broad[Habroad]['value'][0] > EMFit.minsigma_balmer_broad
                sigtest2 = (fit_broad[Habroad]['value'] > fit_broad[Hanarrow]['value'])[0]
                if len(broadsnr) == 0:
                    broadsnrtest = False
                    _broadsnr = 0.
                elif len(broadsnr) == 1:
                    broadsnrtest =  broadsnr[-1] > EMFit.minsnr_balmer_broad
                    _broadsnr = 'S/N ({}) = {:.1f}'.format(fit_broad[Bbroad]['linename'][-1], broadsnr[-1])
                else:
                    broadsnrtest =  np.any(broadsnr[-2:] > EMFit.minsnr_balmer_broad)
                    _broadsnr = 'S/N ({}) = {:.1f}, S/N ({}) = {:.1f}'.format(
                        fit_broad[Bbroad]['linename'][-2], broadsnr[-2], fit_broad[Bbroad]['linename'][-1], broadsnr[-1])
    
                if dchi2test and sigtest1 and sigtest2 and broadsnrtest:
                    log.info('Adopting broad-line model:')
                    log.info('  delta-chi2={:.1f} > delta-ndof={:.0f}'.format(delta_linechi2_balmer, delta_linendof_balmer))
                    log.info('  sigma_broad={:.1f} km/s, sigma_narrow={:.1f} km/s'.format(fit_broad[Habroad]['value'][0], fit_broad[Hanarrow]['value'][0]))
                    if _broadsnr:
                        log.info('  {} > {:.0f}'.format(_broadsnr, EMFit.minsnr_balmer_broad))
                    finalfit, finalmodel, finalchi2 = fit_broad, model_broad, chi2_broad
                else:
                    if dchi2test == False:
                        log.info('Dropping broad-line model: delta-chi2={:.1f} < delta-ndof={:.0f}'.format(
                            delta_linechi2_balmer, delta_linendof_balmer))
                    elif sigtest1 == False:
                        log.info('Dropping broad-line model: Halpha_broad_sigma {:.1f} km/s < {:.0f} km/s (delta-chi2={:.1f}, delta-ndof={:.0f}).'.format(
                            fit_broad[Habroad]['value'][0], EMFit.minsigma_balmer_broad, delta_linechi2_balmer, delta_linendof_balmer))
                    elif sigtest2 == False:
                        log.info('Dropping broad-line model: Halpha_broad_sigma {:.1f} km/s < Halpha_narrow_sigma {:.1f} km/s (delta-chi2={:.1f}, delta-ndof={:.0f}).'.format(
                            fit_broad[Habroad]['value'][0], fit_broad[Hanarrow]['value'][0], delta_linechi2_balmer, delta_linendof_balmer))
                    elif broadsnrtest == False:
                        log.info('Dropping broad-line model: {} < {:.0f}'.format(_broadsnr, EMFit.minsnr_balmer_broad))
                    finalfit, finalmodel, finalchi2 = fit_nobroad, model_nobroad, chi2_nobroad
            else:
                log.info('Insufficient Balmer lines to test the broad-line model.')
                finalfit, finalmodel, finalchi2 = fit_nobroad, model_nobroad, chi2_nobroad
                delta_linechi2_balmer, delta_linendof_balmer = 0, np.int32(0)


        # write out...
    


def main():
    """Main wrapper.

    """
    import argparse

    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('--fast', action='store_true', help='Build the refactored, fast emission-line fitting code.')
    parser.add_argument('--datadir', type=str, default="./data", help='I/O directory.')
    parser.add_argument('--ntargets', type=int, default=None, help='For testing, test on ntargets objects.')
    args = parser.parse_args()

    # disable mutithreaded linear algebra in numpy/scipy, which
    # actually makes fastspecfit slower
    os.environ["MKL_NUM_THREADS"]      = "1"
    os.environ["OPENBLAS_NUM_THREADS"] = "1"
    
    fit_emlines(datadir=args.datadir, fast=args.fast, ntargets=args.ntargets)
        

if __name__ == '__main__':
    main()


