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
* desimodel(?)
* fastspecfit

time python code/desihub/FasterSpecFit/FasterSpecFit/fastspec_emlines.py --build-test-data --ntargets=20
time python code/desihub/FasterSpecFit/FasterSpecFit/fastspec_emlines.py
time python code/desihub/FasterSpecFit/FasterSpecFit/fastspec_emlines.py --fast

"""
import os, time, pdb
import numpy as np

from desiutil.log import get_logger
log = get_logger()


def read_test_data(datadir='.'):
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
    coaddfile = os.path.join(datadir, 'coadd-test-data.fits')
    redrockfile = os.path.join(datadir, 'redrock-test-data.fits')
    #fastfile = os.path.join(datadir, 'fastspec-test-data.fits')

    spec = read_spectra(coaddfile)

    zb = Table(fitsio.read(redrockfile, ext='REDSHIFTS', columns=['TARGETID', 'Z', 'ZWARN', 'SPECTYPE', 'DELTACHI2']))
    fm = Table(fitsio.read(redrockfile, ext='FIBERMAP', columns=['TARGETID', 'TARGET_RA', 'TARGET_DEC']))
    assert(np.all(zb['TARGETID'] == fm['TARGETID']))
    zb.remove_column('TARGETID')
    meta = hstack((fm, zb))#, tsnr2))    

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
        
        specdata.update({'coadd_linemask': coadd_linemask_dict['linemask'],
                         'coadd_linemask_all': coadd_linemask_dict['linemask_all']})

        data.append(specdata)

    return data


def emfit_optimize(emfit, linemodel, emlinewave, emlineflux, weights, redshift,
                   resolution_matrix, camerapix, log=None, get_finalamp=False,
                   verbose=False, debug=False, fast=False):

    from scipy.optimize import least_squares
    
    if log is None:
        from desiutil.log import get_logger, DEBUG
        if verbose:
            log = get_logger(DEBUG)
        else:
            log = get_logger()
    
    parameters, (Ifree, Itied, tiedtoparam, tiedfactor, bounds, doubletindx, doubletpair, \
                 linewaves) = emfit._linemodel_to_parameters(linemodel, emfit.fit_linetable)
    log.debug('Optimizing {} free parameters'.format(len(Ifree)))

    # corner case where all lines are out of the wavelength range, which can
    # happen at high redshift and with the red camera masked, e.g.,
    # iron/main/dark/6642/39633239580608311).
    initial_guesses = parameters[Ifree]

    from fastspecfit.util import trapz_rebin
    if fast:
        from FasterSpecFit import centers_to_edges
        from FasterSpecFit import _objective as objective
        # also import Jacobian later on
    else:
        from fastspecfit.emlines import _objective_function as objective
    
    t0 = time.time()

    # The only difference between the old and new emline fitting is in the
    # arguments passed to the least_squares method
    if fast:

        obs_bin_edges = centers_to_edges(emlinewave, camerapix)
        farg = (obs_bin_edges, np.log(obs_bin_edges), emlineflux,
                weights, redshift, linewaves, resolution_matrix, camerapix, parameters, ) + \
                (Ifree, Itied, tiedtoparam, tiedfactor, doubletindx, doubletpair)
        jac = "2-point"
        
    else:
        
        farg = (emlinewave, emlineflux, weights, redshift, emfit.dlog10wave, 
                resolution_matrix, camerapix, parameters, ) + \
                (Ifree, Itied, tiedtoparam, tiedfactor, doubletindx, 
                 doubletpair, linewaves)
        jac = "2-point"
        
    if len(Ifree) == 0:
        fit_info = {'nfev': 0, 'status': 0}
    else:
        try:
            fit_info = least_squares(objective, initial_guesses, args=farg, max_nfev=5000, 
                                     xtol=1e-10, #x_scale='jac', #ftol=1e-10, gtol=1e-10,
                                     tr_solver='lsmr', tr_options={'regularize': True},
                                     method='trf', bounds=tuple(zip(*bounds)))#, verbose=2)
            parameters[Ifree] = fit_info.x
        except:
            if emfit.uniqueid:
                errmsg = 'Problem in scipy.optimize.least_squares for {}.'.format(emfit.uniqueid)
            else:
                errmsg = 'Problem in scipy.optimize.least_squares.'
            log.critical(errmsg)
            raise RuntimeError(errmsg)

        # If the narrow-line sigma didn't change by more than ~one km/s from
        # its initial guess, then something has gone awry, so perturb the
        # initial guess by 10% and try again. Examples where this matters:
        #   fuji-sv3-bright-28119-39628390055022140
        #   fuji-sv3-dark-25960-1092092734472204
        S = np.where(emfit.sigma_param_bool[Ifree] * (linemodel['isbroad'][Ifree] == False))[0]
        if len(S) > 0:
            sig_init = initial_guesses[S]
            sig_final = parameters[Ifree][S]
            G = np.abs(sig_init - sig_final) < 1.
            if np.any(G):
                log.warning(f'Poor convergence on line-sigma for {emfit.uniqueid}; perturbing initial guess and refitting.')
                initial_guesses[S[G]] *= 0.9
                try:
                    fit_info = least_squares(objective, initial_guesses, args=farg, max_nfev=5000, 
                                             xtol=1e-10, #x_scale='jac', #ftol=1e-10, gtol=1e-10,
                                             tr_solver='lsmr', tr_options={'regularize': True},
                                             method='trf', bounds=tuple(zip(*bounds)))#, verbose=2)
                    parameters[Ifree] = fit_info.x
                except:
                    if emfit.uniqueid:
                        errmsg = f'Problem in scipy.optimize.least_squares for {emfit.uniqueid}.'
                    else:
                        errmsg = 'Problem in scipy.optimize.least_squares.'
                    log.critical(errmsg)
                    raise RuntimeError(errmsg)

    t1 = time.time()
    
    # Conditions for dropping a parameter (all parameters, not just those
    # being fitted):
    # --negative amplitude or sigma
    # --parameter at its default value (fit failed, right??)
    # --parameter within 0.1% of its bounds
    lineamps, linevshifts, linesigmas = np.array_split(parameters, 3) # 3 parameters per line
    notfixed = np.logical_not(linemodel['fixed'])
    
    drop1 = np.hstack((lineamps < 0, np.zeros(len(linevshifts), bool), linesigmas <= 0)) * notfixed
    
    # Require equality, not np.isclose, because the optimization can be very
    # small (<1e-6) but still significant, especially for the doublet
    # ratios. If linesigma is dropped this way, make sure the corresponding
    # line-amplitude is dropped, too (see MgII 2796 on
    # sv1-bright-17680-39627622543528153).
    drop2 = np.zeros(len(parameters), bool)
    
    amp_param_bool = emfit.amp_param_bool[Ifree]
    I = np.where(parameters[Ifree][amp_param_bool] == 0)[0]
    if len(I) > 0:
        _Ifree = np.zeros(len(parameters), bool)
        _Ifree[Ifree] = True
        for pp in linemodel[Ifree][amp_param_bool][I]['param_name']:
            J = np.where(_Ifree * (linemodel['param_name'] == pp.replace('_amp', '_sigma')))[0]
            if len(J) > 0:
                drop2[J] = True
            K = np.where(_Ifree * (linemodel['param_name'] == pp.replace('_amp', '_vshift')))[0]
            if len(K) > 0:
                drop2[K] = True
            #print(pp, J, K, np.sum(drop2))
        
    sigmadropped = np.where(emfit.sigma_param_bool * drop2)[0]
    if len(sigmadropped) > 0:
        for lineindx, dropline in zip(sigmadropped, linemodel[sigmadropped]['linename']):
            # Check whether lines are tied to this line. If so, find the
            # corresponding amplitude and drop that, too.
            T = linemodel['tiedtoparam'] == lineindx
            if np.any(T):
                for tiedline in set(linemodel['linename'][T]):
                    drop2[linemodel['param_name'] == '{}_amp'.format(tiedline)] = True
            drop2[linemodel['param_name'] == '{}_amp'.format(dropline)] = True

    vshiftdropped = np.where(emfit.vshift_param_bool * drop2)[0]
    if len(vshiftdropped) > 0:
        for lineindx, dropline in zip(vshiftdropped, linemodel[vshiftdropped]['linename']):
            # Check whether lines are tied to this line. If so, find the
            # corresponding amplitude and drop that, too.
            T = linemodel['tiedtoparam'] == lineindx
            if np.any(T):
                for tiedline in set(linemodel['linename'][T]):
                    drop2[linemodel['param_name'] == '{}_amp'.format(tiedline)] = True
            drop2[linemodel['param_name'] == '{}_amp'.format(dropline)] = True

    # It's OK for parameters to be *at* their bounds.
    drop3 = np.zeros(len(parameters), bool)
    drop3[Ifree] = np.logical_or(parameters[Ifree] < linemodel['bounds'][Ifree, 0], 
                                 parameters[Ifree] > linemodel['bounds'][Ifree, 1])
    drop3 *= notfixed
        
    log.debug('Dropping {} negative-amplitude lines.'.format(np.sum(drop1))) # linewidth can't be negative
    log.debug('Dropping {} sigma,vshift parameters of zero-amplitude lines.'.format(np.sum(drop2)))
    log.debug('Dropping {} parameters which are out-of-bounds.'.format(np.sum(drop3)))
    Idrop = np.where(np.logical_or.reduce((drop1, drop2, drop3)))[0]
        
    if len(Idrop) > 0:
        log.debug('  Dropping {} unique parameters.'.format(len(Idrop)))
        parameters[Idrop] = 0.0
            
    # apply tied constraints
    if len(Itied) > 0:
        for I, indx, factor in zip(Itied, tiedtoparam, tiedfactor):
            parameters[I] = parameters[indx] * factor
        
    # Now loop back through and drop Broad balmer lines that:
    #   (1) are narrower than their narrow-line counterparts;
    #   (2) have a narrow line whose amplitude is smaller than that of the broad line
    #      --> Deprecated! main-dark-32303-39628176678192981 is an example
    #          of an object where there's a broad H-alpha line but no other
    #          forbidden lines!

    out_linemodel = linemodel.copy()
    out_linemodel['value'] = parameters
    out_linemodel.meta['nfev'] = fit_info['nfev']
    out_linemodel.meta['status'] = fit_info['status']

    # Get the final line-amplitudes, after resampling and convolution (see
    # https://github.com/desihub/fastspecfit/issues/139). Some repeated code
    # from build_emline_model...
    if get_finalamp:
        lineamps, linevshifts, linesigmas = np.array_split(parameters, 3) # 3 parameters per line
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

            C_LIGHT = 299792.458
            # line-width [log-10 Angstrom] and redshifted wavelength [log-10 Angstrom]
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
                
    return out_linemodel, (t1 - t0)


def fit_emlines(datadir='.', fast=False):
    """Use the current (main) version of the emission-line fitting code.

    """
    from fastspecfit.emlines import EMFitTools

    t0 = time.time()    
    specdata = read_test_data(datadir=datadir)
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
        #EMFit.populate_linemodel(linemodel_broad, initial_guesses, param_bounds, log=log)
        
        emlineivar = np.hstack(data['ivar'])
        camerapix = data['camerapix']
        resolution_matrix = data['res']
        
        weights = np.sqrt(emlineivar)
        
        # run once without timing to ensure that JIT happens

        if fast and iobj == 0:
            emfit_optimize(EMFit, linemodel_nobroad, emlinewave, emlineflux, weights, redshift,
                           resolution_matrix, camerapix, log=log, debug=False, get_finalamp=True,
                           fast=fast)
            
        fit_nobroad, t_elapsed = emfit_optimize(EMFit, linemodel_nobroad, emlinewave, emlineflux, weights, redshift,
                                                resolution_matrix, camerapix, log=log, debug=False, get_finalamp=True,
                                                fast=fast)
        
        model_nobroad = EMFit.bestfit(fit_nobroad, redshift, emlinewave, resolution_matrix, camerapix)
        chi2_nobroad, ndof_nobroad, nfree_nobroad = EMFit.chi2(fit_nobroad, emlinewave, emlineflux, emlineivar, model_nobroad, return_dof=True)
        log.info(f'{iobj}: line-fitting with no broad lines and {nfree_nobroad} free parameters took {t_elapsed:.4f} seconds '
                 f'[niter={fit_nobroad.meta["nfev"]}, rchi2={chi2_nobroad:.4f}].')
                
        # write out...

    
def build_test_data(datadir='./', ntargets=None):
    """Build the test dataset; will only work at NERSC if run by Moustakas.

    """
    import fitsio
    from astropy.table import Table
    from desiutil.dust import dust_transmission
    from desispec.io import read_spectra, write_spectra
    from desispec.io.util import replace_prefix    
    from desitarget import geomask
    from fastspecfit.io import write_fastspecfit

    out_coaddfile = os.path.join(datadir, 'coadd-test-data.fits')
    out_redrockfile = os.path.join(datadir, 'redrock-test-data.fits')
    out_fastfile = os.path.join(datadir, 'fastspec-test-data.fits')
    if os.path.isfile(out_coaddfile):
        log.info(f'Warning: overwriting existing output file {out_coaddfile}')

    # select spectra with strong line-emission and no broad lines (for now)
    specprod, coadd_type = 'iron', 'healpix'
    survey, program, healpix = 'sv1', 'bright', 5060
    
    coaddfile = os.path.join(os.getenv('DESI_ROOT'), 'spectro', 'redux', specprod, coadd_type, survey, program,
                             str(healpix//100), str(healpix), f'coadd-{survey}-{program}-{healpix}.fits')
    redrockfile = replace_prefix(coaddfile, 'coadd', 'redrock')
        
    fastfile = os.path.join(os.getenv('DESI_ROOT'), 'spectro', 'fastspecfit', specprod, coadd_type, survey, program,
                             str(healpix//100), str(healpix), f'fastspec-{survey}-{program}-{healpix}.fits.gz')
    fast = Table(fitsio.read(fastfile, 'FASTSPEC', columns=['TARGETID', 'Z', 'HALPHA_EW', 'HALPHA_EW_IVAR', 'HALPHA_BROAD_FLUX_IVAR']))
    meta = Table(fitsio.read(fastfile, 'METADATA', columns=['Z', 'Z_RR']))

    I = np.where((fast['HALPHA_EW'] > 5.) * (fast['HALPHA_EW'] * np.sqrt(fast['HALPHA_EW_IVAR']) > 5.) *
                 (fast['HALPHA_BROAD_FLUX_IVAR'] == 0.) * (meta['Z'] == meta['Z_RR']))[0]
    if ntargets is not None and ntargets <= len(I):
        I = I[:ntargets]
    nobj = len(I)

    fast = Table(fitsio.read(fastfile, 'FASTSPEC', rows=I))
    meta = Table(fitsio.read(fastfile, 'METADATA', rows=I))
    targetids = meta['TARGETID'].data

    write_fastspecfit(fast, meta, outfile=out_fastfile, coadd_type=coadd_type, specprod=specprod)

    # read the best-fitting continuum models
    models, hdr = fitsio.read(fastfile, 'MODELS', header=True)

    modelwave = hdr['CRVAL1'] + np.arange(hdr['NAXIS1']) * hdr['CDELT1']
    models = np.squeeze(models[I, :, :])
    continuum = np.squeeze(models[:, 0, :] + models[:, 1, :])

    # read the original data (spectra) and redshifts of the selected targets
    spec = read_spectra(coaddfile).select(targets=targetids)
    assert(np.all(spec.target_ids() == targetids))    

    alltargetids = fitsio.read(redrockfile, ext='REDSHIFTS', columns='TARGETID')
    rows = np.where(np.isin(alltargetids, targetids))[0]
    zb = fitsio.read(redrockfile, ext='REDSHIFTS', rows=rows)
    fm = fitsio.read(redrockfile, ext='FIBERMAP', rows=rows)
    
    I = geomask.match_to(zb['TARGETID'], targetids)
    zb = zb[I]
    fm = fm[I]
    assert(np.all(zb['TARGETID'] == targetids))
    assert(np.all(fm['TARGETID'] == targetids))
        
    # Correct for dust extinction and subtract the stellar
    # continuum. Unfortunately, dust_transmission isn't vectorized, so we might
    # as well loop.
    for iobj in range(nobj):
        for cam in spec.bands:
            mw_transmission_spec = dust_transmission(spec.wave[cam], meta[iobj]['EBV'])
            spec.flux[cam][iobj, :] /= mw_transmission_spec
            spec.ivar[cam][iobj, :] *= mw_transmission_spec**2

            # this isn't quite right but shouldn't matter for these tests
            spec.flux[cam][iobj, :] -= np.interp(spec.wave[cam], modelwave, continuum[iobj, :])

    # write out
    log.info(f'Writing {len(targetids)} targets to {out_coaddfile}')
    write_spectra(out_coaddfile, spec)

    log.info(f'Writing {len(targetids)} targets to {out_redrockfile}')
    fitsio.write(out_redrockfile, zb, extname='REDSHIFTS', clobber=True)
    fitsio.write(out_redrockfile, fm, extname='FIBERMAP')


def main():
    """Main wrapper.

    """
    import argparse

    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('--build-test-data', action='store_true', help='Build the test dataset.')
    parser.add_argument('--fast', action='store_true', help='Build the refactored, fast emission-line fitting code.')
    parser.add_argument('--datadir', type=str, default="./data", help='I/O directory.')
    parser.add_argument('--ntargets', type=int, default=None, help='For testing, test on ntargets objects.')
    args = parser.parse_args()

    # Build the test dataset (will only work at NERSC if run by Moustakas).
    if args.build_test_data:
        build_test_data(args.datadir, ntargets=args.ntargets)
        return

    fit_emlines(datadir=args.datadir, fast=args.fast)
        

if __name__ == '__main__':
    main()


