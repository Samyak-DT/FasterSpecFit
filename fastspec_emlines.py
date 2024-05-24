#!/usr/bin/env python

"""Toy version of fastspec with simplified I/O and no stellar-continuum fitting
which will allow us to benchmark updated and sped-up emission-line fitting
algorithms.

Dependencies:
* astropy
* fitsio
* desispec
* desiutil
* desimodel(?)
* fastspecfit

python code/desihub/FasterSpecFit/fastspec_emlines.py --build-test-data --ntargets=3
python code/desihub/FasterSpecFit/fastspec_emlines.py --emlines-main

"""
import os, time, pdb
import numpy as np

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
            res = Resolution(spec.resolution_data[cam][iobj, :, :])

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


def run_emlines_main(datadir='.'):
    """Use the current (main) version of the emission-line fitting code.

    """
    from desiutil.log import get_logger
    from fastspecfit.emlines import EMFitTools

    log = get_logger()
    
    specdata = read_test_data(datadir=datadir)

    # loop on each spectrum
    for data in specdata:
        EMFit = EMFitTools(uniqueid=data['uniqueid'])

        # Combine all three cameras; we will unpack them to build the
        # best-fitting model (per-camera) below.
        redshift = data['zredrock']
        emlinewave = np.hstack(data['wave'])
        emlineivar = np.hstack(data['ivar'])
        emlineflux = np.hstack(data['flux']) # we already subtracted the continuum for this test
        resolution_matrix = data['res']
        camerapix = data['camerapix']
    
        weights = np.sqrt(emlineivar)

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

        # Initial fit - initial_linemodel_nobroad
        t0 = time.time()
        fit_nobroad = EMFit.optimize(linemodel_nobroad, emlinewave, emlineflux, weights, redshift,
                                     resolution_matrix, camerapix, log=log, debug=False, get_finalamp=True)
        model_nobroad = EMFit.bestfit(fit_nobroad, redshift, emlinewave, resolution_matrix, camerapix)
        chi2_nobroad, ndof_nobroad, nfree_nobroad = EMFit.chi2(fit_nobroad, emlinewave, emlineflux, emlineivar, model_nobroad, return_dof=True)
        log.info('Line-fitting with no broad lines and {} free parameters took {:.2f} seconds [niter={}, rchi2={:.4f}].'.format(
            nfree_nobroad, time.time()-t0, fit_nobroad.meta['nfev'], chi2_nobroad))

        # write out...

    
def run_emlines_fast(datadir='.'):
    pass



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
        print(f'Warning: overwriting existing output file {out_coaddfile}')

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
    print(f'Writing {len(targetids)} targets to {out_coaddfile}')
    write_spectra(out_coaddfile, spec)

    print(f'Writing {len(targetids)} targets to {out_redrockfile}')
    fitsio.write(out_redrockfile, zb, extname='REDSHIFTS', clobber=True)
    fitsio.write(out_redrockfile, fm, extname='FIBERMAP')


def main():
    """Main wrapper.

    """
    import argparse

    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('--build-test-data', action='store_true', help='Build the test dataset.')
    parser.add_argument('--emlines-main', action='store_true', help='Build the main emission-line fitting version.')
    parser.add_argument('--emlines-fast', action='store_true', help='Build the refactored, fast emission-line fitting code.')
    parser.add_argument('--datadir', type=str, default='./', help='I/O directory.')
    parser.add_argument('--ntargets', type=int, default=None, help='For testing, test on ntargets objects.')
    args = parser.parse_args()

    # Build the test dataset (will only work at NERSC if run by Moustakas).
    if args.build_test_data:
        build_test_data(args.datadir, ntargets=args.ntargets)

    if args.emlines_main:
        run_emlines_main(datadir='.')

    if args.emlines_fast:
        run_emlines_fast(datadir='.')
        

if __name__ == '__main__':
    main()


