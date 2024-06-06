#!/usr/bin/env python

"""
Build the test dataset (will only work at NERSC if run by Moustakas).

"""
def build_test_data(datadir='./', ntargets=None):
    """Build the test dataset; will only work at NERSC if run by Moustakas.

    """
    import os
    import numpy as np
    import fitsio
    from astropy.table import Table
    from desiutil.dust import dust_transmission
    from desispec.io import read_spectra, write_spectra
    from desispec.io.util import replace_prefix    
    from desitarget import geomask
    from fastspecfit.io import write_fastspecfit

    from desiutil.log import get_logger, INFO
    log = get_logger(INFO)

    specprod, coadd_type = 'iron', 'healpix'
    survey, program, healpix, targetid = 'sv1', 'bright', 5060, None
    #survey, program, healpix, targetid = 'sv3', 'bright', 28119, 39628390055022140
    #survey, program, healpix, targetid = 'sv3', 'dark', 25960, 1092092734472204
                
    out_coaddfile = os.path.join(datadir, f'coadd-{survey}-{program}-{healpix}.fits')
    out_redrockfile = os.path.join(datadir, f'redrock-{survey}-{program}-{healpix}.fits')
    out_fastfile = os.path.join(datadir, f'fastspec-{survey}-{program}-{healpix}.fits')
    if os.path.isfile(out_coaddfile):
        log.info(f'Warning: overwriting existing output file {out_coaddfile}')

    coaddfile = os.path.join(os.getenv('DESI_ROOT'), 'spectro', 'redux', specprod, coadd_type, survey, program,
                             str(healpix//100), str(healpix), f'coadd-{survey}-{program}-{healpix}.fits')
    redrockfile = replace_prefix(coaddfile, 'coadd', 'redrock')
        
    fastfile = os.path.join(os.getenv('DESI_ROOT'), 'spectro', 'fastspecfit', specprod, coadd_type, survey, program,
                             str(healpix//100), str(healpix), f'fastspec-{survey}-{program}-{healpix}.fits.gz')
    fast = Table(fitsio.read(fastfile, 'FASTSPEC', columns=['TARGETID', 'Z', 'HALPHA_EW', 'HALPHA_EW_IVAR', 'HALPHA_BROAD_FLUX_IVAR']))
    meta = Table(fitsio.read(fastfile, 'METADATA', columns=['Z', 'Z_RR']))

    if False:
        I = np.where((fast['HALPHA_EW'] > 5.) * (fast['HALPHA_EW'] * np.sqrt(fast['HALPHA_EW_IVAR']) > 5.) *
                     (fast['HALPHA_BROAD_FLUX_IVAR'] == 0.) * (meta['Z'] == meta['Z_RR']))[0]
    else:
        I = np.arange(len(fast))

    if targetid is not None:
        I = np.where(fast['TARGETID'] == targetid)[0]
        
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
    if nobj == 1:
        continuum = np.squeeze(models[0, :] + models[1, :])
    else:
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
            if nobj == 1:
                spec.flux[cam] /= mw_transmission_spec
                spec.ivar[cam] *= mw_transmission_spec**2
                # this isn't quite right but shouldn't matter for these tests
                spec.flux[cam] -= np.interp(spec.wave[cam], modelwave, continuum)
            else:
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


if __name__ == '__main__':

    import argparse

    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('--datadir', type=str, default="./data", help='I/O directory.')
    parser.add_argument('--ntargets', type=int, default=None, help='For testing, test on ntargets objects.')
    args = parser.parse_args()

    build_test_data(args.datadir, ntargets=args.ntargets)
