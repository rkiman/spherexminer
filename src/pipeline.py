"""
Shared extraction pipeline used by main_one_object.py and main_large_sample.py.
One object = query IRSA -> measure every image (serial or parallel) -> one spectrum.
"""
import os
import time
import shutil
import tempfile
from concurrent.futures import ProcessPoolExecutor, as_completed

import numpy as np
import requests
import matplotlib
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
from astropy.io import fits
from astropy.wcs import WCS
from astropy.table import Table
from astropy.coordinates import SkyCoord
from astropy import units as u

from .run_extraction import (quiet_warnings, as_float, clean_name, spectrum_file,
                             load_input_table, write_summary,
                             SPHEREX_COLLECTION, SEARCH_RADIUS_ARCSEC)
from .extract_spectrum import (move_forward_position, get_closest_psf,
                               study_different_apertures, get_aperture_corr,
                               calculate_flux_and_bkg, checkpoint_path,
                               _load_checkpoint, _save_checkpoint)
from .extras import update_psf_header, get_spherex_channel, detector_from_bandpass

MAX_DOWNLOAD_TRIES = 10    # per image, with increasing wait between tries
DOWNLOAD_TIMEOUT = 120     # seconds without data before a try is abandoned
TMP_DIR = None             # temporary image downloads; None -> system temp folder (outside Dropbox)


# ---------------------------------------------------------------------------
# Input: look up one object, or select objects from a table
# ---------------------------------------------------------------------------
def lookup_object(name, table_file, name_cols, col_map):
    """ra, dec, pmra, pmdec, epoch (2000) of `name` from a table."""
    tab = Table.read(table_file)
    key = name.replace(' ', '').lower()
    for col in name_cols:
        names = np.array([str(n).replace(' ', '').lower() for n in tab[col]])
        idx = np.where(names == key)[0]
        if len(idx):
            r = tab[idx[0]]
            vals = [float(as_float([r[col_map[k]]])[0]) for k in ['ra', 'dec', 'pmra', 'pmdec']]
            return (*vals, 2000.)
    raise ValueError(f"'{name}' not found in {table_file}")


_NUMERIC_OPS = {'>': np.greater, '>=': np.greater_equal,
                '<': np.less, '<=': np.less_equal}


def build_mask(known_data, mask=None, apply_mask=True, verbose=True):
    """Rows to keep: always need finite astrometry; plus every cut in `mask` (AND) if apply_mask.
    A cut is (column, op, value) with op in > >= < <= == != in 'not in' finite, or f(table) -> bool array."""
    cols = known_data.colnames
    keep = np.ones(len(known_data), dtype=bool)
    for c in ['ra', 'dec', 'pmra', 'pmdec', 'epoch']:
        keep &= np.isfinite(as_float(known_data[c]))
    if verbose:
        print(f'Astrometry: {keep.sum()} of {len(known_data)} have ra, dec, pm, epoch.')
    if not apply_mask or not mask:
        return keep

    for cut in mask:
        if callable(cut):
            m = np.asarray(cut(known_data), dtype=bool)
            label = getattr(cut, '__name__', 'custom function')
        else:
            col, op, *value = cut
            value = value[0] if value else None
            label = f'{col} {op} {value if value is not None else ""}'.strip()
            if col not in cols:   # missing column: warn and skip the cut
                print(f'  WARNING: column {col!r} not in table, skipping cut: {label}')
                continue
            if op in _NUMERIC_OPS:
                x = as_float(known_data[col])
                with np.errstate(invalid='ignore'):
                    m = _NUMERIC_OPS[op](x, value) & np.isfinite(x)
            elif op == 'finite':
                m = np.isfinite(as_float(known_data[col]))
            elif op in ('==', '!=', 'in', 'not in'):
                x = np.asarray(known_data[col])
                m = np.isin(x, [value] if op in ('==', '!=') else list(value))
                if op in ('!=', 'not in'):
                    m = ~m
            else:
                raise ValueError(f'Unknown operator {op!r} in mask entry {cut!r}')
        keep &= m
        if verbose:
            print(f'  {label}: {keep.sum()} left')
    return keep


# ---------------------------------------------------------------------------
# One image
# ---------------------------------------------------------------------------
def download_image(url, tmp_dir, max_tries=MAX_DOWNLOAD_TRIES, timeout=DOWNLOAD_TIMEOUT):
    """Download one image to a unique temporary file; retry with increasing wait (IRSA fails sometimes)."""
    for attempt in range(1, max_tries + 1):
        fd, output_path = tempfile.mkstemp(suffix='.fits', dir=tmp_dir)
        os.close(fd)
        try:
            with requests.get(url, stream=True, timeout=timeout) as r:
                r.raise_for_status()
                with open(output_path, 'wb') as f:
                    for chunk in r.iter_content(chunk_size=4 * 1024 * 1024):
                        if chunk:
                            f.write(chunk)
            return output_path
        except Exception as e:
            if os.path.exists(output_path):
                os.remove(output_path)
            if attempt == max_tries:
                raise RuntimeError(f'download failed after {max_tries} tries: {e}')
            time.sleep(min(2 * attempt, 30))


def measure_image(url, t_min, ra_deg, dec_deg, pmra, pmdec, epoch, perc,
                  fig_dir, tmp_dir, bandpass=''):
    """Download one image and measure the object for every aperture in perc.
    Returns the dict stored in the checkpoint (wave, flux, ..., channel)."""
    output_path = download_image(url, tmp_dir)
    try:
        hdulist = fits.open(output_path)
        hdulist = update_psf_header(hdulist)

        # Diagnostic plots
        fig = plt.figure(figsize=(15, 8))
        gs = GridSpec(2, 4, figure=fig)
        ax0, ax1, ax2, ax3 = [fig.add_subplot(gs[0, k]) for k in range(4)]
        ax4 = fig.add_subplot(gs[1, :2])
        ax5 = fig.add_subplot(gs[1, 2:])

        header = hdulist[1].header
        psf_header, psfcube = hdulist[5].header, hdulist[5].data
        flags_header, flags_data = hdulist[2].header, hdulist[2].data
        image_data = hdulist[1].data
        image_data_err = np.sqrt(hdulist[3].data)
        obs_date_mjd = header['MJD-OBS']

        # Pixel position of the object at the epoch of this image
        new_ra, new_dec = move_forward_position(ra_deg, dec_deg, pmra, pmdec, epoch, t_min)
        x_rel, y_rel = WCS(header).world_to_pixel(SkyCoord(ra=new_ra, dec=new_dec, unit='deg'))

        # Wavelength and SPHEREx channel (one of the 102) at that pixel
        spectral_wcs = WCS(header=hdulist['IMAGE'].header, fobj=hdulist, key='W')
        spectral_wcs.sip = None
        wl, bw = spectral_wcs.pixel_to_world(x_rel, y_rel)
        detector = header.get('DETECTOR', detector_from_bandpass(bandpass))
        det, sub, chan, from_map = get_spherex_channel(x_rel, y_rel, detector, float(wl.value))

        # Aperture radius so that perc of the flux is included
        colors_perc = plt.colormaps['plasma'](np.linspace(0, 1, len(perc)))
        psf = get_closest_psf(psf_header, psfcube, x_rel, y_rel)
        c_psf = psf.shape[0] / 2 - 1
        r_aperture, r_in, r_out = study_different_apertures(
            x_rel, y_rel, image_data, image_data_err, flags_header, flags_data,
            psf, c_psf, c_psf, psf_header['OVERSAMP'], perc, ax4, ax5, colors_perc)

        omega_st = header['HIERARCH OMEGA_MEDIAN'] / 206265**2      # sr
        omega_st_err = header['HIERARCH OMEGA_STDDEV'] / 206265**2  # sr

        flux_j, flux_err_j = [], []
        for j in range(len(perc)):
            aperture_corr = get_aperture_corr(psf, c_psf, c_psf,
                                              r_aperture[j] * psf_header['OVERSAMP'],
                                              ax0, colors_perc[j])
            aper_flux, aper_flux_err, total_bkg, total_bkg_err = calculate_flux_and_bkg(
                x_rel, y_rel, image_data, image_data_err, flags_header, flags_data,
                r_aperture[j], r_in[j], r_out[j], ax1, ax2, ax3, colors_perc[j])

            # Background-subtracted flux, MJy/sr -> uJy
            f = aper_flux - total_bkg
            f_err = np.sqrt(aper_flux_err**2 + total_bkg_err**2)
            k = omega_st * 1e12 * aperture_corr
            flux_j.append(float((f * k)[0]))
            flux_err_j.append(float(np.sqrt((f_err * k)**2 + (f * omega_st_err * 1e12 * aperture_corr)**2)[0]))

        plt.tight_layout()
        plt.savefig(os.path.join(fig_dir, str(np.round(wl.value, 2)) + '_phot.png'),
                    dpi=100, bbox_inches='tight')
        plt.close(fig)
        hdulist.close()
    finally:
        if os.path.exists(output_path):
            os.remove(output_path)

    return {'wave': float(wl.value), 'flux': flux_j, 'flux_err': flux_err_j,
            'r_ap': [float(r) for r in r_aperture], 'r_in': [float(r) for r in r_in],
            'r_out': [float(r) for r in r_out], 'obs_date_mjd': float(obs_date_mjd),
            'x': float(x_rel), 'y': float(y_rel), 'detector': int(det), 'subchan': int(sub),
            'channel': int(chan), 'channel_from_map': bool(from_map)}


def query_images(ra_deg, dec_deg, max_tries=5):
    """SPHEREx images covering (ra, dec); retry since IRSA sometimes times out."""
    from astroquery.ipac.irsa import Irsa
    pos = (SkyCoord(ra_deg, dec_deg, unit='deg'), SEARCH_RADIUS_ARCSEC * u.arcsec)
    for attempt in range(1, max_tries + 1):
        try:
            return Irsa.query_sia(pos=pos, collection=SPHEREX_COLLECTION)
        except Exception:
            if attempt == max_tries:
                raise
            time.sleep(5 * attempt)


def _worker_init():
    """Run at the start of every worker process."""
    matplotlib.use('Agg')
    quiet_warnings()
    from astropy import log
    log.setLevel('WARNING')   # hide the repeated SIP INFO messages


def _measure_all(jobs, n_workers):
    """Yield (i, result or exception) for each job: serial if n_workers <= 1, else in a process pool."""
    if n_workers <= 1:
        for i, job in jobs.items():
            try:
                yield i, measure_image(**job)
            except Exception as e:
                yield i, e
    else:
        with ProcessPoolExecutor(max_workers=n_workers, initializer=_worker_init) as pool:
            futures = {pool.submit(measure_image, **job): i for i, job in jobs.items()}
            for fut in as_completed(futures):
                try:
                    yield futures[fut], fut.result()
                except Exception as e:
                    yield futures[fut], e


# ---------------------------------------------------------------------------
# All images -> one spectrum
# ---------------------------------------------------------------------------
def compile_spectrum(entries, name, ra_deg, dec_deg, pmra, pmdec, epoch, perc, fig_dir):
    """Table with one row per image (sorted by wavelength) + comparison plot."""
    spec = Table()
    spec['Wave'] = np.array([e['wave'] for e in entries])
    spec['Wave'].unit = 'um'
    spec['Wave'].description = 'Observed wavelength'

    for j, p in enumerate(perc):
        p = str(p)
        cols = [('Flux_', 'flux', 'uJy', f'Flux density with {p} percent aperture'),
                ('Fluxerr_', 'flux_err', 'uJy', f'1-sigma flux uncertainty with {p} percent aperture'),
                ('r_aperture_', 'r_ap', 'pix', f'Photometric aperture radius with {p} percent aperture'),
                ('r_in_', 'r_in', 'pix', f'Sky annulus inner radius with {p} percent aperture'),
                ('r_out_', 'r_out', 'pix', f'Sky annulus outer radius with {p} percent aperture')]
        for prefix, key, unit, desc in cols:
            spec[prefix + p] = np.array([e[key][j] for e in entries])
            spec[prefix + p].unit = unit
            spec[prefix + p].description = desc

    spec['OBSMJD'] = np.array([e['obs_date_mjd'] for e in entries])
    spec['OBSMJD'].description = 'Observation date (MJD)'
    spec['DETECTOR'] = np.array([e['detector'] for e in entries], dtype=int)
    spec['DETECTOR'].description = 'SPHEREx detector (1-6)'
    spec['SUBCHAN'] = np.array([e['subchan'] for e in entries], dtype=int)
    spec['SUBCHAN'].description = 'Spectral sub-channel within the detector (1-17)'
    spec['CHANNEL'] = np.array([e['channel'] for e in entries], dtype=int)
    spec['CHANNEL'].description = 'SPHEREx channel id = 100*DETECTOR + SUBCHAN (102 channels)'
    spec['CHANNEL_FROM_MAP'] = np.array([e['channel_from_map'] for e in entries], dtype=bool)
    spec['CHANNEL_FROM_MAP'].description = ('True: channel from the CHANNEL_MAP at the source pixel; '
                                            'False: outside nominal channels, closest in wavelength')

    spec.meta.update(NAME=name, RA=ra_deg, DEC=dec_deg, PMRA=pmra, PMDEC=pmdec,
                     EPOCH=epoch, PERC=','.join(str(p) for p in perc))
    spec.meta['comments'] = [
        'NAME: Target identifier',
        'RA, DEC: ICRS coordinates in degrees',
        'PMRA, PMDEC: proper motion in mas/yr',
        'EPOCH: epoch of the position',
        'PERC: aperture correction percentage',
        'CHANNEL: 100*DETECTOR + SUBCHAN, from spectral_channels_spx_cal-sch-v1-2026-106.fits',
    ]
    spec.sort('Wave')

    # Final comparison plot
    colors_perc = plt.colormaps['plasma'](np.linspace(0, 1, len(perc)))
    fig = plt.figure()
    for j, p in enumerate(perc):
        plt.errorbar(spec['Wave'], spec['Flux_' + str(p)], yerr=spec['Fluxerr_' + str(p)],
                     fmt='.', label='Percentage aperture ' + str(p), color=colors_perc[j])
    plt.legend(title=name)
    plt.xscale('log')
    plt.xlabel('Wavelength (um)')
    plt.ylabel('Flux (uJy)')
    plt.savefig(os.path.join(fig_dir, '_comparison_spec.png'), dpi=100)
    plt.close(fig)
    return spec


def process_object(name, ra_deg, dec_deg, pmra, pmdec, epoch,
                   path_figures='Figures/', path_data='Data/spectra/',
                   perc=(0.9, 0.95, 0.98), overwrite=False, delete_figures=False,
                   n_workers=1, label=''):
    """Extract and save the spectrum of one object, measuring n_workers images in parallel.
    Returns (name, status): 'done', 'skipped', 'no_coords', 'no_data' or 'error: ...'."""
    _worker_init()

    name = clean_name(name)
    tag = f'{label} {name}'.strip()
    perc = list(perc) if isinstance(perc, (list, tuple)) else [perc]
    file_name = spectrum_file(name, path_data)
    fig_dir = os.path.join(path_figures, name)

    if os.path.exists(file_name) and not overwrite:
        print(f'{tag}: already done, skipping.')
        return name, 'skipped'
    if not np.all(np.isfinite([ra_deg, dec_deg, pmra, pmdec, epoch])):
        print(f'{tag}: missing coordinates / proper motion, skipping.')
        return name, 'no_coords'

    try:
        # Overwrite -> start from scratch, without the old checkpoint
        checkpoint_file = checkpoint_path(name, path_data)
        if overwrite:
            for f in (checkpoint_file, checkpoint_file + '.tmp'):
                if os.path.exists(f):
                    os.remove(f)

        print(f'{tag}: querying IRSA...')
        results = query_images(ra_deg, dec_deg)
        if len(results) == 0:
            print(f'{tag}: no SPHEREx images found.')
            return name, 'no_data'
        os.makedirs(fig_dir, exist_ok=True)

        urls = [str(x) for x in results['access_url']]
        bandpasses = [str(x) for x in results['energy_bandpassname']]
        n = len(urls)

        # Only images not in the checkpoint are measured; each one is saved as it finishes
        checkpoint = _load_checkpoint(checkpoint_file)
        jobs = {i: dict(url=urls[i], t_min=float(results['t_min'][i]), ra_deg=ra_deg,
                        dec_deg=dec_deg, pmra=pmra, pmdec=pmdec, epoch=epoch, perc=perc,
                        fig_dir=fig_dir, tmp_dir=TMP_DIR, bandpass=bandpasses[i])
                for i in range(n) if urls[i] not in checkpoint}
        print(f'{tag}: {n} images, {n - len(jobs)} in checkpoint, {len(jobs)} to process '
              f'with {n_workers} worker(s).')

        failed, t0 = {}, time.time()
        for k, (i, res) in enumerate(_measure_all(jobs, n_workers), 1):
            if isinstance(res, Exception):
                failed[urls[i]] = str(res)
                print(f'  {tag} image {i+1}/{n} FAILED: {res}', flush=True)
            else:
                checkpoint[urls[i]] = res
                _save_checkpoint(checkpoint_file, checkpoint)
                print(f'  {tag} image {k}/{len(jobs)} done: {res["wave"]:.3f} um', flush=True)
        print(f'{tag}: images finished in {time.time() - t0:.0f} s.')

        if failed:
            msg = (f'{len(failed)} of {n} images failed; the rest are in the checkpoint. '
                   f'Rerun to retry only the failed ones.')
            print(f'{tag}: ERROR - {msg}')
            return name, f'error: {msg}'

        # Checkpoints from older versions have no channel: get it from the band + wavelength
        entries = [checkpoint[url] for url in urls]
        for e, bp in zip(entries, bandpasses):
            if 'channel' not in e:
                det, sub, chan, from_map = get_spherex_channel(
                    e.get('x'), e.get('y'), detector_from_bandpass(bp), e['wave'])
                e.update(detector=det, subchan=sub, channel=chan, channel_from_map=from_map)

        spec = compile_spectrum(entries, name, ra_deg, dec_deg, pmra, pmdec, epoch, perc, fig_dir)
        spec.write(file_name, format='fits', overwrite=True)
        os.remove(checkpoint_file)
        if delete_figures:
            shutil.rmtree(fig_dir, ignore_errors=True)

        print(f'{tag}: done ({n} images) -> {file_name}')
        return name, 'done'

    except Exception as e:
        print(f'{tag}: ERROR - {e}')
        return name, f'error: {e}'


# ---------------------------------------------------------------------------
# Many objects
# ---------------------------------------------------------------------------
def _process_job(job):
    """Picklable wrapper so process_object can run in a process pool."""
    return process_object(**job)


def run_sample(jobs, n_workers=1, summary_file=None):
    """Run process_object for every job (dict of its arguments), n_workers objects in parallel.
    Writes a csv with the status of each object, even if the run is interrupted."""
    summary = {}
    try:
        if n_workers <= 1:
            for job in jobs:
                name, status = _process_job(job)
                summary[name] = status
        else:   # FIFO queue: objects start in table order
            with ProcessPoolExecutor(max_workers=n_workers, initializer=_worker_init) as pool:
                futures = {pool.submit(_process_job, job): job['name'] for job in jobs}
                for fut in as_completed(futures):
                    try:
                        name, status = fut.result()
                    except Exception as e:   # e.g. a worker crashed
                        name, status = clean_name(futures[fut]), f'error: {e}'
                    summary[name] = status
    finally:
        write_summary(summary, summary_file)
    return summary
