"""
Shared helpers for the three main scripts:
    main_one_object.py    -> one object
    main_small_table.py   -> a small table, processed one by one
    main_large_sample.py  -> a large sample, processed in parallel

This is a NEW module; it only wraps src.extract_spectrum.extract_spectrum
and does not change any existing code.
"""
import os
import shutil
import logging
import warnings

import numpy as np
from astropy.table import Table
from astropy.coordinates import SkyCoord
from astropy import units as u

from .extract_spectrum import extract_spectrum

SPHEREX_COLLECTION = 'spherex_qr2'
SEARCH_RADIUS_ARCSEC = 3


def quiet_warnings():
    """Silence the WCS / ERFA warnings (call again inside each subprocess)."""
    logging.getLogger('astropy.wcs').setLevel(logging.ERROR)
    from erfa import ErfaWarning
    warnings.filterwarnings('ignore', category=ErfaWarning,
                            message='.*pmsafe.*distance overridden.*')


def as_float(col, fill=np.nan):
    """Return a plain float array, with masked entries replaced by `fill`."""
    return np.ma.filled(np.ma.asarray(col, dtype=float), fill)


def clean_name(name):
    """Name used for output files (same convention as the old scripts)."""
    return str(name).replace(' ', '')


def spectrum_file(name, path_data):
    return os.path.join(path_data, 'Spherex_spec_' + clean_name(name) + '.fits')


def load_input_table(file_name, col_map=None, epoch_default=2000.):
    """
    Read a table and return it with standard columns:
        Name, ra, dec, pmra, pmdec, epoch   (+ any other original columns)

    col_map: dict {standard_name: column_in_file}. Columns already named
    Name/ra/dec/pmra/pmdec/epoch don't need to be listed. If there is no
    epoch column, `epoch_default` is used.
    """
    tab = Table.read(file_name)
    col_map = dict(col_map or {})
    out = Table()
    for std in ['Name', 'ra', 'dec', 'pmra', 'pmdec', 'epoch']:
        src = col_map.get(std, std)
        if src in tab.colnames:
            out[std] = tab[src] if std == 'Name' else as_float(tab[src])
        elif std == 'epoch':
            out['epoch'] = np.full(len(tab), float(epoch_default))
        else:
            raise KeyError(f"Column for '{std}' not found in {file_name}. "
                           f"Set it in COL_MAP. Available: {tab.colnames}")
    out['Name'] = [str(n) for n in out['Name']]
    # keep the rest of the original columns (useful for selections / sorting)
    used = set(col_map.get(s, s) for s in ['Name', 'ra', 'dec', 'pmra', 'pmdec', 'epoch'])
    for c in tab.colnames:
        if c not in used and c not in out.colnames:
            out[c] = tab[c]
    return out


def process_object(name, ra_deg, dec_deg, pmra, pmdec, epoch,
                   path_figures='Figures/', path_data='Data/spectra/',
                   perc=(0.9, 0.95, 0.98), overwrite=False,
                   delete_figures=False, label=''):
    """
    Query IRSA for one object, extract its SPHEREx spectrum and save it to
    Data/spectra/Spherex_spec_<name>.fits.

    Returns (name, status) with status in
        'done', 'skipped', 'no_coords', 'no_data', 'error: ...'
    """
    from astroquery.ipac.irsa import Irsa   # imported here so workers are clean
    quiet_warnings()

    name = clean_name(name)
    tag = f'{label} {name}'.strip()
    file_name = spectrum_file(name, path_data)
    fig_dir = os.path.join(path_figures, name) + '/'

    if os.path.exists(file_name) and not overwrite:
        print(f'{tag}: already done, skipping.')
        return name, 'skipped'

    if not np.all(np.isfinite([ra_deg, dec_deg, pmra, pmdec, epoch])):
        print(f'{tag}: missing coordinates / proper motion, skipping.')
        return name, 'no_coords'

    try:
        print(f'{tag}: querying IRSA...')
        coord = SkyCoord(ra_deg, dec_deg, unit='deg')
        results = Irsa.query_sia(pos=(coord, SEARCH_RADIUS_ARCSEC * u.arcsec),
                                 collection=SPHEREX_COLLECTION)
        if len(results) == 0:
            print(f'{tag}: no SPHEREx images found.')
            return name, 'no_data'

        os.makedirs(fig_dir, exist_ok=True)
        os.makedirs(path_data, exist_ok=True)

        # extract_spectrum builds paths as path + name, so keep trailing '/'
        pf = path_figures if path_figures.endswith('/') else path_figures + '/'
        pd = path_data if path_data.endswith('/') else path_data + '/'
        spec = extract_spectrum(name, results, ra_deg, dec_deg, pmra, pmdec,
                                epoch, list(perc), pf, pd)
        spec.sort('Wave')
        spec.write(file_name, format='fits', overwrite=True)

        if delete_figures and os.path.isdir(fig_dir):
            shutil.rmtree(fig_dir, ignore_errors=True)

        print(f'{tag}: done ({len(results)} images).')
        return name, 'done'

    except Exception as e:
        print(f'{tag}: ERROR - {e}')
        return name, f'error: {e}'


def _process_row(kwargs):
    """Picklable wrapper for ProcessPoolExecutor."""
    return process_object(**kwargs)


def write_summary(results_summary, file_name):
    """Save {name: status} as a csv and print the counts."""
    from collections import Counter
    t = Table(rows=list(results_summary.items()), names=('Name', 'status'),
              dtype=(str, str)) if results_summary else Table(names=('Name', 'status'))
    if file_name:
        t.write(file_name, overwrite=True)
    counts = Counter(results_summary.values())
    print('\n=== Run complete ===')
    for status, n in sorted(counts.items()):
        print(f'  {status}: {n}')
    if file_name:
        print(f'Summary saved to {file_name}')
    return t
