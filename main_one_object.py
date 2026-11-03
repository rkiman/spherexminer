"""
Extract the SPHEREx spectrum of ONE object, measuring N_WORKERS images in parallel.

    python main_one_object.py                                   -> CONFIG below
    python main_one_object.py "TRAPPIST-1" ra dec pmra pmdec [epoch]
    python main_one_object.py "2MASS J00011217+1535355"         -> coords from LOOKUP_TABLE

Output: Data/spectra/Spherex_spec_<name>.fits and Figures/<name>/ (pipeline in src/pipeline.py)
"""
import matplotlib
matplotlib.use('Agg')   # non-interactive backend, safe for multiprocessing

import sys
from src.pipeline import process_object, lookup_object, quiet_warnings

quiet_warnings()

# ---------------------------------------------------------------------------
# CONFIG
# ---------------------------------------------------------------------------
NAME = 'TRAPPIST-1'
RA = 346.622368728579      # deg (at EPOCH)
DEC = -5.0413992505183     # deg (at EPOCH)
PMRA = 930.788             # mas/yr
PMDEC = -479.038           # mas/yr
EPOCH = 2000.

# Used only when a name is given without coordinates
LOOKUP_TABLE = 'Data/tables/UltracoolSheet_plus.csv'
LOOKUP_NAME_COLS = ['name_simbadable', 'name']
LOOKUP_COLS = dict(ra='ra_j2000_formula', dec='dec_j2000_formula',
                   pmra='pmra_formula', pmdec='pmdec_formula')

path_figures = 'Figures/'
path_data = 'Data/spectra/'
perc = [0.9, 0.95, 0.98]   # Percentage of flux to choose r_aperture
OVERWRITE = True           # True -> redo even if the .fits exists (also deletes the checkpoint)
DELETE_FIGURES = False     # True -> remove Figures/<name>/ after saving
N_WORKERS = 5              # images in parallel (try 4-10; limited by IRSA, network and RAM)


if __name__ == '__main__':
    args = sys.argv[1:]
    if len(args) == 0:
        name, ra, dec, pmra, pmdec, epoch = NAME, RA, DEC, PMRA, PMDEC, EPOCH
    elif len(args) == 1:
        name = args[0]
        ra, dec, pmra, pmdec, epoch = lookup_object(name, LOOKUP_TABLE, LOOKUP_NAME_COLS, LOOKUP_COLS)
    elif len(args) in (5, 6):
        name = args[0]
        ra, dec, pmra, pmdec = map(float, args[1:5])
        epoch = float(args[5]) if len(args) == 6 else 2000.
    else:
        sys.exit(__doc__)

    print(f'{name}: ra={ra} dec={dec} pmra={pmra} pmdec={pmdec} epoch={epoch}')
    name, status = process_object(name, ra, dec, pmra, pmdec, epoch,
                                  path_figures=path_figures, path_data=path_data, perc=perc,
                                  overwrite=OVERWRITE, delete_figures=DELETE_FIGURES,
                                  n_workers=N_WORKERS)
    print(f'\n{name}: {status}')
