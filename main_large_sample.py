"""
Extract SPHEREx spectra for a LARGE sample, N_WORKERS objects in parallel.

    python main_large_sample.py [table.csv]

Objects with an existing .fits are skipped, so the run can be stopped and restarted.
A csv with the status of every object is saved in SUMMARY_FILE (pipeline in src/pipeline.py).
"""
import matplotlib
matplotlib.use('Agg')   # non-interactive backend, safe for multiprocessing

import sys
from src.pipeline import (load_input_table, build_mask, run_sample, quiet_warnings,
                          as_float)

quiet_warnings()

# ---------------------------------------------------------------------------
# CONFIG
# ---------------------------------------------------------------------------
INPUT_TABLE = 'Data/tables/UltracoolSheet_plus.csv'
COL_MAP = dict(Name='name_simbadable', ra='ra_j2000_formula',
               dec='dec_j2000_formula', pmra='pmra_formula',
               pmdec='pmdec_formula')   # None if table already has Name, ra, ...

# Selection: rows without astrometry are always dropped. If APPLY_MASK, every cut is applied (AND).
# A cut is (column, op, value) with op in > >= < <= == != in 'not in' finite, or f(table) -> bool array.
APPLY_MASK = True
MASK = [
    ('sptnum_opt_formula', '>=', 10),     # 10 = L0
    ('sptnum_opt_formula', '<', 29),      # 28 = T8 (inclusive up to T8.x)
    # ('age_singlevalue_gyr_formula', '>', 1),       # e.g. field objects
    # ('age_singlevalue_gyr_formula', '<=', 1),      # e.g. young objects
    # ('name_simbadable', 'in', ['2MASS J0355+1133']),
    # lambda t: as_float(t['Ks_2MASS']) < 14,
]

N_WORKERS = 5             # objects in parallel (try 4-8, limited by IRSA/network)
path_figures = 'Figures/'
path_data = 'Data/spectra/'
perc = [0.9, 0.95, 0.98]  # Percentage of flux to choose r_aperture
OVERWRITE = False         # True -> redo objects with a .fits (also deletes their checkpoint)
DELETE_FIGURES = True     # remove Figures/<name>/ once the .fits is saved
SUMMARY_FILE = 'Data/summary_large_sample.csv'


if __name__ == '__main__':
    table = load_input_table(sys.argv[1] if len(sys.argv) > 1 else INPUT_TABLE, COL_MAP)
    keep = build_mask(table, MASK, apply_mask=APPLY_MASK)
    table = table[keep]
    n = len(table)
    print(f'Selection: {n} objects to process with {N_WORKERS} workers.\n')

    # Each object is one job; its images are measured one after the other (n_workers=1)
    jobs = [dict(name=row['Name'], ra_deg=row['ra'], dec_deg=row['dec'], pmra=row['pmra'],
                 pmdec=row['pmdec'], epoch=row['epoch'], path_figures=path_figures,
                 path_data=path_data, perc=perc, overwrite=OVERWRITE,
                 delete_figures=DELETE_FIGURES, n_workers=1, label=f'[{i + 1}/{n}]')
            for i, row in enumerate(table)]
    run_sample(jobs, n_workers=N_WORKERS, summary_file=SUMMARY_FILE)
