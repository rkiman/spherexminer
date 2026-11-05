# spherexminer
Extract and analyze data from SPHEREx.

## What it does

For a given object (name, coordinates and proper motion), the code:

1. Queries IRSA for all SPHEREx images (`spherex_qr2`) that cover the object.
2. Moves the object's position to the epoch of each image using its proper motion.
3. Measures aperture photometry on each image (apertures enclosing 90%, 95% and 98% of the PSF flux).
4. Combines all measurements into one SPHEREx spectrum.

Outputs:

- `Data/spectra/Spherex_spec_<name>.fits` — the extracted spectrum
- `Figures/<name>/` — diagnostic figures for each image

The pipeline lives in `src/pipeline.py`. 

## Setup

The code needs the official SPHEREx spectral channel definitions, `spectral_channels_spx_cal-sch-v1-2026-106.fits`, which must be downloaded from IRSA and placed in `src/`:

- Download: https://irsa.ipac.caltech.edu/ibe/data/spherex
- Description of the file: https://caltech-ipac.github.io/spherex-archive-documentation/spherex-data-products/

## Usage

Run from the repository folder.

### One object — `main_one_object.py`

```bash
# Use the object defined in the CONFIG section of the script (TRAPPIST-1 by default)
python main_one_object.py

# Give the name, RA, Dec (deg), pmRA, pmDec (mas/yr) and, optionally, the epoch (default 2000)
python main_one_object.py "TRAPPIST-1" 346.622368728579 -5.0413992505183 930.788 -479.038 2000

# Give only the name; coordinates are looked up in Data/tables/UltracoolSheet_plus.csv
python main_one_object.py "2MASS J00011217+1535355"
```

The images of the object are measured in parallel to make the run faster (`N_WORKERS`, set in the script).

### Large sample — `main_large_sample.py`

```bash
# Use the table and selection defined in the CONFIG section of the script
python main_large_sample.py

# Or give another input table
python main_large_sample.py Data/tables/my_targets.csv
```

- Several objects are processed in parallel (`N_WORKERS`).
- The input table needs a name, `ra`, `dec`, `pmra`, `pmdec` and (optionally) `epoch`. If your column names differ, map them with `COL_MAP`.
- Select a subsample with `MASK` (e.g. spectral type L0–T8, age cuts).
- Objects that already have a `.fits` are skipped, so the run can be stopped and restarted.
- The status of every object is saved in `Data/summary_large_sample.csv`.

Other options (output paths, `OVERWRITE`, `DELETE_FIGURES`) are set at the top of each script.

## Cleaning the spectra (in a Jupyter notebook)

The `.fits` files from the main scripts are *raw*: they have every measurement (one per image/epoch) at all three apertures.
`clean_sample_v2` (in `src/check_spectra.py`) turns them into clean spectra:

- For each object it picks the aperture (90, 95 or 98%) whose synthetic photometry best matches the literature photometry.
- It groups the points by SPHEREx channel, using the `CHANNEL` column of the raw spectrum (or the official channel definitions from the wavelength, for older extractions without it), and combines all epochs in each channel (weighted mean; the error is inflated if the epochs scatter more than their errors).
- It shows an interactive window to flag extreme outliers by hand (`check_outliers=True`, see [Interactive outlier check](#interactive-outlier-check)).

Example adapted from `low-gravity-LTdwarfs/standards.ipynb`:

```python
import os, sys
import numpy as np
from astropy.table import Table

sys.path.append('../spherexminer/')    # path to this repository
from src import *

# Table with Name and literature photometry (columns <filter> and <filter>_err)
known_data = Table.read('../spherexminer/Data/tables/standards.csv')
FILTER_PATH = '/path/to/Filters/'      # filter transmission curves

# Raw spectra that exist
spectra, idx = [], []
for i, name in enumerate(known_data['Name']):
    path = '../spherexminer/Data/spectra/Spherex_spec_' + name.replace(' ', '') + '.fits'
    if os.path.exists(path):
        spectra.append(path)
        idx.append(i)

# Literature photometry used to choose the aperture
filters = ['J_2MASS', 'H_2MASS', 'K_2MASS', 'W1', 'I1', 'I2', 'Lp_MKO', 'Mp_MKO']
lit = [literature_photometry(known_data, i, filters) for i in idx]
lit_mag = np.array([x[0] for x in lit])
lit_err = np.array([x[1] for x in lit])

# Clean and save to data/spectra_v2/ (same file names)
summary = clean_sample_v2(spectra, 'data/spectra_v2/', perc='literature',
                          phot_measurements=lit_mag, phot_err_measurements=lit_err,
                          fil_phot_synt=filters, filter_path=FILTER_PATH,
                          error_method='birge', overwrite=True)

# Read one clean spectrum: one point per SPHEREx channel
spec = Table.read('data/spectra_v2/' + os.path.basename(spectra[0]))
spec['Wave', 'Flux', 'Fluxerr', 'Bandwidth', 'Npoints']
```

Options:

- To use a single aperture for everyone, set `perc=0.95` (no photometry needed).
- To also drop points you removed by hand in an earlier version, add `reference_dir='data/spectra_v1/'`.
- To save the points you remove in the outlier check (one `<name>_removed_points.ecsv` per object), add `outlier_dir='data/spectra_v2/outliers/'`. On later runs the saved choices are reused without asking again; set `recheck_outliers=True` to redo the check.

### Interactive outlier check

With `check_outliers=True` and `interactive=True` (the defaults), each object with extreme outliers (more than `outlier_sigma`, default 8σ, from a running median) opens a window with all its measurements. The flagged points are circled in red and preselected. Click or drag a box to select points (shift-drag deselects); **Select flagged** and **Clear** reset the selection. Press **Done** (or Enter, or close the window) to remove the selected points and move to the next object.

You can run it from a notebook with `%matplotlib inline`. The click-and-wait loop only works in a desktop window, so `clean_sample_v2` switches matplotlib once to the first desktop backend that loads, in the order given by `GUI_BACKENDS` in `src/check_spectra.py` (`macosx`, `qtagg`, `tkagg`), and switches back to the notebook backend when it finishes. If no desktop backend is available (for example on a remote server), it falls back to a static plot with the flagged points numbered and a text prompt to choose which to remove.

To skip the manual step, use `interactive=False` (all flagged points are removed) or `check_outliers=False` (no outlier check).
