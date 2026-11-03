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
- It shows an interactive window to flag extreme outliers by hand (`check_outliers=True`).

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
