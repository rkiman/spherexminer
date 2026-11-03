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
