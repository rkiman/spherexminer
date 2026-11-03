
import os
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.ticker import AutoMinorLocator
from astropy.table import Table
from astropy.stats import sigma_clipped_stats
from .extras import *
from .flux_renormalize import *

# Keys we use ('n', 'u', 'a', 's', 'q') collide with matplotlib's default
# keyboard shortcuts - notably 's' opens a "save figure" dialog and 'q'
# closes the window - so both our handler and the built-in one fire on
# the same keypress. Disabling these default keymaps for the duration
# of the tool (restored afterwards) avoids that.
_KEYMAPS_TO_CLEAR = [
    'keymap.save', 'keymap.quit', 'keymap.quit_all', 'keymap.fullscreen',
    'keymap.grid', 'keymap.grid_minor', 'keymap.yscale', 'keymap.xscale',
    'keymap.home', 'keymap.back', 'keymap.forward', 'keymap.pan',
    'keymap.zoom', 'keymap.help', 'keymap.all_axes',
]

DEFAULT_FIL_PHOT_SYNT = ['J_2MASS', 'H_2MASS', 'K_2MASS', 'W1', 'I1', 'I2', 'Lp_MKO', 'Mp_MKO']
DEFAULT_FILTER_PATH = '/Users/rociokiman/Hunter College Dropbox/Rocio Kiman/papers/_Published/Coconuts2/Data/Filters/'

# every column that follows the '<prefix>_<perc>' naming convention -
# when a percentile is chosen, all of the OTHER percentiles' versions
# of these columns get dropped, and the chosen one gets renamed to
# just '<prefix>' (suffix removed)
PERC_COLUMN_PREFIXES = ('Flux', 'Fluxerr', 'r_aperture', 'r_in', 'r_out')

# Safety cap on how many times we'll redisplay-and-recheck the best
# percentile for a single spectrum before just saving with whatever it
# last settled on (guards against pathological flip-flopping).
MAX_REDISPLAYS = 5


def _synth_phot(spectrum, wave_full, perc, phot_measurement, phot_err_measurement,
                 exclude_mask, fil_phot_synt, filter_path):
    """
    Synthetic photometry (in fil_phot_synt) for one spectrum at one flux
    percentile, skipping rows in `exclude_mask` (e.g. points already
    marked bad by hand) on top of the usual NaN/non-positive flux mask.
    Returns (mag, mag_err), or (None, None) if too few points remain.
    """
    col_flux = 'Flux_' + str(perc)
    col_fluxerr = 'Fluxerr_' + str(perc)
    flux_col = np.array(spectrum[col_flux])

    valid = np.array([True if x > 0 and str(x) != 'nan' else False for x in flux_col])
    valid &= ~exclude_mask

    if valid.sum() < 5:
        return None, None

    wave = wave_full[valid]
    flux = flux_col[valid]
    fluxerr = np.array(spectrum[col_fluxerr])[valid]

    flux_erg = jy_to_erg_s_cm2_AA(flux * 1e-6, wave)
    fluxerr_erg = jy_to_erg_s_cm2_AA(fluxerr * 1e-6, wave)

    _, _, _, mag, emag, _, _, _, _ = renormalize_spectrum(
        fil_phot_synt, phot_measurement, phot_err_measurement,
        wave, flux_erg, fluxerr_erg, filter_path=filter_path)
    return np.array(mag), np.array(emag)


def _pick_best_perc(spectrum, wave_full, perc_options, phot_measurement, phot_err_measurement,
                     exclude_mask, fil_phot_synt, filter_path):
    """
    Try every percentile in perc_options, and return whichever one's
    synthetic photometry best matches the literature photometry
    (smallest summed |synthetic - literature| across fil_phot_synt).
    """
    results = {}
    for p in perc_options:
        mag, emag = _synth_phot(spectrum, wave_full, p, phot_measurement, phot_err_measurement,
                                 exclude_mask, fil_phot_synt, filter_path)
        if mag is not None:
            results[p] = (mag, emag)

    if not results:
        # fall back to the first option if nothing had enough points
        return perc_options[0], None, None

    perc_best = min(results, key=lambda p: np.nansum(np.abs(results[p][0] - phot_measurement)))
    mag_best, emag_best = results[perc_best]
    return perc_best, mag_best, emag_best


def clean_spectra_interactively(spectra, idx, known_data, output_dir='spectra/clean', pick_radius=10,
                                 perc_options=(0.9, 0.95, 0.98), fil_phot_synt=None, filter_path=None):
    """

    Controls, while a spectrum's window is open:
      left-click near a point   toggle that point for removal (shows a red x)
      'n'                       save (or, if the best percentile changed, redisplay for review)
      'u'                       undo - go back to the state when this round's window opened
      'a'                       clear all removals (keep every point) - restart selection from scratch
      's'                       skip this spectrum (nothing saved) and move on
      'q'                       stop the whole loop

    Parameters
    ----------
    spectra : list of str
        Paths to the FITS spectra to clean, in order.
    idx : sequence
        idx[i] is the row in `known_data` with the literature photometry
        for spectra[i] (same convention used elsewhere in this project).
    known_data : astropy.table.Table
        Table with literature photometry columns named after
        fil_phot_synt (plus '<filter>_err' columns).
    output_dir : str
        Where cleaned FITS files are written (created if missing).
    pick_radius : float
        Click tolerance in screen pixels for selecting the nearest point.
    perc_options : sequence
        All percentile values that appear as Flux_<perc>/Fluxerr_<perc>
        column suffixes in the input spectra.
    fil_phot_synt : list or None
        Filters to synthesize for the best-percentile comparison.
        Defaults to DEFAULT_FIL_PHOT_SYNT.
    filter_path : str or None
        Passed through to renormalize_spectrum. Defaults to
        DEFAULT_FILTER_PATH.

    Returns
    -------
    dict with keys 'data_phot', 'data_phot_err', 'syn_phot',
    'syn_phot_err' 
    """
    os.makedirs(output_dir, exist_ok=True)
    if fil_phot_synt is None:
        fil_phot_synt = DEFAULT_FIL_PHOT_SYNT
    if filter_path is None:
        filter_path = DEFAULT_FILTER_PATH


    keys_present = [k for k in _KEYMAPS_TO_CLEAR if k in mpl.rcParams]
    old_keymaps = {k: list(mpl.rcParams[k]) for k in keys_present}
    for k in keys_present:
        mpl.rcParams[k] = []

    try:
        phot_results = _clean_loop(spectra, idx, known_data, output_dir, pick_radius, perc_options,
                                    fil_phot_synt, filter_path)
    finally:
        for k, v in old_keymaps.items():
            mpl.rcParams[k] = v

    if len(phot_results['data_phot']) == 0:
        print('Nothing was saved this run - returning empty arrays.')

    return {
        'data_phot': np.array(phot_results['data_phot']),
        'data_phot_err': np.array(phot_results['data_phot_err']),
        'syn_phot': np.array(phot_results['syn_phot']),
        'syn_phot_err': np.array(phot_results['syn_phot_err']),
        'best_perc': np.array(phot_results['best_perc']),
    }


def _clean_loop(spectra, idx, known_data, output_dir, pick_radius, perc_options,
                 fil_phot_synt, filter_path):
    perc_options = list(perc_options)
    phot_results = {'data_phot': [], 'data_phot_err': [], 'syn_phot': [], 'syn_phot_err': [], 'best_perc': []}

    fig, ax = plt.subplots(figsize=(10, 5))
    plt.show(block=False)

    quit_all = False

    try:
        for i, spectrum_path in enumerate(spectra):
            spectrum = Table.read(spectrum_path, format='fits')
            wave_full = np.array(spectrum['Wave'])

            i_known_data = idx[i]
            phot_measurement = np.array([known_data[fil][i_known_data] for fil in fil_phot_synt])
            phot_err_measurement = np.array([known_data[fil + '_err'][i_known_data] for fil in fil_phot_synt])

            no_exclusions = np.zeros(len(spectrum), dtype=bool)

            # check for a previously-saved clean version of this spectrum
            out_path = os.path.join(output_dir, os.path.basename(spectrum_path))
            already_cleaned = os.path.exists(out_path)


            reused_saved_perc = False
            if already_cleaned:
                clean_table = Table.read(out_path)
                removed_wave_set = set(
                    wave_full[~np.isin(wave_full, np.array(clean_table['Wave']))].tolist())
                if 'PERC' in clean_table.meta:
                    perc_saved = clean_table.meta['PERC']

                    perc_display = next((p for p in perc_options if float(p) == float(perc_saved)), perc_saved)
                    reused_saved_perc = True
                else:
                    perc_display, _, _ = _pick_best_perc(spectrum, wave_full, perc_options, phot_measurement,
                                                          phot_err_measurement, no_exclusions, fil_phot_synt, filter_path)
            else:
                removed_wave_set = set()
                perc_display, _, _ = _pick_best_perc(spectrum, wave_full, perc_options, phot_measurement,
                                                      phot_err_measurement, no_exclusions, fil_phot_synt, filter_path)

            skip_this = False
            saved_this_spectrum = False
            redisplay_round = 0


            while not saved_this_spectrum and not skip_this and not quit_all:
                col_flux = 'Flux_' + str(perc_display)
                col_fluxerr = 'Fluxerr_' + str(perc_display)

                flux_full = np.array(spectrum[col_flux])
                fluxerr_full = np.array(spectrum[col_fluxerr])

                valid = np.array([True if x > 0 and str(x) != 'nan' else False for x in flux_full])
                valid_idx = np.where(valid)[0]
                wave = wave_full[valid_idx]
                flux = flux_full[valid_idx]
                fluxerr = fluxerr_full[valid_idx]

                removed = np.isin(wave, list(removed_wave_set))

                point_color = 'tab:green' if already_cleaned else 'tab:orange'

                ax.clear()
                ax.errorbar(wave, flux, yerr=fluxerr, fmt='none', ecolor='lightgray', zorder=0)
                ax.plot(wave, flux, 'o', color=point_color, markersize=4, zorder=1)
                marks_bad, = ax.plot([], [], 'x', color='red', markersize=9, markeredgewidth=2, zorder=5)
                ax.set_xlabel(r'Wavelength ($\mu$m)')
                ax.set_ylabel(r'Flux ($\mu$Jy)')

                round_tag = f"   [review round {redisplay_round + 1}]" if redisplay_round > 0 else ""
                status = (f"{removed.sum()} pt(s) currently marked removed"
                          if (already_cleaned or redisplay_round > 0) else "new spectrum, never cleaned")
                ax.set_title(f"{os.path.basename(spectrum_path)}   [{i+1}/{len(spectra)}]   perc={perc_display}   "
                            f"{status}{round_tag}\n"
                            "click=toggle point, n=save+next, u=undo, a=keep all (restart), s=skip, q=quit")

                print(f"[{i+1}/{len(spectra)}] {os.path.basename(spectrum_path)}: displaying perc={perc_display}"
                      + (" (matching saved clean version)" if reused_saved_perc and redisplay_round == 0 else "")
                      + (f", review round {redisplay_round + 1}" if redisplay_round > 0 else "")
                      + f". {removed.sum()} point(s) currently marked removed.")

                baseline_removed = removed.copy()  # what 'u' undoes back to (start of this round)
                if removed.any():
                    marks_bad.set_data(wave[removed], flux[removed])

                state = {'done': False, 'action': None}

                def redraw():
                    marks_bad.set_data(wave[removed], flux[removed])
                    fig.canvas.draw_idle()

                def on_click(event):
                    if event.inaxes != ax or event.xdata is None or event.ydata is None:
                        return
                    disp = ax.transData.transform(np.column_stack([wave, flux]))
                    click_disp = ax.transData.transform((event.xdata, event.ydata))
                    dist = np.hypot(disp[:, 0] - click_disp[0], disp[:, 1] - click_disp[1])
                    j = np.argmin(dist)
                    if dist[j] < pick_radius:
                        removed[j] = not removed[j]
                        redraw()

                def on_key(event):
                    if event.key == 'n':
                        state['action'] = 'save'
                    elif event.key == 'u':
                        removed[:] = baseline_removed.copy()
                        redraw()
                        return
                    elif event.key == 'a':
                        removed[:] = False
                        redraw()
                        return
                    elif event.key == 's':
                        state['action'] = 'skip'
                    elif event.key == 'q':
                        state['action'] = 'quit'
                    else:
                        return
                    state['done'] = True

                def on_close(event):
                    # window closed via the OS close button rather than a key
                    if not state['done']:
                        state['action'] = 'quit'
                        state['done'] = True

                cid_click = fig.canvas.mpl_connect('button_press_event', on_click)
                cid_key = fig.canvas.mpl_connect('key_press_event', on_key)
                cid_close = fig.canvas.mpl_connect('close_event', on_close)

                fig.canvas.draw_idle()
                while not state['done']:
                    if not plt.fignum_exists(fig.number):
                        state['action'] = 'quit'
                        state['done'] = True
                        break
                    plt.pause(0.05)

                for cid in (cid_click, cid_key, cid_close):
                    try:
                        fig.canvas.mpl_disconnect(cid)
                    except Exception:
                        pass

                if state['action'] == 'quit':
                    quit_all = True
                    break

                if state['action'] == 'skip':
                    skip_this = True
                    break

                # 'save' pressed - lock in this round's removals (by
                # wavelength) and check whether the best percentile for the
                # resulting cleaned spectrum still matches what's displayed
                removed_wave_set = set(wave[removed].tolist())
                manually_removed_mask = np.isin(wave_full, list(removed_wave_set))

                perc_best, phot_synt_mag, ephot_synt_mag = _pick_best_perc(
                    spectrum, wave_full, perc_options, phot_measurement, phot_err_measurement,
                    manually_removed_mask, fil_phot_synt, filter_path)

                if perc_best != perc_display and redisplay_round < MAX_REDISPLAYS:
                    print(f"  best percentile for this cleaned selection is {perc_best}, not the displayed "
                          f"{perc_display} - redisplaying at {perc_best} so you can review it before saving.")
                    perc_display = perc_best
                    redisplay_round += 1
                    continue

                if perc_best != perc_display:
                    print(f"  percentile did not stabilize after {MAX_REDISPLAYS} review rounds "
                          f"(last recomputed value: {perc_best}) - saving with {perc_best} without further review.")
                    perc_display = perc_best

                # --- percentile is stable: perc_display is exactly what
                # was reviewed AND what's about to be saved ---
                if phot_synt_mag is not None:
                    for fil, syn, lit in zip(fil_phot_synt, phot_synt_mag, phot_measurement):
                        print(f"    {fil}: synthetic={syn:.3f}  literature={lit:.3f}")

                col_flux_best = 'Flux_' + str(perc_display)
                col_fluxerr_best = 'Fluxerr_' + str(perc_display)
                flux_best_full = np.array(spectrum[col_flux_best])
                valid_best = np.array([True if x > 0 and str(x) != 'nan' else False for x in flux_best_full])

                # rows: valid for the chosen percentile, and not marked for removal
                keep_mask = valid_best & (~manually_removed_mask)

                cleaned = spectrum[keep_mask]

                # columns: for every '<prefix>_<perc>' family (Flux, Fluxerr,
                # r_aperture, r_in, r_out, ...), drop every percentile except
                # the chosen one, and rename the survivor so the _<perc>
                # suffix is gone
                best_col_names = {f'{prefix}_{perc_display}': prefix for prefix in PERC_COLUMN_PREFIXES}
                other_perc_cols = {
                    f'{prefix}_{p}' for p in perc_options for prefix in PERC_COLUMN_PREFIXES
                } - set(best_col_names)
                cols_to_drop = [c for c in other_perc_cols if c in cleaned.colnames]
                if cols_to_drop:
                    cleaned.remove_columns(cols_to_drop)
                for old_name, new_name in best_col_names.items():
                    if old_name in cleaned.colnames:
                        cleaned.rename_column(old_name, new_name)

                # header: PERC currently lists all the percentile options -
                # overwrite it with just the one actually selected
                cleaned.meta['PERC'] = perc_display

                cleaned.write(out_path, format='fits', overwrite=True)
                n_removed_clicks = int(manually_removed_mask.sum())
                n_nan_dropped = int((~valid_best).sum())
                print(f'Saved {out_path}  (perc={perc_display}, {n_removed_clicks} point(s) removed by hand, '
                      f'{n_nan_dropped} NaN/invalid point(s) dropped)')

                # stash this object's photometry for the final comparison figure
                phot_results['data_phot'].append(phot_measurement)
                phot_results['data_phot_err'].append(phot_err_measurement)
                if phot_synt_mag is not None:
                    phot_results['syn_phot'].append(phot_synt_mag)
                    phot_results['syn_phot_err'].append(ephot_synt_mag)
                else:
                    nan_row = np.full(len(fil_phot_synt), np.nan)
                    phot_results['syn_phot'].append(nan_row)
                    phot_results['syn_phot_err'].append(nan_row)
                phot_results['best_perc'].append(perc_display)

                saved_this_spectrum = True

            if quit_all:
                print('Stopped early - nothing after this point was processed.')
                break
            if skip_this:
                print(f'Skipped (not saved): {spectrum_path}')
                continue
    finally:
        if plt.fignum_exists(fig.number):
            plt.close(fig)

    return phot_results


# Example usage:
# %matplotlib widget
# phot = clean_spectra_interactively(spectra, idx, known_data, output_dir='spectra/clean')
# data_phot, syn_phot = phot['data_phot'], phot['syn_phot']
# best_perc = phot['best_perc']


# v2 cleaning: multi-epoch points combined per official SPHEREx channel, merged from spherex_spectra_v2.py.
DEFAULT_PERC = 0.95                      # used when the aperture cannot be selected
PERC_OPTIONS = (0.9, 0.95, 0.98)         # apertures available in the raw extractions
CHANNEL_COLUMN = 'CHANNEL'              # channel id (100*DETECTOR + SUBCHAN) written by the extraction
DEFAULT_SELECTION_FILTERS = ['J_2MASS', 'H_2MASS', 'K_2MASS', 'W1', 'I1', 'Lp_MKO', 'Mp_MKO']


def _match_keys(wave, mjd):
    return list(zip(np.round(np.asarray(wave, dtype=float), 8), np.round(np.asarray(mjd, dtype=float), 6)))


def select_aperture(raw, perc=DEFAULT_PERC):
    """Measurements of a raw spherexminer extraction at one aperture."""
    if isinstance(raw, str):
        raw = Table.read(raw, format='fits')
    p = str(perc)
    if 'Flux_' + p not in raw.colnames:
        raise KeyError('Aperture %s not in the raw spectrum (columns: %s)' % (p, raw.colnames))
    cols = [raw['Wave']] + [raw[prefix + '_' + p] for prefix in PERC_COLUMN_PREFIXES if prefix + '_' + p in raw.colnames]
    names = ['Wave'] + [prefix for prefix in PERC_COLUMN_PREFIXES if prefix + '_' + p in raw.colnames]
    t = Table([np.array(c) for c in cols], names=names, meta=dict(raw.meta))
    t['OBSMJD'] = np.array(raw['OBSMJD'])
    # Channel measured at extraction (newer extractions); used to group the points per channel.
    if CHANNEL_COLUMN in raw.colnames:
        t[CHANNEL_COLUMN] = np.array(raw[CHANNEL_COLUMN])
    t.meta['PERC'] = float(perc)
    return t


def assign_channels(points, channel_table=None):
    """
    Row of the channel table for each measurement.

    If the points have the CHANNEL column written by the spectrum extraction
    (100*DETECTOR + SUBCHAN, from the pixel CHANNEL_MAP), each point is combined with the
    other epochs in that same channel. Spectra without the column (older extractions), and
    points with an invalid channel (e.g. -1, masked), are assigned from the wavelength only
    (assign_spherex_channels).
    """
    ch = read_spherex_channels(channel_table)
    wave = np.atleast_1d(np.asarray(points['Wave'], dtype=float))
    row = assign_spherex_channels(wave, ch)
    if CHANNEL_COLUMN not in points.colnames:
        return row
    lookup = {int(cid): i for i, cid in enumerate(channel_ids(ch))}
    chan_pts = np.ma.filled(np.ma.asarray(points[CHANNEL_COLUMN]), -1).astype(int)
    r = np.array([lookup.get(int(c), -1) for c in chan_pts], dtype=int)
    ok = r >= 0
    row[ok] = r[ok]
    return row


def in_reference(points, reference):
    """True for measurements present in a reference (e.g. v1 hand-cleaned) spectrum, matched by
    wavelength and observation date. Measurements removed by hand in the reference are False."""
    if isinstance(reference, str):
        reference = Table.read(reference, format='fits')
    ref_keys = set(_match_keys(reference['Wave'], reference['OBSMJD']))
    return np.array([k in ref_keys for k in _match_keys(points['Wave'], points['OBSMJD'])])


def combine_per_channel(points, channel_table=None, error_method='birge', clip_sigma=None):
    """Combine all measurements in the same SPHEREx channel."""
    ch = read_spherex_channels(channel_table)
    wave = np.asarray(points['Wave'], dtype=float)
    flux = np.asarray(points['Flux'], dtype=float)
    err = np.asarray(points['Fluxerr'], dtype=float)
    rap = np.asarray(points['r_aperture'], dtype=float) if 'r_aperture' in points.colnames else np.full(len(wave), np.nan)
    mjd = np.asarray(points['OBSMJD'], dtype=float) if 'OBSMJD' in points.colnames else np.full(len(wave), np.nan)

    good = np.isfinite(wave) & np.isfinite(flux) & np.isfinite(err) & (err > 0)
    row = assign_channels(points[good], ch)
    wave, flux, err, rap, mjd = wave[good], flux[good], err[good], rap[good], mjd[good]

    det = np.asarray(ch['DETECTOR'], dtype=int)
    sub = np.asarray(ch['SUBCHAN'], dtype=int)
    wl_c = np.asarray(ch['WAVELENGTH'], dtype=float)
    bw = np.asarray(ch['BANDWIDTH'], dtype=float)

    out = {k: [] for k in ['Wave', 'Bandwidth', 'Wave_mean', 'Wave_min', 'Wave_max', 'Flux', 'Fluxerr', 'Fluxerr_stat', 'Chi2red',
                           'Npoints', 'Nclipped', 'Detector', 'Subchannel', 'Channel', 'r_aperture',
                           'OBSMJD_min', 'OBSMJD_max']}
    for k in np.unique(row):
        m = np.where(row == k)[0]
        keep = np.ones(len(m), dtype=bool)
        if clip_sigma is not None:
            while keep.sum() >= 3:
                w = 1.0 / err[m][keep] ** 2
                mean = np.sum(w * flux[m][keep]) / np.sum(w)
                dev = np.abs(flux[m] - mean) / err[m]
                new_keep = keep & (dev <= clip_sigma)
                if new_keep.sum() == keep.sum() or new_keep.sum() < 2:
                    break
                keep = new_keep
        mk = m[keep]
        n = len(mk)
        w = 1.0 / err[mk] ** 2
        f_mean = np.sum(w * flux[mk]) / np.sum(w)
        e_stat = 1.0 / np.sqrt(np.sum(w))
        chi2_red = np.sum(w * (flux[mk] - f_mean) ** 2) / (n - 1) if n > 1 else np.nan
        if n > 1 and error_method == 'birge':
            e_tot = e_stat * np.sqrt(max(1.0, chi2_red))
        elif n > 1 and error_method == 'quadrature':
            e_tot = np.sqrt(e_stat ** 2 + np.std(flux[mk], ddof=1) ** 2 / n)
        elif n > 1:
            raise ValueError("error_method must be 'birge' or 'quadrature'")
        else:
            e_tot = e_stat

        out['Wave'].append(wl_c[k])
        out['Bandwidth'].append(bw[k])
        out['Wave_mean'].append(np.sum(w * wave[mk]) / np.sum(w))
        out['Wave_min'].append(np.min(wave[mk]))
        out['Wave_max'].append(np.max(wave[mk]))
        out['Flux'].append(f_mean)
        out['Fluxerr'].append(e_tot)
        out['Fluxerr_stat'].append(e_stat)
        out['Chi2red'].append(chi2_red)
        out['Npoints'].append(n)
        out['Nclipped'].append(len(m) - n)
        out['Detector'].append(det[k])
        out['Subchannel'].append(sub[k])
        out['Channel'].append(100 * det[k] + sub[k])
        out['r_aperture'].append(np.nanmedian(rap[mk]) if np.isfinite(rap[mk]).any() else np.nan)
        out['OBSMJD_min'].append(np.nanmin(mjd[mk]) if np.isfinite(mjd[mk]).any() else np.nan)
        out['OBSMJD_max'].append(np.nanmax(mjd[mk]) if np.isfinite(mjd[mk]).any() else np.nan)

    t = Table(out)
    t.sort('Wave')

    desc = {
        'Wave': ('um', 'Official SPHEREx channel band center (median per-pixel wavelength)'),
        'Bandwidth': ('um', 'Official channel bandwidth (band center / R)'),
        'Wave_mean': ('um', 'Inverse-variance weighted mean of the measured wavelengths'),
        'Wave_min': ('um', 'Shortest measured wavelength combined in the channel'),
        'Wave_max': ('um', 'Longest measured wavelength combined in the channel'),
        'Flux': ('uJy', 'Inverse-variance weighted mean flux density of all epochs in the channel'),
        'Fluxerr': ('uJy', '1-sigma uncertainty including the epoch-to-epoch scatter'),
        'Fluxerr_stat': ('uJy', '1-sigma uncertainty from the measurement errors only'),
        'Chi2red': (None, 'Reduced chi2 of the measurements around the weighted mean'),
        'Npoints': (None, 'Number of measurements (epochs) combined'),
        'Nclipped': (None, 'Number of measurements rejected by sigma clipping'),
        'Detector': (None, 'SPHEREx detector (band) 1-6'),
        'Subchannel': (None, 'Channel within the detector 1-17'),
        'Channel': (None, 'Channel identifier 100*Detector + Subchannel'),
        'r_aperture': ('pix', 'Median photometric aperture radius'),
        'OBSMJD_min': ('d', 'First observation date (MJD) in the channel'),
        'OBSMJD_max': ('d', 'Last observation date (MJD) in the channel'),
    }
    for c, (unit, d) in desc.items():
        if unit is not None:
            t[c].unit = unit
        t[c].description = d
    return t


def _running_median(wave, flux, half_window=7):
    """Median of the flux of the 2*half_window nearest measurements in wavelength (excluding itself)."""
    order = np.argsort(wave)
    f = flux[order]
    n = len(f)
    med = np.empty(n)
    for j in range(n):
        lo, hi = max(0, j - half_window), min(n, j + half_window + 1)
        neigh = np.concatenate([f[lo:j], f[j + 1:hi]])
        med[j] = np.median(neigh) if len(neigh) else f[j]
    out = np.empty(n)
    out[order] = med
    return out


def _running_mad(wave, resid, half_window=15):
    """Robust (MAD) scatter of the residuals of the 2*half_window nearest measurements (excluding
    itself)."""
    order = np.argsort(wave)
    r = resid[order]
    n = len(r)
    out = np.empty(n)
    for j in range(n):
        neigh = np.concatenate([r[max(0, j - half_window):j], r[j + 1:j + half_window + 1]])
        out[j] = 1.4826 * np.median(np.abs(neigh - np.median(neigh))) if len(neigh) else 0.0
    res = np.empty(n)
    res[order] = out
    return res


def find_extreme_outliers(points, nsigma=8.0, half_window=7, scatter='local', scatter_window=15):
    """Flag extreme outliers among the individual measurements of one aperture."""
    wave = np.asarray(points['Wave'], dtype=float)
    flux = np.asarray(points['Flux'], dtype=float)
    err = np.asarray(points['Fluxerr'], dtype=float)
    flag = np.zeros(len(wave), dtype=bool)
    good = np.isfinite(wave) & np.isfinite(flux) & np.isfinite(err)
    if good.sum() < 5:
        return flag
    resid = flux[good] - _running_median(wave[good], flux[good], half_window)
    if scatter == 'local':
        s = _running_mad(wave[good], resid, scatter_window)
    elif scatter == 'global':
        s = 1.4826 * np.median(np.abs(resid - np.median(resid)))
    else:
        raise ValueError("scatter must be 'local' or 'global'")
    z = resid / np.sqrt(err[good] ** 2 + s ** 2)
    flag[np.where(good)[0]] = np.abs(z) > nsigma
    return flag
    resid = flux[good] - _running_median(wave[good], flux[good], half_window)
    scatter = 1.4826 * np.median(np.abs(resid - np.median(resid)))
    z = resid / np.sqrt(err[good] ** 2 + scatter ** 2)
    flag[np.where(good)[0]] = np.abs(z) > nsigma
    return flag


def _switch_to_gui_backend():
    """The click-and-wait loop only works in a desktop window."""
    import matplotlib
    import matplotlib.pyplot as plt
    orig = matplotlib.get_backend()
    low = orig.lower()
    if not (any(k in low for k in _NOTEBOOK_BACKEND_KEYS) or low in _NONINTERACTIVE_BACKENDS):
        return orig, orig                      # already a desktop backend (e.g. %matplotlib qt / osx)
    for b in GUI_BACKENDS:
        try:
            plt.switch_backend(b)
            return orig, b
        except Exception:
            continue
    return orig, None


def _select_points_by_prompt(points, flagged, title=''):
    """Fallback without a GUI: static plot with the flagged points numbered + a text prompt."""
    import matplotlib.pyplot as plt
    wave = np.asarray(points['Wave'], dtype=float)
    flux = np.asarray(points['Flux'], dtype=float)
    idx = np.where(flagged)[0]
    fig, ax = plt.subplots(figsize=(10, 4))
    ax.scatter(wave, flux, s=8, c='k')
    ax.scatter(wave[idx], flux[idx], s=60, facecolors='none', edgecolors='r')
    for i, j in enumerate(idx):
        ax.annotate(str(i), (wave[j], flux[j]), xytext=(4, 4), textcoords='offset points', color='r')
    ax.set_xlabel('Wavelength [um]')
    ax.set_ylabel('Flux [uJy]')
    ax.set_title(title, fontsize=9)
    plt.show()
    for i, j in enumerate(idx):
        print('  %d: Wave = %.4f um, Flux = %.1f uJy, MJD = %.4f' % (i, wave[j], flux[j], points['OBSMJD'][j]))
    ans = input('Numbers of the flagged points to remove (e.g. 0,2), Enter = all, "none" = keep all: ').strip().lower()
    remove = np.zeros(len(wave), dtype=bool)
    if ans == '':
        remove[idx] = True
    elif ans != 'none':
        for tok in ans.replace(' ', ',').split(','):
            if tok.strip().isdigit() and int(tok) < len(idx):
                remove[idx[int(tok)]] = True
    return remove


def select_points_interactively(points, flagged=None, title='', preselect=True):
    """Let the user choose which measurements to remove (see _select_points_gui)."""
    import matplotlib.pyplot as plt
    flagged = np.zeros(len(points), dtype=bool) if flagged is None else np.asarray(flagged, dtype=bool)
    orig, gui = _switch_to_gui_backend()
    if gui is None:
        return _select_points_by_prompt(points, flagged, title)
    try:
        return _select_points_gui(points, flagged, title, preselect)
    except Exception as exc:
        print('Could not open the plot window (%s); using a text prompt instead.' % exc)
        if gui != orig:
            plt.switch_backend(orig)
            gui = orig
        return _select_points_by_prompt(points, flagged, title)
    finally:
        if gui != orig:
            plt.switch_backend(orig)


def _select_points_gui(points, flagged=None, title='', preselect=True):
    """Plot the individual measurements and let the user choose which ones to remove."""
    import matplotlib.pyplot as plt
    from matplotlib.widgets import Button, RectangleSelector

    wave = np.asarray(points['Wave'], dtype=float)
    flux = np.asarray(points['Flux'], dtype=float)
    err = np.asarray(points['Fluxerr'], dtype=float)
    n = len(wave)
    flagged = np.zeros(n, dtype=bool) if flagged is None else np.asarray(flagged, dtype=bool)
    selected = flagged.copy() if preselect else np.zeros(n, dtype=bool)

    fig, axes = plt.subplots(2, 1, figsize=(11, 8), sharex=True)
    fig.subplots_adjust(bottom=0.14, hspace=0.08, top=0.92)
    fig.suptitle('%s\n%d measurements, %d flagged as extreme outliers (red rings). '
                 'Click / drag to select points to remove.' % (title, n, flagged.sum()), fontsize=10)
    ok = ~flagged & np.isfinite(flux)
    if ok.sum() > 2:
        lo, hi = np.nanpercentile(flux[ok], [0.5, 99.5])
        pad = 0.15 * (hi - lo)
        zoom_ylim = (lo - pad, hi + pad)
    else:
        zoom_ylim = None

    sel_artists = []
    for ax in axes:
        ax.errorbar(wave, flux, yerr=err, fmt='none', ecolor='0.75', lw=0.6, zorder=1)
        ax.scatter(wave, flux, s=10, c='k', zorder=2)
        ax.scatter(wave[flagged], flux[flagged], s=70, facecolors='none', edgecolors='r', lw=1.2, zorder=3)
        sel_artists.append(ax.scatter([], [], s=40, marker='x', c='tab:blue', lw=1.5, zorder=4))
        ax.set_ylabel('Flux [uJy]')
    if zoom_ylim is not None:
        axes[1].set_ylim(zoom_ylim)
    axes[1].set_xlabel('Wavelength [um]')
    axes[0].set_title('full range', fontsize=9, loc='left')
    axes[1].set_title('zoom', fontsize=9, loc='left')
    status = fig.text(0.01, 0.02, '', fontsize=9)

    def redraw():
        xy = np.column_stack([wave[selected], flux[selected]]) if selected.any() else np.empty((0, 2))
        for art in sel_artists:
            art.set_offsets(xy)
        status.set_text('%d points selected for removal (blue x)' % selected.sum())
        fig.canvas.draw_idle()

    def toolbar_busy():
        tb = getattr(fig.canvas, 'toolbar', None)
        return tb is not None and str(getattr(tb, 'mode', '')) != ''

    press = {}

    def on_press(event):
        if event.inaxes in axes and event.button == 1:
            press['xy'] = (event.x, event.y)

    def on_release(event):
        start = press.pop('xy', None)
        if start is None or event.inaxes not in axes or toolbar_busy():
            return
        if np.hypot(event.x - start[0], event.y - start[1]) > 5:
            return                                     # a drag, handled by the box selector
        disp = event.inaxes.transData.transform(np.column_stack([wave, flux]))
        d = np.hypot(disp[:, 0] - event.x, disp[:, 1] - event.y)
        d[~np.isfinite(d)] = np.inf
        j = int(np.argmin(d))
        if d[j] < 10:
            selected[j] = ~selected[j]
            redraw()

    def on_box(eclick, erelease):
        if toolbar_busy():
            return
        x0, x1 = sorted([eclick.xdata, erelease.xdata])
        y0, y1 = sorted([eclick.ydata, erelease.ydata])
        inside = (wave >= x0) & (wave <= x1) & (flux >= y0) & (flux <= y1)
        deselect = eclick.key is not None and 'shift' in eclick.key
        selected[inside] = not deselect
        redraw()

    selectors = [RectangleSelector(ax, on_box, useblit=True, button=[1], minspanx=5, minspany=5,
                                   spancoords='pixels', interactive=False) for ax in axes]

    state = {'done': False}

    def finish(*args):
        # idempotent; the wait loop below polls this flag (no reliance on stop_event_loop)
        state['done'] = True

    def set_flagged(event):
        selected[:] = flagged
        redraw()

    def clear(event):
        selected[:] = False
        redraw()

    b_flag = Button(fig.add_axes([0.55, 0.015, 0.13, 0.05]), 'Select flagged')
    b_clear = Button(fig.add_axes([0.70, 0.015, 0.10, 0.05]), 'Clear')
    b_done = Button(fig.add_axes([0.82, 0.015, 0.10, 0.05]), 'Done')
    b_flag.on_clicked(set_flagged)
    b_clear.on_clicked(clear)
    b_done.on_clicked(finish)
    fig.canvas.mpl_connect('button_press_event', on_press)
    fig.canvas.mpl_connect('button_release_event', on_release)
    fig.canvas.mpl_connect('key_press_event', lambda e: finish() if e.key == 'enter' else None)
    cid_close = fig.canvas.mpl_connect('close_event', finish)
    fig._outlier_widgets = (selectors, b_flag, b_clear, b_done)     # keep references alive

    redraw()
    plt.show(block=False)
    _raise_window(fig)
    print('Select the points to remove in the plot window, then press Done '
          '(if you do not see it, it may be behind the browser).')
    # Run the GUI event loop in short slices until Done/Enter/close. Polling a flag is more
    # robust than start_event_loop(timeout=-1), which can wait forever on the 2nd+ window.
    while not state['done'] and plt.fignum_exists(fig.number):
        fig.canvas.start_event_loop(0.1)
    fig.canvas.mpl_disconnect(cid_close)
    plt.close(fig)
    try:                                               # let the GUI actually remove the window
        plt.pause(0.05) if plt.get_fignums() else fig.canvas.flush_events()
    except Exception:
        pass
    return selected.copy()


def _raise_window(fig):
    """Best effort to bring the figure window to the front (Qt, Tk; macosx raises on show)."""
    win = getattr(fig.canvas.manager, 'window', None)
    if win is None:
        return
    for meth in ('showNormal', 'raise_', 'activateWindow', 'lift', 'focus_force'):
        try:
            getattr(win, meth)()
        except Exception:
            pass
    try:                                               # Tk: pop to the front once
        win.attributes('-topmost', True)
        win.after(300, lambda: win.attributes('-topmost', False))
    except Exception:
        pass


def check_extreme_outliers(raw, perc_options=PERC_OPTIONS, plot_perc=DEFAULT_PERC, reference_clean=None,
                           nsigma=8.0, half_window=7, interactive=True, preselect=True, title=None):
    """Look for extreme outliers in a raw extraction before the aperture is selected."""
    if isinstance(raw, str):
        if title is None:
            title = os.path.basename(raw)
        raw = Table.read(raw, format='fits')
    pts = select_aperture(raw, plot_perc)
    use = np.isfinite(np.asarray(pts['Flux'], dtype=float)) & np.isfinite(np.asarray(pts['Fluxerr'], dtype=float))
    if reference_clean is not None:
        use &= in_reference(pts, reference_clean)
    flagged = np.zeros(len(pts), dtype=bool)
    for p in perc_options:
        if 'Flux_' + str(p) in raw.colnames:
            flagged[use] |= find_extreme_outliers(select_aperture(raw, p)[use], nsigma=nsigma, half_window=half_window)
    n_flagged = int(flagged.sum())
    if n_flagged == 0:
        return [], 0
    sub = pts[use]
    if interactive:
        remove = select_points_interactively(sub, flagged[use], title=(title or '') + '  (aperture %s)' % plot_perc,
                                             preselect=preselect)
    else:
        remove = flagged[use]
    return [k for k, r in zip(_match_keys(sub['Wave'], sub['OBSMJD']), remove) if r], n_flagged


def _remove_keys_from(remove_points):
    """Set of (Wave, OBSMJD) keys from a list of keys or a Table with Wave and OBSMJD."""
    if remove_points is None:
        return set()
    if isinstance(remove_points, str):
        remove_points = Table.read(remove_points)
    if isinstance(remove_points, Table):
        return set(_match_keys(remove_points['Wave'], remove_points['OBSMJD']))
    return set(_match_keys([k[0] for k in remove_points], [k[1] for k in remove_points])) if len(remove_points) else set()


def clean_spectrum_v2(raw, perc=DEFAULT_PERC, reference_clean=None, channel_table=None,
                      error_method='birge', clip_sigma=None, drop_nonpositive=True, perc_info=None,
                      remove_points=None):
    """Clean one raw spherexminer spectrum (v2 method)."""
    raw_path = raw if isinstance(raw, str) else None
    points = select_aperture(raw, perc)
    n_raw = len(points)

    valid = np.isfinite(np.asarray(points['Flux'], dtype=float)) & np.isfinite(np.asarray(points['Fluxerr'], dtype=float))
    #if drop_nonpositive:
    #    valid &= np.asarray(points['Flux'], dtype=float) > 0
    n_invalid = int((~valid).sum())

    if reference_clean is not None:
        in_ref = in_reference(points, reference_clean)
        n_removed_ref = int((valid & ~in_ref).sum())
        valid &= in_ref
    else:
        n_removed_ref = 0

    remove_keys = _remove_keys_from(remove_points)
    if remove_keys:
        is_out = np.array([k in remove_keys for k in _match_keys(points['Wave'], points['OBSMJD'])])
        n_removed_out = int((valid & is_out).sum())
        valid &= ~is_out
    else:
        n_removed_out = 0

    points = points[valid]
    spec = combine_per_channel(points, channel_table, error_method=error_method, clip_sigma=clip_sigma)
    points['Channel'] = channel_ids(channel_table)[assign_channels(points, channel_table)]

    meta = {}
    for key in ['NAME', 'RA', 'DEC', 'PMRA', 'PMDEC', 'EPOCH']:
        if key in points.meta:
            meta[key] = points.meta[key]
    ch_file = channel_table if isinstance(channel_table, str) else DEFAULT_CHANNEL_FILE
    meta.update({
        'PERC': float(perc),
        'PERCSEL': 'fixed',
        'VERSION': 'v2',
        'COMBINE': 'per-channel inverse-variance weighted mean',
        'ERRMETH': error_method,
        'CLIPSIG': -1.0 if clip_sigma is None else float(clip_sigma),
        'CHANFILE': os.path.basename(ch_file) if channel_table is None or isinstance(channel_table, str) else 'user table',
        'CHANSRC': 'channel' if CHANNEL_COLUMN in points.colnames else 'wavelength',
        'NCHAN': len(spec),
        'NMEAS': int(np.sum(spec['Npoints'])),
        'NRAW': n_raw,
        'NINVALID': n_invalid,
        'NREMREF': n_removed_ref,
        'NREMOUT': n_removed_out,
        'REFCLEAN': os.path.basename(reference_clean) if isinstance(reference_clean, str) else ('none' if reference_clean is None else 'table'),
        'RAWFILE': os.path.basename(raw_path) if raw_path else 'table',
    })
    if perc_info:
        meta.update(perc_info)
    meta['comments'] = [
        'NAME: Target identifier',
        'RA, DEC: ICRS coordinates in degrees',
        'PMRA, PMDEC: proper motion in mas/yr',
        'EPOCH: epoch of the position',
        'PERC: aperture correction percentage (encircled-energy fraction) used',
        'PERCSEL: how PERC was chosen: literature = best match to the literature',
        '  photometry among PERCOPT; fixed = given; reference = PERC of REFCLEAN',
        'PERCOPT: apertures compared; PERCFILT: filters used for the selection',
        'PERCRES: mean |synthetic - literature| (mag) for each aperture in PERCOPT',
        'VERSION: v2 cleaning (check_spectra.py)',
        'COMBINE: all epochs in each official SPHEREx channel are combined',
        'Wave, Bandwidth: official channel band center and bandwidth (WAVELENGTH/R)',
        'ERRMETH: birge = error of the weighted mean x sqrt(max(1, chi2_red))',
        'CLIPSIG: sigma-clipping threshold within each channel (-1 = none)',
        'CHANFILE: IRSA SPHEREx spectral channel definition file',
        'CHANSRC: how points were grouped: channel = CHANNEL column of the raw spectrum;',
        '  wavelength = channel assigned from the wavelength only (older extractions)',
        'NCHAN: number of channels with data; NMEAS: measurements combined',
        'NRAW: measurements in the raw extraction',
        'NINVALID: measurements dropped (NaN or flux <= 0)',
        'NREMREF: measurements removed because absent from REFCLEAN',
        'NREMOUT: extreme outliers removed by hand (check_extreme_outliers)',
        'REFCLEAN: reference clean spectrum (points removed by hand)',
        'RAWFILE: raw spherexminer extraction',
        'Columns (one row per channel; f_i, s_i = flux and error of epoch i,',
        '  w_i = 1/s_i^2, N = Npoints):',
        'Wave [um]: official SPHEREx channel band center',
        'Bandwidth [um]: official channel bandwidth (band center / R)',
        'Wave_mean [um]: weighted mean of the measured wavelengths',
        'Wave_min, Wave_max [um]: range of measured wavelengths in the channel',
        'Flux [uJy]: weighted mean flux, sum(w_i f_i) / sum(w_i)',
        'Fluxerr [uJy]: final 1-sigma error including epoch-to-epoch scatter;',
        '  birge: Fluxerr_stat x sqrt(max(1, Chi2red));',
        '  quadrature: sqrt(Fluxerr_stat^2 + std(f_i)^2 / N)',
        'Fluxerr_stat [uJy]: error from the measurement errors only,',
        '  1 / sqrt(sum(w_i))',
        'Chi2red: reduced chi2 of the epochs around the weighted mean,',
        '  sum(w_i (f_i - Flux)^2) / (N - 1); ~1 = scatter consistent with',
        '  the errors, >1 = excess scatter; NaN if N = 1',
        'Npoints: number of measurements (epochs) combined',
        'Nclipped: measurements rejected by sigma clipping (CLIPSIG)',
        'Detector: SPHEREx detector 1-6; Subchannel: channel within detector 1-17',
        'Channel: channel identifier 100 x Detector + Subchannel',
        'r_aperture [pix]: median photometric aperture radius',
        'OBSMJD_min, OBSMJD_max [d]: first and last observation MJD in channel',
    ]
    spec.meta = meta
    return spec, points


def select_aperture_by_photometry(raw, phot_measurement, phot_err_measurement=None,
                                  fil_phot_synt=DEFAULT_SELECTION_FILTERS, filter_path=None,
                                  perc_options=PERC_OPTIONS, reference_clean=None, channel_table=None,
                                  error_method='birge', clip_sigma=None, drop_nonpositive=True,
                                  min_filters=2, remove_points=None):
    """Choose the aperture whose v2 synthetic photometry best matches the literature photometry."""
    if filter_path is None:
        raise ValueError('filter_path is needed to select the aperture')
    lit = np.asarray(phot_measurement, dtype=float)
    results = {}
    for p in perc_options:
        spec, _ = clean_spectrum_v2(raw, perc=p, reference_clean=reference_clean, channel_table=channel_table,
                                    error_method=error_method, clip_sigma=clip_sigma,
                                    drop_nonpositive=drop_nonpositive, remove_points=remove_points)
        mag, emag, _ = synthetic_photometry_v2(spec, fil_phot_synt, filter_path, zero_point_err=False,
                                                   drop_nonpositive=drop_nonpositive)
        results[p] = {'mag': mag, 'emag': emag}
    usable = np.isfinite(lit)
    for p in perc_options:
        usable &= np.isfinite(results[p]['mag'])
    for p in perc_options:
        results[p]['score'] = np.mean(np.abs(results[p]['mag'][usable] - lit[usable])) if usable.sum() else np.nan
    results['filters_used'] = [f for f, u in zip(fil_phot_synt, usable) if u]
    if usable.sum() < min_filters:
        return DEFAULT_PERC, results
    best = min(perc_options, key=lambda p: results[p]['score'])
    return best, results


def write_spectrum_v2(spec, out_path, overwrite=True):
    """Write a v2 spectrum to FITS (header keywords from spec.meta, as in the v1 files)."""
    os.makedirs(os.path.dirname(os.path.abspath(out_path)), exist_ok=True)
    spec.write(out_path, format='fits', overwrite=overwrite)
    return out_path


def clean_sample_v2(raw_paths, out_dir, reference_dir=None, perc='literature', phot_measurements=None,
                    phot_err_measurements=None, fil_phot_synt=DEFAULT_SELECTION_FILTERS, filter_path=None,
                    perc_options=PERC_OPTIONS, channel_table=None, error_method='birge', clip_sigma=None,
                    drop_nonpositive=True, overwrite=True, verbose=True, check_outliers=True,
                    outlier_sigma=8.0, interactive=True, outlier_dir=None, recheck_outliers=False):
    """Clean a list of raw spectra and write them to out_dir with the same file names."""
    os.makedirs(out_dir, exist_ok=True)
    if perc == 'literature' and (phot_measurements is None or filter_path is None):
        raise ValueError("perc='literature' needs phot_measurements and filter_path")
    # Switch to a desktop backend ONCE for the whole sample (switching back and forth for every
    # target is what made the window hang after the first object) and restore it at the end.
    orig_backend, gui_backend = (None, None)
    if check_outliers and interactive:
        orig_backend, gui_backend = _switch_to_gui_backend()
        if verbose and gui_backend is not None and gui_backend != orig_backend:
            print('Interactive outlier check: using the %s backend (restored at the end).' % gui_backend)
    try:
        return _clean_sample_loop(raw_paths, out_dir, reference_dir, perc, phot_measurements,
                                  phot_err_measurements, fil_phot_synt, filter_path, perc_options,
                                  channel_table, error_method, clip_sigma, drop_nonpositive, overwrite,
                                  verbose, check_outliers, outlier_sigma, interactive, outlier_dir,
                                  recheck_outliers)
    finally:
        if gui_backend is not None and gui_backend != orig_backend:
            import matplotlib.pyplot as plt
            plt.close('all')
            plt.switch_backend(orig_backend)


def _clean_sample_loop(raw_paths, out_dir, reference_dir, perc, phot_measurements, phot_err_measurements,
                       fil_phot_synt, filter_path, perc_options, channel_table, error_method, clip_sigma,
                       drop_nonpositive, overwrite, verbose, check_outliers, outlier_sigma, interactive,
                       outlier_dir, recheck_outliers):
    rows = []
    for i, raw_path in enumerate(raw_paths):
        fname = os.path.basename(raw_path)
        out_path = os.path.join(out_dir, fname)
        ref = None
        if reference_dir is not None:
            cand = os.path.join(reference_dir, fname)
            ref = cand if os.path.exists(cand) else None
        if os.path.exists(out_path) and not overwrite:
            if verbose:
                print('exists, skipped:', out_path)
            continue
        perc_ref = np.nan
        if ref is not None:
            hdr_perc = Table.read(ref, format='fits').meta.get('PERC', np.nan)
            try:
                perc_ref = float(hdr_perc)
            except (TypeError, ValueError):
                perc_ref = np.nan

        remove_keys, n_flagged = [], 0
        if check_outliers:
            log_path = None
            if outlier_dir is not None:
                os.makedirs(outlier_dir, exist_ok=True)
                log_path = os.path.join(outlier_dir, fname.replace('.fits', '') + '_removed_points.ecsv')
            if log_path is not None and os.path.exists(log_path) and not recheck_outliers:
                remove_keys = sorted(_remove_keys_from(log_path))
                n_flagged = -1
            else:
                remove_keys, n_flagged = check_extreme_outliers(
                    raw_path, perc_options=perc_options, reference_clean=ref, nsigma=outlier_sigma,
                    interactive=interactive, title=fname)
                if log_path is not None:
                    Table(rows=remove_keys if remove_keys else None, names=['Wave', 'OBSMJD'],
                          dtype=[float, float]).write(log_path, format='ascii.ecsv', overwrite=True)
            if verbose and (n_flagged != 0 or remove_keys):
                print('%s: %s extreme outliers flagged, %d points removed' % (
                    fname, 'saved' if n_flagged < 0 else n_flagged, len(remove_keys)))

        perc_info = {'PERCOPT': ','.join(str(p) for p in perc_options)}
        score = np.nan
        if perc == 'literature':
            perc_i, res = select_aperture_by_photometry(
                raw_path, phot_measurements[i], None if phot_err_measurements is None else phot_err_measurements[i],
                fil_phot_synt=fil_phot_synt, filter_path=filter_path, perc_options=perc_options,
                reference_clean=ref, channel_table=channel_table, error_method=error_method,
                clip_sigma=clip_sigma, drop_nonpositive=drop_nonpositive, remove_points=remove_keys)
            n_used = len(res['filters_used'])
            perc_info['PERCSEL'] = 'literature' if n_used >= 2 else 'default (no photometry)'
            perc_info['PERCFILT'] = ','.join(res['filters_used']) if n_used else 'none'
            perc_info['PERCRES'] = ','.join('%s:%.3f' % (p, res[p]['score']) for p in perc_options)
            score = res[perc_i]['score']
        elif perc == 'reference':
            perc_i = perc_ref if np.isfinite(perc_ref) else DEFAULT_PERC
            perc_info['PERCSEL'] = 'reference' if np.isfinite(perc_ref) else 'default (no reference)'
        else:
            perc_i = float(perc)
            perc_info['PERCSEL'] = 'fixed'

        spec, points = clean_spectrum_v2(raw_path, perc=perc_i, reference_clean=ref, channel_table=channel_table,
                                         error_method=error_method, clip_sigma=clip_sigma,
                                         drop_nonpositive=drop_nonpositive, perc_info=perc_info,
                                         remove_points=remove_keys)
        if len(spec) == 0:
            if verbose:
                print(f'{fname}: SKIPPED (empty spectrum after cleaning)')
            continue
        write_spectrum_v2(spec, out_path)
        m = spec.meta
        rows.append([fname.replace('Spherex_spec_', '').replace('.fits', ''), float(perc_i), perc_ref,
                     m['PERCSEL'], score, m['NRAW'], m['NINVALID'], m['NREMREF'], m['NREMOUT'], m['NMEAS'], m['NCHAN'],
                     float(np.median(spec['Npoints'])), int(np.max(spec['Npoints'])),
                     float(np.nanmedian(spec['Fluxerr'] / spec['Fluxerr_stat'])), ref is not None])
        if verbose:
            print('%s: aperture %.2f (%s) | %d measurements -> %d channels (median %.1f per channel)%s' % (
                fname, perc_i, m['PERCSEL'], m['NMEAS'], m['NCHAN'], np.median(spec['Npoints']),
                '' if ref else '  [no reference clean spectrum]'))
    summary = Table(rows=rows, names=['Name', 'PERC', 'PERC v1', 'PERC selection', 'Mean |Syn-Lit|', 'N raw',
                                      'N invalid', 'N removed (ref)', 'N removed (outliers)', 'N used', 'N channels', 'Median N/channel',
                                      'Max N/channel', 'Median err inflation', 'Reference used'])
    summary['Mean |Syn-Lit|'].format = '.3f'
    summary['Median err inflation'].format = '.2f'
    return summary
