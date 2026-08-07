
# Interactive manual point removal tool.
#
# Requires an interactive matplotlib backend - in Jupyter/VSCode run
# one of these in its own cell before calling the function:
#   %matplotlib widget      (ipympl - works inline in the notebook)
#   %matplotlib qt          (pops out a separate window)
# The inline backend ("%matplotlib inline") does NOT support clicking.
#
# Also assumes `jy_to_erg_s_cm2_AA` and `renormalize_spectrum` are
# already defined/imported in the notebook (as in the rest of this
# project) - they're used to pick the best flux percentile for each
# object, the same way as in the standalone best_perc-selection script.

import os
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.ticker import AutoMinorLocator
from astropy.table import Table
from astropy.stats import sigma_clipped_stats
from .extras import *
from .flux_renormalize import *

# Keys we use ('n', 'u', 's', 'q') collide with matplotlib's default
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

DEFAULT_FIL_PHOT_SYNT = ['J_2MASS', 'H_2MASS', 'K_2MASS', 'W1', 'I1','I2','Lp_MKO','Mp_MKO']
DEFAULT_FILTER_PATH = '/Users/rociokiman/Hunter College Dropbox/Rocio Kiman/papers/_Published/Coconuts2/Data/Filters/'

# every column that follows the '<prefix>_<perc>' naming convention -
# when a percentile is chosen, all of the OTHER percentiles' versions
# of these columns get dropped, and the chosen one gets renamed to
# just '<prefix>' (suffix removed)
PERC_COLUMN_PREFIXES = ('Flux', 'Fluxerr', 'r_aperture', 'r_in', 'r_out')


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
    Loop through `spectra` one at a time, click to mark bad points, and
    save a cleaned version of each into `output_dir`.

    Unlike an earlier version of this tool, the flux percentile to save
    is NOT passed in - it's chosen automatically. For each object, a
    percentile is first picked from the raw spectrum (same method as
    the standalone best_perc script) just to decide what to display for
    point-picking. After you finish marking bad points but before
    saving, the percentile choice is redone using the cleaned spectrum
    (points you removed are excluded from the comparison), since
    removing bad points can change which percentile's photometry best
    matches the literature values. That final choice is what gets saved.

    The saved file keeps only 'Wave' plus the winning percentile's
    columns from each '<prefix>_<perc>' family (Flux, Fluxerr,
    r_aperture, r_in, r_out - see PERC_COLUMN_PREFIXES), with every
    other percentile's version of those columns dropped, and the
    survivors renamed to drop the _<perc> suffix (e.g. 'Flux_0.95' ->
    'Flux'). Rows are dropped if their flux is NaN/non-positive for the
    chosen percentile, or if you marked them for removal. The FITS
    header's PERC keyword (which lists all the percentile options in
    the raw spectrum) is overwritten with just the one selected.

    If a cleaned version of a spectrum already exists in `output_dir`
    (same filename), it's loaded and used to pre-mark the points that
    were removed last time, so you're resuming rather than starting
    over. Points are colored to show this at a glance: green for a
    spectrum that already has a saved clean version (points still
    kept, i.e. not previously removed), orange for a spectrum that has
    never been cleaned before. Previously/currently removed points
    always show as a red x, either way.

    Controls, while a spectrum's window is open:
      left-click near a point   toggle that point for removal (shows a red x)
      'n'                       save the cleaned spectrum and move to the next one
      'u'                       undo - go back to the state when the window opened
      's'                       skip this spectrum (nothing saved) and move on
      'q'                       stop the whole loop
      'a'                       clear all removals (keep every point) - restart selection from scratch

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
    'syn_phot_err' (each an array of shape (n_saved, len(fil_phot_synt)),
    in the same filter order as fil_phot_synt) and 'best_perc' (a 1D
    array of length n_saved, the percentile actually used for each
    spectrum). One row per spectrum you saved this run (skipped ones
    aren't included). Feed the photometry arrays into
    plot_photometry_comparison(fil_phot_synt, ...) yourself, or plot
    them however you like.
    """
    os.makedirs(output_dir, exist_ok=True)
    if fil_phot_synt is None:
        fil_phot_synt = DEFAULT_FIL_PHOT_SYNT
    if filter_path is None:
        filter_path = DEFAULT_FILTER_PATH

    # some keymap.* rcParams have been removed across matplotlib versions
    # (e.g. keymap.all_axes) - only touch the ones that actually exist
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

    # One window for the whole run, reused across every spectrum (only
    # its contents are cleared/redrawn each iteration). Creating and
    # destroying a new Tk/Qt window on every iteration is what caused
    # the tool to hang after the last spectrum; a single long-lived
    # window avoids that, and plt.show(block=False) + plt.pause() is
    # what actually waits for you in Jupyter/IPython - a plain
    # plt.show() is non-blocking there regardless of its `block` value,
    # since IPython manages the GUI event loop itself (that's what
    # caused every window to pop up at once instead of waiting).
    fig, ax = plt.subplots(figsize=(10, 5))
    plt.show(block=False)

    try:
        for i, spectrum_path in enumerate(spectra):
            spectrum = Table.read(spectrum_path, format='fits')
            wave_full = np.array(spectrum['Wave'])

            i_known_data = idx[i]
            phot_measurement = np.array([known_data[fil][i_known_data] for fil in fil_phot_synt])
            phot_err_measurement = np.array([known_data[fil + '_err'][i_known_data] for fil in fil_phot_synt])

            no_exclusions = np.zeros(len(spectrum), dtype=bool)

            # pick a percentile to DISPLAY, based on the raw (uncleaned) spectrum
            perc_display, _, _ = _pick_best_perc(spectrum, wave_full, perc_options, phot_measurement,
                                                  phot_err_measurement, no_exclusions, fil_phot_synt, filter_path)
            col_flux = 'Flux_' + str(perc_display)
            col_fluxerr = 'Fluxerr_' + str(perc_display)

            flux_full = np.array(spectrum[col_flux])
            fluxerr_full = np.array(spectrum[col_fluxerr])

            valid = np.array([True if x > 0 and str(x) != 'nan' else False for x in flux_full])
            valid_idx = np.where(valid)[0]
            wave = wave_full[valid_idx]
            flux = flux_full[valid_idx]
            fluxerr = fluxerr_full[valid_idx]

            # check for a previously-saved clean version of this spectrum
            out_path = os.path.join(output_dir, os.path.basename(spectrum_path))
            already_cleaned = os.path.exists(out_path)

            removed = np.zeros(len(valid_idx), dtype=bool)  # aligned with valid_idx / wave / flux
            if already_cleaned:
                clean_table = Table.read(out_path)
                # a row was removed previously if its wavelength no longer
                # appears in the saved clean spectrum
                removed_full_mask = ~np.isin(wave_full, np.array(clean_table['Wave']))
                removed = removed_full_mask[valid_idx]

            point_color = 'tab:green' if already_cleaned else 'tab:orange'

            ax.clear()
            ax.errorbar(wave, flux, yerr=fluxerr, fmt='none', ecolor='lightgray', zorder=0)
            ax.plot(wave, flux, 'o', color=point_color, markersize=4, zorder=1)
            marks_bad, = ax.plot([], [], 'x', color='red', markersize=9, markeredgewidth=2, zorder=5)
            ax.set_xlabel(r'Wavelength ($\mu$m)')
            ax.set_ylabel(r'Flux ($\mu$Jy)')
            status = (f"clean version exists ({removed.sum()} pt(s) previously removed) - green = kept"
                      if already_cleaned else "new spectrum, never cleaned - orange = untouched")
            ax.set_title(f"{os.path.basename(spectrum_path)}   [{i+1}/{len(spectra)}]   perc={perc_display}   {status}\n"
                        "click=toggle point, n=save+next, u=undo, a=keep all (restart), s=skip, q=quit")
            print(f"[{i+1}/{len(spectra)}] {os.path.basename(spectrum_path)}: displaying perc={perc_display}. "
                  + (f"Existing clean spectrum found, {removed.sum()} point(s) previously removed."
                     if already_cleaned else "No existing clean spectrum - starting fresh."))

            baseline_removed = removed.copy()  # what 'u' undoes back to
            if already_cleaned:
                marks_bad.set_data(wave_full[valid_idx[removed]], flux_full[valid_idx[removed]])

            state = {'done': False, 'action': None}

            def redraw():
                bad = valid_idx[removed]
                marks_bad.set_data(wave_full[bad], flux_full[bad])
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
                print('Stopped early - nothing after this point was processed.')
                break

            if state['action'] == 'skip':
                print(f'Skipped (not saved): {spectrum_path}')
                continue

            # --- bad point selection is done: now pick the final percentile,
            # using the cleaned spectrum (manually-removed rows excluded) ---
            manually_removed_mask = np.zeros(len(spectrum), dtype=bool)
            manually_removed_mask[valid_idx[removed]] = True

            perc_best, phot_synt_mag, ephot_synt_mag = _pick_best_perc(
                spectrum, wave_full, perc_options, phot_measurement, phot_err_measurement,
                manually_removed_mask, fil_phot_synt, filter_path)

            if perc_best != perc_display:
                print(f"  best percentile changed after cleaning: {perc_display} -> {perc_best}")
            if phot_synt_mag is not None:
                for fil, syn, lit in zip(fil_phot_synt, phot_synt_mag, phot_measurement):
                    print(f"    {fil}: synthetic={syn:.3f}  literature={lit:.3f}")

            col_flux_best = 'Flux_' + str(perc_best)
            col_fluxerr_best = 'Fluxerr_' + str(perc_best)
            flux_best_full = np.array(spectrum[col_flux_best])
            valid_best = np.array([True if x > 0 and str(x) != 'nan' else False for x in flux_best_full])

            # rows: valid for the chosen percentile, and not marked for removal
            keep_mask = valid_best & (~manually_removed_mask)

            cleaned = spectrum[keep_mask]

            # columns: for every '<prefix>_<perc>' family (Flux, Fluxerr,
            # r_aperture, r_in, r_out, ...), drop every percentile except the
            # chosen one, and rename the survivor so the _<perc> suffix is gone
            best_col_names = {f'{prefix}_{perc_best}': prefix for prefix in PERC_COLUMN_PREFIXES}
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
            cleaned.meta['PERC'] = perc_best

            cleaned.write(out_path, format='fits', overwrite=True)
            n_removed_clicks = int(manually_removed_mask.sum())
            n_nan_dropped = int((~valid_best).sum())
            print(f'Saved {out_path}  (perc={perc_best}, {n_removed_clicks} point(s) removed by hand, '
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
            phot_results['best_perc'].append(perc_best)
    finally:
        if plt.fignum_exists(fig.number):
            plt.close(fig)

    return phot_results


# Example usage:
# %matplotlib widget
# phot = clean_spectra_interactively(spectra, idx, known_data, output_dir='spectra/clean')
# data_phot, syn_phot = phot['data_phot'], phot['syn_phot']
# best_perc = phot['best_perc']
# # ... or, to reproduce the comparison figure yourself:
# # fig, axs = plot_photometry_comparison(fil_phot_synt, phot['data_phot'], phot['data_phot_err'],
# #                                        phot['syn_phot'], phot['syn_phot_err'])
