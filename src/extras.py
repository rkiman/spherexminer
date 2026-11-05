from astropy.table import Table
import numpy as np
import astropy.units as u
from astropy.io import fits
import matplotlib.pyplot as plt
from scipy.interpolate import interp1d
from packaging.version import Version
import re
import os

c = 3*1e8*u.m/u.s
def jy_to_erg_s_cm2_AA(flux_jy,wavelength_nu):
    flux_jy = flux_jy*u.Jy
    wavelength_nu = wavelength_nu*u.um
    flux_erg_s_cm2_Hz = flux_jy.to(u.erg/u.s/u.cm**2/u.Hz)
    flux_erg_s_cm2_AA = (flux_erg_s_cm2_Hz*c/wavelength_nu**2).to(u.erg/u.s/u.cm**2/u.AA)
    return flux_erg_s_cm2_AA.value

def erg_s_cm2_Hz_to_erg_s_cm2_AA(flux_erg_s_cm2_Hz,wavelength_nu):
    flux_erg_s_cm2_Hz = flux_erg_s_cm2_Hz*u.erg/u.s/u.cm**2/u.Hz
    wavelength_nu = wavelength_nu*u.um
    flux_erg_s_cm2_AA = (flux_erg_s_cm2_Hz*c/wavelength_nu**2).to(u.erg/u.s/u.cm**2/u.AA)
    return flux_erg_s_cm2_AA.value

def Hz_to_nu(wavelegth_Hz):
    wavelegth_Hz = wavelegth_Hz*u.Hz
    model_wavelenght_um = ((c/wavelegth_Hz).to(u.um))
    return model_wavelenght_um.value

def turn_spt_to_num(spt_letter):
    spt_letter = np.atleast_1d(spt_letter)

    spt_num = []
    for x in spt_letter:
        if 'L' in x:
            spt_num.append(float(x.replace('L','')) + 10)
        if 'T' in x:
            spt_num.append(float(x.replace('T','')) + 20)

    spt_num = np.array(spt_num)

    if spt_num.size == 1:
        return spt_letter[0]
    
    return spt_num

def turn_num_to_spt(spt_num):
    spt_num = np.atleast_1d(spt_num)

    spt_letter = []
    for x in spt_num:
        if 0 <= x < 10:
            spt_letter.append("M{0:.1f}".format(np.round(x,1)))
        elif 9 < x < 20:
            spt_letter.append("L{0:.1f}".format(np.round(x,1)-10))
        elif 19 < x < 30:
            spt_letter.append("T{0:.1f}".format(np.round(x,1)-20))
        else:
            spt_letter.append(None)

    spt_letter = np.array(spt_letter)

    if spt_letter.size == 1:
        return spt_letter[0]

    return spt_letter

def get_closest_standard_spectrum(spt_object,path_data):
    standards = Table.read(path_data + 'tables/standards.csv')
    standards_spt_num = turn_spt_to_num(standards['SpT'])

    find_standard = np.argmin(abs(standards_spt_num-spt_object))
    name_standard = standards['Name'][find_standard]
    path_standard = path_data + 'spectra/Spherex_spec_'+name_standard.replace(' ','')+'.fits'

    return path_standard,find_standard

def _spt_to_num(spt):
    """'L3' -> 13, 'T4.5' -> 24.5 (same convention as spherexminer turn_spt_to_num)."""
    spt = str(spt).strip()
    for letter, offset in (('M', 0.), ('L', 10.), ('T', 20.), ('Y', 30.)):
        if spt.startswith(letter):
            return float(spt[1:]) + offset
    return np.nan


def get_closest_standard_spectrum_v2(spt_object, path_data, spectra_dir='data/spectra_v2/'):
    """Same as spherexminer get_closest_standard_spectrum, but returns the path to the v2 clean
    spectrum of the closest field standard (written by clean_sample_v2 in standards_new.ipynb)"""
    standards = Table.read(os.path.join(path_data, 'tables', 'standards.csv'))
    standards_spt_num = np.array([_spt_to_num(s) for s in standards['SpT']])
    find_standard = int(np.nanargmin(np.abs(standards_spt_num - spt_object)))
    name_standard = str(standards['Name'][find_standard])
    path_standard = os.path.join(spectra_dir, 'Spherex_spec_' + name_standard.replace(' ', '') + '.fits')
    return path_standard, find_standard


def get_not_good_young_standard():
    the_ones_that_dont_look_good = ['Gaia DR2 6273537856003401728','V* DH Tau B','2MASS J07410404-0359495','2MASS J16212830-2529558','Cl* Melotte   25   CFHT      19',
                                    '2MASS J02292794-0053282','2MASS J20135152-2806020','HD  44627B','ROXs 12B','2MASS J06272161-5308428','2MASS J11271382-3735076',
                                    '2MASS J00464841+0715177','2MASS J09175035+2944455','Cl* Melotte   22   PLIZ      35','2MASS J19350976-6200473','WISE J052857.68+090104.4',
                                    '* bet Pic b','Cl* Melotte   22    BRB      17','2MASS J04373705+2331080','CD-52   381B','PSO J078.9904+31.0171','2MASS J07123786-6155528',
                                    'HD 106906B','LSPM J1108+6830', '2MASS J05082480+2422518','2MASS J02495436-0558015','2MASS J04144158+2809583','2MASS J20544332-2745512',
                                    '2MASS J01531463-6744181','CD-35  2722B', 'G 196-3B','2MASS J12563961-2718455','2MASS J15382417-1953116','LEHPM 1481B','RX J1609.5-2105B',
                                    'Cl* Melotte   22    BRB      23','NAME 2MASS J02192210-3925225b','2MASSI J0103320+193536','2MASS J08132232-1522043','HD 116434C',
                                    'SIPS J1256-1257B','NAME AF Lep b','2MASS J00470038+6803543','2MASS J17410280-4642218','HD 203030B',
                                    'CWISE J050626.96+073842.4','WISE J124332.17+600126.6','*  51 Eri b',
                                    '* kap And B', '2MASS J15270961-2626574','2MASS J23225299-6151275','PSO J039.6352-21.7746','2MASS J07180871-6415310']

    return the_ones_that_dont_look_good

def get_closest_youngstandard_spectrum(spt_object, path_data):
    standards = Table.read(path_data + 'tables/youngstandards.csv')

    the_ones_that_dont_look_good = get_not_good_young_standard()
    mask_good = np.array([str(x) not in the_ones_that_dont_look_good for x in standards['Name']])
    standards = standards[mask_good]

    standards_spt_num = standards['SpT']

    closest_spt = np.argmin(abs(standards_spt_num-spt_object))

    find_standards = np.argwhere((standards_spt_num > standards_spt_num[closest_spt] - 1) * (standards_spt_num < standards_spt_num[closest_spt] + 1)).flatten()
    name_standard = standards['Name'][find_standards]
    path_standard = np.array([path_data + 'spectra/Spherex_spec_'+y+'.csv' for y in name_standard])

    return path_standard,find_standards

# SPHEREx detector (band) wavelength ranges in microns. Each of the 6 detectors
# is divided into 17 nominal channels -> 102 channels in total.
# Bands are contiguous so that no measurement falls in a gap.
SPHEREX_BAND_EDGES = [(0.75, 1.11), (1.11, 1.64), (1.64, 2.42),
                      (2.42, 3.82), (3.82, 4.42), (4.42, 5.00)]
SPHEREX_N_CHANNELS_PER_BAND = 17


def get_spherex_channel_edges(band_edges=SPHEREX_BAND_EDGES,
                              n_channels=SPHEREX_N_CHANNELS_PER_BAND):
    """
    Nominal edges of the 102 SPHEREx spectral channels.

    Within each detector the LVF gives constant resolving power, so the
    channels are evenly spaced in log(wavelength) between the band edges.

    Returns
    -------
    edges : list of np.ndarray
        One array of n_channels+1 edges (um) per band.
    """
    return [np.geomspace(wmin, wmax, n_channels + 1) for wmin, wmax in band_edges]


_HERE = os.path.dirname(os.path.abspath(__file__))
DEFAULT_CHANNEL_FILE = os.path.join(_HERE, 'spectral_channels_spx_cal-sch-v1-2026-106.fits')
_channel_cache = {}


def read_spherex_channels(channel_table=None):
    """Official SPHEREx channel definitions (IRSA spectral_channels_spx_cal-sch-*.fits)."""
    if channel_table is None:
        channel_table = DEFAULT_CHANNEL_FILE
    if isinstance(channel_table, str):
        if channel_table not in _channel_cache:
            _channel_cache[channel_table] = Table.read(channel_table, hdu='SPECTRAL_CHANNELS')
        return _channel_cache[channel_table]
    return channel_table


def channel_ids(channel_table=None):
    """Channel identifier 100 * DETECTOR + SUBCHAN (e.g. 305 = detector 3, channel 5)."""
    ch = read_spherex_channels(channel_table)
    return 100 * np.asarray(ch['DETECTOR'], dtype=int) + np.asarray(ch['SUBCHAN'], dtype=int)


def assign_spherex_channels(wave, channel_table=None):
    """Row of the channel table for each wavelength."""
    ch = read_spherex_channels(channel_table)
    wave = np.atleast_1d(np.asarray(wave, dtype=float))
    wl_c = np.asarray(ch['WAVELENGTH'], dtype=float)
    wl_min = np.asarray(ch['WL_MIN'], dtype=float)
    wl_max = np.asarray(ch['WL_MAX'], dtype=float)

    dist = np.abs(wave[:, None] - wl_c[None, :])
    inside = (wave[:, None] >= wl_min[None, :]) & (wave[:, None] <= wl_max[None, :])
    row = np.argmin(np.where(inside, dist, np.inf), axis=1)
    outside = ~inside.any(axis=1)
    row[outside] = np.argmin(dist[outside], axis=1)
    return row


_channel_map_cache = {}


def read_spherex_channel_map(channel_file=None):
    """Pixel -> sub-channel map (CHANNEL_MAP hdu, values 1-17; 0/18 = outside nominal channels)."""
    if channel_file is None:
        channel_file = DEFAULT_CHANNEL_FILE
    if channel_file not in _channel_map_cache:
        _channel_map_cache[channel_file] = fits.getdata(channel_file, 'CHANNEL_MAP')
    return _channel_map_cache[channel_file]


def get_spherex_channel(x, y, detector, wave=None, channel_file=None):
    """
    SPHEREx spectral channel of one measurement.

    The sub-channel (1-17) is read from the official CHANNEL_MAP at the
    (0-based) pixel position x, y of the source in the L2 image; the map is
    the same for the 6 detectors. If the pixel lies outside the 17 nominal
    channels (map value 0 or 18, detector edges) or x, y are not available,
    the closest channel of that detector in wavelength is used instead.

    Returns
    -------
    detector : int (1-6)
    subchan : int (1-17)
    channel : int, 100 * detector + subchan (e.g. 305), as in channel_ids()
    from_map : bool, True if the sub-channel comes from the pixel map
    """
    detector = int(detector)
    subchan, from_map = 0, False
    if x is not None and y is not None and np.isfinite(x) and np.isfinite(y):
        cmap = read_spherex_channel_map(channel_file)
        ix, iy = int(np.round(float(x))), int(np.round(float(y)))
        if 0 <= iy < cmap.shape[0] and 0 <= ix < cmap.shape[1]:
            subchan = int(cmap[iy, ix])
            from_map = 1 <= subchan <= SPHEREX_N_CHANNELS_PER_BAND
    if not from_map:
        if wave is None or not np.isfinite(wave):
            return detector, -1, -1, False
        ch = read_spherex_channels(channel_file)
        sel = np.asarray(ch['DETECTOR'], dtype=int) == detector
        wl_c = np.asarray(ch['WAVELENGTH'], dtype=float)[sel]
        sub = np.asarray(ch['SUBCHAN'], dtype=int)[sel]
        subchan = int(sub[np.argmin(np.abs(wl_c - float(wave)))])
    return detector, subchan, 100 * detector + subchan, from_map


def detector_from_bandpass(bandpassname):
    """Detector number from IRSA's energy_bandpassname, e.g. 'SPHEREx-D3' -> 3."""
    m = re.search(r'D(\d)', str(bandpassname))
    return int(m.group(1)) if m else -1


def combine_spherex_per_channel(wave, flux, fluxerr, error_method='birge',
                                wave_method='weighted', channel_table=None):
    """
    Combine all multi-epoch SPHEREx measurements that fall in the same channel
    into a single measurement.

    Parameters
    ----------
    wave, flux, fluxerr : array-like
        Individual measurements (any flux unit; errors in the same unit).
    error_method : 'birge' or 'quadrature'
        How the epoch-to-epoch scatter is propagated into the channel error.
        'birge'      : error of the weighted mean inflated by sqrt(chi2_red)
                       when the points scatter more than their errors predict.
        'quadrature' : error of the weighted mean added in quadrature to the
                       standard error of the mean of the scatter (std/sqrt(N)).
    wave_method : 'weighted' or 'center'
        Wavelength assigned to the channel: inverse-variance weighted mean of
        the measured wavelengths, or the nominal channel center.
    channel_table : None, str or Table
        Official SPHEREx channel definitions (spectral_channels_spx_cal-sch-*.fits).
        If None, an approximate definition is used (17 log-spaced channels per
        detector between the design band edges).

    Returns
    -------
    wave_c, flux_c, fluxerr_c, npoints, bandwidth_c, channel_id : np.ndarray
        channel_id is 100*DETECTOR + SUBCHAN (e.g. 305 = detector 3, channel 5).
    """
    wave = np.asarray(wave, dtype=float)
    flux = np.asarray(flux, dtype=float)
    fluxerr = np.asarray(fluxerr, dtype=float)

    good = np.isfinite(wave) & np.isfinite(flux) & np.isfinite(fluxerr) & (fluxerr > 0)
    wave, flux, fluxerr = wave[good], flux[good], fluxerr[good]

    if channel_table is not None:
        channel_table = read_spherex_channels(channel_table)
        row = assign_spherex_channels(wave, channel_table)
        chan_center = np.asarray(channel_table['WAVELENGTH'], dtype=float)
        chan_bw = np.asarray(channel_table['BANDWIDTH'], dtype=float)
        chan_id = 100 * np.asarray(channel_table['DETECTOR'], dtype=int) + \
                  np.asarray(channel_table['SUBCHAN'], dtype=int)
    else:
        all_edges = get_spherex_channel_edges()
        lo = np.concatenate([e[:-1] for e in all_edges])
        hi = np.concatenate([e[1:] for e in all_edges])
        chan_center = np.sqrt(lo * hi)
        chan_bw = hi - lo
        chan_id = np.concatenate([100 * (b + 1) + np.arange(1, len(e))
                                  for b, e in enumerate(all_edges)])
        # Points slightly outside the nominal 0.75-5.00 um range go to the
        # first/last channel (only used for binning)
        wave_bin = np.clip(wave, lo[0], np.nextafter(hi[-1], 0))
        row = np.searchsorted(hi, wave_bin, side='right')
        row = np.clip(row, 0, len(lo) - 1)

    wave_c, flux_c, fluxerr_c, npoints, bw_c, id_c = [], [], [], [], [], []
    for k in np.unique(row):
        m = row == k
        n = m.sum()
        w = 1.0 / fluxerr[m]**2
        f_mean = np.sum(w * flux[m]) / np.sum(w)
        e_mean = 1.0 / np.sqrt(np.sum(w))
        if n > 1:
            chi2_red = np.sum(w * (flux[m] - f_mean)**2) / (n - 1)
            if error_method == 'birge':
                e_mean = e_mean * np.sqrt(max(1.0, chi2_red))
            elif error_method == 'quadrature':
                e_mean = np.sqrt(e_mean**2 + np.std(flux[m], ddof=1)**2 / n)
            else:
                raise ValueError("error_method must be 'birge' or 'quadrature'")
        if wave_method == 'weighted':
            wl = np.sum(w * wave[m]) / np.sum(w)
        else:
            wl = chan_center[k]
        wave_c.append(wl)
        flux_c.append(f_mean)
        fluxerr_c.append(e_mean)
        npoints.append(n)
        bw_c.append(chan_bw[k])
        id_c.append(chan_id[k])

    order = np.argsort(wave_c)
    return (np.array(wave_c)[order], np.array(flux_c)[order],
            np.array(fluxerr_c)[order], np.array(npoints)[order],
            np.array(bw_c)[order], np.array(id_c)[order])


def clean_spherex_spec(spec, snr=1,nbins=4,change_band=False,bin=False,plot=False,
                       combine_channels=False,error_method='birge',wave_method='weighted',
                       channel_table=None):
    """
    combine_channels : if True, all multi-epoch measurements falling in the same
        nominal SPHEREx channel (102 channels, 17 per detector) are combined into
        one point with combine_spherex_per_channel (see error_method / wave_method).
        The output then has the columns Wave [um], Flux [uJy], Flux err [uJy],
        Npoints, Bandwidth [um] and Channel (100*detector + subchannel).
    channel_table : path to (or Table of) the official SPHEREx channel definitions
        (spectral_channels_spx_cal-sch-*.fits). If given, it is used to assign
        channels and bandwidths; if combine_channels=False the points are kept
        and Bandwidth [um] and Channel columns are added. If None, an approximate
        17-channels-per-detector log-spaced definition is used.
        This only returns a new table; nothing is written to disk.
    """
    columns = spec.colnames
    # Sorting by wavelength 
    spec.sort(columns[0])

    # Changing names of columns so all agree
    spec.rename_column(columns[0], 'Wave [um]')
    spec.rename_column(columns[1], 'Flux [uJy]')
    spec.rename_column(columns[2], 'Flux err [uJy]')
    columns = spec.colnames

    # Clean points in the change of band 1 to 2
    if change_band:
        mask = (spec[columns[0]] > 1.06) * (spec[columns[0]] < 1.13) * (spec[columns[2]]/spec[columns[1]]>0.5)
        if plot:
            plt.errorbar(spec[columns[0]],spec[columns[1]],yerr=spec[columns[2]],fmt='.')
            plt.errorbar(spec[columns[0]][mask],spec[columns[1]][mask],yerr=spec[columns[2]][mask],fmt='.')
            plt.axvline(x=1.06)
            plt.axvline(x=1.13)
            plt.xlim(0.8,1.5)
            plt.show()
        spec = spec[~mask]

    # Binning
    if bin:
        delta_avg = nbins
        wave = spec[columns[0]]
        flux = spec[columns[1]]
        fluxerr = spec[columns[2]]

        '''
        for j in np.arange(0,len(wave)-delta_avg,delta_avg):
            if (wave[j] > 3.809):
                #idx_to_change = np.argwhere(mask)
                #idx_to_change = idx_to_change.flatten()
                #median_wave = wave[mask][0] + (wave[mask][-1]-wave[mask][0])/2
                #idx_median = np.argmin(abs(wave-median_wave))

                w = 1.0 / fluxerr[j:j+delta_avg]**2
                f_comb = np.sum(flux[j:j+delta_avg] * w) / np.sum(w)
                e_comb = 1.0 / np.sqrt(np.sum(w))
                w_comb = np.average(wave[j:j+delta_avg], weights=w)

                # reduced chi-square of points around the weighted mean
                chi2 = np.sum(w * (flux[j:j+delta_avg] - f_comb)**2)
                dof = len(wave[j:j+delta_avg]) - 1
                chi2_red = chi2 / dof if dof > 0 else 1.0

                # inflate error if points scatter more than their errors predict
                e_comb = max(e_comb,np.sqrt(chi2_red))

                flux[j+1:j+delta_avg] = np.nan
                fluxerr[j+1:j+delta_avg] = np.nan

                wave[j] = w_comb
                flux[j] = f_comb
                fluxerr[j] = e_comb
        '''


        for j in np.arange(0,len(wave)-delta_avg,delta_avg):
            if (wave[j] > 3.809):
                flux_j = 0
                n_avg = 0
                for d in range(delta_avg):
                    if str(flux[j+d])!='nan':
                        flux_j += np.random.normal(flux[j+d],fluxerr[j+d],10000)
                        n_avg+=1

                idx_to_change = int(delta_avg/2)
                flux[j+idx_to_change] = np.nanmedian(flux_j/n_avg)
                fluxerr[j+idx_to_change] = np.nanstd(flux_j/n_avg)
                for d in range(0,delta_avg):
                    if d!=idx_to_change:
                        flux[j+d] = np.nan
                        fluxerr[j+d] = np.nan
        spec[columns[0]] = wave
        spec[columns[1]] = flux
        spec[columns[2]] = fluxerr

        # Signal to noise cut
        #spec = spec[spec['Flux [uJy]']/spec['Flux err [uJy]'] > snr]

    # Combine multi-epoch measurements per SPHEREx channel
    if combine_channels:
        wave_c, flux_c, fluxerr_c, npoints, bw_c, id_c = combine_spherex_per_channel(
            spec['Wave [um]'], spec['Flux [uJy]'], spec['Flux err [uJy]'],
            error_method=error_method, wave_method=wave_method,
            channel_table=channel_table)
        spec = Table([wave_c, flux_c, fluxerr_c, npoints, bw_c, id_c],
                     names=['Wave [um]', 'Flux [uJy]', 'Flux err [uJy]', 'Npoints',
                            'Bandwidth [um]', 'Channel'])
    elif channel_table is not None:
        # Keep all points, but add the bandwidth and channel of each one
        channel_table = read_spherex_channels(channel_table)
        row = assign_spherex_channels(spec['Wave [um]'], channel_table)
        spec['Bandwidth [um]'] = np.asarray(channel_table['BANDWIDTH'], dtype=float)[row]
        spec['Channel'] = 100 * np.asarray(channel_table['DETECTOR'], dtype=int)[row] + \
                          np.asarray(channel_table['SUBCHAN'], dtype=int)[row]

    return spec

def gravity_to_num(x):
    if x =='gamma':
        return 2
    elif x == 'delta':
        return 3
    elif x == 'beta':
        return 1
    elif x == 'INT-G':
        return 1
    elif x == 'low' or x == 'lowg':
        return 0.5
    elif x == 'null':
        return np.nan
    elif x == '(VL-G)' or x == 'VL-G':
        return 2
    elif x == 'FLD-G':
        return 0
    else:
        return x

def gravity_to_latex(x):
    if x =='gamma':
        return r'$\gamma$'
    elif x == 'delta':
        return r'$\delta$'
    elif x == 'beta':
        return r'$\beta$'
    elif x == 'low' or x == 'lowg':
        return 'LG'
    elif x == 'null':
        return '-'
    elif x == '(VL-G)':
        return 'VL-G'
    else:
        return x




def spectral_agreement(wave1, flux1, err1, wave2, flux2, err2,
                        wave_range=None, n_params=0):
    """
    Quantify agreement between two spectra (different wavelength sampling).
    Returns reduced chi2, RMS residual, and median % residual.

    wave_range : optional (wmin, wmax) tuple to restrict the comparison,
                 e.g. to avoid a noisy tail.
    """
    wmin = max(wave1.min(), wave2.min())
    wmax = min(wave1.max(), wave2.max())
    if wave_range is not None:
        wmin = max(wmin, wave_range[0])
        wmax = min(wmax, wave_range[1])

    mask = (wave2 >= wmin) & (wave2 <= wmax)
    wave2_ov, flux2_ov, err2_ov = wave2[mask], flux2[mask], err2[mask]

    flux1_interp = interp1d(wave1, flux1, kind='linear',
                             bounds_error=False, fill_value=np.nan)(wave2_ov)
    err1_interp = interp1d(wave1, err1, kind='linear',
                            bounds_error=False, fill_value=np.nan)(wave2_ov)

    good = np.isfinite(flux1_interp) & np.isfinite(err1_interp) & \
           np.isfinite(flux2_ov) & np.isfinite(err2_ov)

    resid = flux1_interp[good] - flux2_ov[good]
    sigma_comb = np.sqrt(err1_interp[good]**2 + err2_ov[good]**2)

    chi2 = np.sum((resid / sigma_comb)**2)
    dof = good.sum() - n_params
    chi2_red = chi2 / dof

    rms = np.sqrt(np.mean(resid**2))
    pct_resid = 100 * np.abs(resid) / flux2_ov[good]
    median_pct = np.median(pct_resid)

    return {
        'chi2_red': chi2_red,
        'rms': rms,
        'median_pct_residual': median_pct,
        'n_points': good.sum()
    }


def update_psf_header(old_hdul):
    """
    Fix an old PSF FITS file header by rewriting only the per-plane header metadata
    so that plane k corresponds to x-fast ordering:
        k0 = iy * bins_x + ix

    The cube data are left untouched.

    Parameters
    ----------
    old_hdul : fits.HDUList
        Old SPHEREx Spectral Image HDUL

    Return
    ----------
    new_hdul : fits.HDUList
        New SPHEREx Spectral Image HDUL with updated PSF zone data in header and updated version number
    """

    VERSION_FIXED = Version("6.5.6")
    PSF_FIX_TAG = "psffix1"

    def psf_fix_applied(hdul) -> bool:
        """
        Return True if the PSF fix has been applied.

        Rules:
        - If the VERSION header is missing in the primary HDU, the fix is not applied.
        - If VERSION >= VERSION_FIXED, the fix is included in the software release.
        - Otherwise the local version tag (+...) must contain PSF_FIX_TAG.
        """
        header = hdul[0].header

        if "VERSION" not in header:
            return False

        v = Version(header["VERSION"])

        if v >= VERSION_FIXED:
            return True

        return v.local is not None and PSF_FIX_TAG in v.local

    if psf_fix_applied(old_hdul):
        return old_hdul

    ## Define some auxiliary functions -------
    def parse_ixiy_from_comment(comment):
        _zone_pat = re.compile(r"\((\d+)\s*,\s*(\d+)\)")
        m = _zone_pat.search(str(comment))
        if not m:
            raise ValueError(f"Could not parse zone indices from comment: {comment!r}")
        return int(m.group(1)), int(m.group(2))

    def infer_grid_shape_from_header_comments(hdr, nzone):
        max_ix = -1
        max_iy = -1

        for k1 in range(1, nzone + 1):
            key = f"XCTR_{k1}"
            if key not in hdr:
                raise KeyError(f"Missing required key: {key}")
            ix, iy = parse_ixiy_from_comment(hdr.comments[key])
            max_ix = max(max_ix, ix)
            max_iy = max(max_iy, iy)

        bins_x = max_ix + 1
        bins_y = max_iy + 1

        if bins_x * bins_y != nzone:
            raise ValueError(
                f"Inconsistent grid inferred from comments: "
                f"bins_x={bins_x}, bins_y={bins_y}, nzone={nzone}"
            )

        return bins_x, bins_y

    def collect_axis_values_by_zone(hdr, nzone):
        """
        Read the old header and collect unique x/y centers and widths by zone index
        labels found in the comments.

        This uses the old header only to recover the per-axis values for each ix, iy.
        It does NOT use the old plane ordering as truth.
        """
        x_center_by_ix = {}
        y_center_by_iy = {}
        x_width_by_ix = {}
        y_width_by_iy = {}

        for k1 in range(1, nzone + 1):
            ix, iy = parse_ixiy_from_comment(hdr.comments[f"XCTR_{k1}"])

            xck = f"XCTR_{k1}"
            yck = f"YCTR_{k1}"
            xwk = f"XWID_{k1}"
            ywk = f"YWID_{k1}"

            if xck in hdr:
                val = hdr[xck]
                if ix in x_center_by_ix and not np.isclose(x_center_by_ix[ix], val):
                    raise ValueError(
                        f"Inconsistent XCTR for ix={ix}: "
                        f"{x_center_by_ix[ix]} vs {val}"
                    )
                x_center_by_ix[ix] = val

            if yck in hdr:
                val = hdr[yck]
                if iy in y_center_by_iy and not np.isclose(y_center_by_iy[iy], val):
                    raise ValueError(
                        f"Inconsistent YCTR for iy={iy}: "
                        f"{y_center_by_iy[iy]} vs {val}"
                    )
                y_center_by_iy[iy] = val

            if xwk in hdr:
                val = hdr[xwk]
                if ix in x_width_by_ix and not np.isclose(x_width_by_ix[ix], val):
                    raise ValueError(
                        f"Inconsistent XWID for ix={ix}: "
                        f"{x_width_by_ix[ix]} vs {val}"
                    )
                x_width_by_ix[ix] = val

            if ywk in hdr:
                val = hdr[ywk]
                if iy in y_width_by_iy and not np.isclose(y_width_by_iy[iy], val):
                    raise ValueError(
                        f"Inconsistent YWID for iy={iy}: "
                        f"{y_width_by_iy[iy]} vs {val}"
                    )
                y_width_by_iy[iy] = val

        return x_center_by_ix, y_center_by_iy, x_width_by_ix, y_width_by_iy
    ## End defining some auxiliary functions --------

    ## Get Header
    extname = "PSF"
    hdu = old_hdul[extname]
    cube = np.asarray(hdu.data)
    hdr_in = hdu.header.copy()

    if cube.ndim != 3:
        raise ValueError(f"Expected 3D PSF cube, got shape {cube.shape}")

    nzone = cube.shape[0]
    bins_x, bins_y = infer_grid_shape_from_header_comments(hdr_in, nzone)

    print(f"Detected bins_x={bins_x}, bins_y={bins_y}, nzone={nzone}")

    x_center_by_ix, y_center_by_iy, x_width_by_ix, y_width_by_iy = collect_axis_values_by_zone(
        hdr_in, nzone
    )

    # Validate that all needed axis values were recovered
    missing = []
    for ix in range(bins_x):
        if ix not in x_center_by_ix:
            missing.append(f"x_center[{ix}]")
        if ix not in x_width_by_ix:
            missing.append(f"x_width[{ix}]")
    for iy in range(bins_y):
        if iy not in y_center_by_iy:
            missing.append(f"y_center[{iy}]")
        if iy not in y_width_by_iy:
            missing.append(f"y_width[{iy}]")

    if missing:
        raise ValueError(f"Missing axis metadata recovered from old header: {missing}")

    hdr_out = hdr_in.copy()

    # Rewrite only the per-plane metadata so plane k matches x-fast ordering.
    # plane k0 should correspond to:
    #   ix = k0 % bins_x
    #   iy = k0 // bins_x
    for k0 in range(nzone):
        ix = k0 % bins_x
        iy = k0 // bins_x
        k1 = k0 + 1

        hdr_out[f"XCTR_{k1}"] = (
            x_center_by_ix[ix],
            f"Center of x zone ({ix}, {iy})"
        )
        hdr_out[f"YCTR_{k1}"] = (
            y_center_by_iy[iy],
            f"Center of y zone ({ix}, {iy})"
        )
        hdr_out[f"XWID_{k1}"] = (
            x_width_by_ix[ix],
            f"Width of x zone ({ix}, {iy})"
        )
        hdr_out[f"YWID_{k1}"] = (
            y_width_by_iy[iy],
            f"Width of y zone ({ix}, {iy})"
        )

    # Optional but useful provenance note
    hdr_out["HISTORY"] = "Rewrote PSF per-plane zone metadata to x-fast ordering."
    hdr_out["HISTORY"] = "Cube plane data left unchanged."



    new_hdu = fits.ImageHDU(data=cube, header=hdr_out, name=hdu.name)

    ext_index = old_hdul.index_of(extname)
    new_hdul = fits.HDUList()
    for i, old in enumerate(old_hdul):
        if i == ext_index:
            new_hdul.append(new_hdu)
        else:
            new_hdul.append(old.copy())

    ## TO DO: UPDATE VERSION
    new_hdul['PRIMARY'].header["VERSION"] = new_hdul['PRIMARY'].header["VERSION"] + "+psffix1" # SET NEW VERSION HERE

    return(new_hdul)

def calculate_abs_mag(mag,mag_err,parallax,parallax_err):
    abs_mag = np.ones(len(mag))*np.nan
    mag_abs_err = np.ones(len(mag))*np.nan
    
    for i in range(len(abs_mag)):
        dist_mag = np.random.normal(mag[i],mag_err[i],10000)
        dist_parallax = np.random.normal(parallax[i],parallax_err[i],10000)

        dist_abs_mag = dist_mag - 5*(np.log10(1e3/dist_parallax)-1)
        abs_mag[i] = np.nanmedian(dist_abs_mag)
        mag_abs_err[i] = np.nanstd(dist_abs_mag)
    #dist = 1e3 / parallax
    #abs_mag = mag - 5 * (np.log10(dist) - 1)
    
    # partial derivative w.r.t. parallax
    #d_abs_d_plx = 5 / (parallax * np.log(10))
    
    #mag_abs_err = np.sqrt(mag_err**2 + (d_abs_d_plx * parallax_err)**2)
    return abs_mag,mag_abs_err

def calculate_color(mag1,mag1_err,mag2,mag2_err):
    color = mag1-mag2
    color_err = np.sqrt(mag1_err**2+mag2_err**2)
    return color,color_err