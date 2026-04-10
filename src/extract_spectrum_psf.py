from astropy.time import Time
from astropy.coordinates import SkyCoord
from astropy import units as u
import numpy as np
from photutils.aperture import CircularAperture, aperture_photometry, CircularAnnulus
from photutils.aperture import ApertureStats
from astropy.stats import SigmaClip
import re
from astropy.table import Table
import matplotlib.pyplot as plt
from scipy.interpolate import interp1d, RectBivariateSpline
from scipy.optimize import least_squares
import matplotlib.cm as cm
from astropy.wcs import WCS
from astropy.io import fits
from matplotlib.gridspec import GridSpec
from .flux_renormalize import *
import astropy
import requests
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry
import os
from astroquery.gaia import Gaia
try:
    import emcee
except ImportError:
    emcee = None   # MCMC unavailable; _fit_psf_mcmc will raise a clear error


def extract_spectrum(name, results, results_spec, ra_deg, dec_deg, pmra, pmdec, epoch, perc,
                    test_external_ring, known_data, i_known_data, path_figures, path_data,
                    increase_ring=1, method='aperture', psf_half_size=10, psf_centroid_slack=1.5,
                    mcmc_nwalkers=32, mcmc_nsteps=500, mcmc_nburn=200, mcmc_use=False):
    """
    Extract a SPHEREx spectrum slice by slice.

    Parameters
    ----------
    name, results, results_spec, ra_deg, dec_deg, pmra, pmdec, epoch :
        Same as before.
    perc : float
        Encircled-energy fraction used to set the aperture radius (aperture method only).
    test_external_ring : bool
        Run ring-stability diagnostic plots (aperture method only).
    known_data, i_known_data, path_figures, path_data :
        Same as before.
    increase_ring : float
        Scale factor applied to the background annulus radii (aperture method only).
    method : str
        'aperture'  - classic aperture photometry with annulus background subtraction.
        'psf'       - simultaneous PSF + tilted-plane background fitting.
                      When using the PSF method, a Gaia DR3 cone search is run
                      automatically on the first slice to find contaminating
                      neighbours within 2x the SPHEREx pixel scale.  No extra
                      download is needed -- the WCS is read from the FITS file
                      that is already being downloaded for that slice.  The
                      neighbour table is reused for all subsequent slices;
                      only their pixel positions are re-computed per slice via
                      proper-motion propagation from Gaia epoch J2016.0.
    psf_half_size : int
        Half-width (pixels) of the fitting box used by the PSF method around the
        target, before expansion to include neighbours (default 10).
    psf_centroid_slack : float
        Maximum centroid shift allowed for the *target* during the PSF fit
        (pixels, default 1.5).  Neighbour centroids are always fixed at their
        proper-motion-corrected catalogue positions.
    mcmc_use : bool
        If True and method='psf', run an MCMC sampler (emcee) after the
        least_squares initialisation to obtain realistic flux uncertainties.
        The least_squares solution is used as the starting point so burn-in
        is short.  A noise-inflation parameter log_f is sampled alongside the
        fluxes to account for underestimated per-pixel errors, which is the
        recommended fix when reduced chi2 >> 1 with random (unstructured)
        residuals.  Default False.
    mcmc_nwalkers : int
        Number of emcee walkers (default 32).  Must be even and > 2 * n_params.
    mcmc_nsteps : int
        Number of MCMC steps per walker after burn-in (default 500).
    mcmc_nburn : int
        Number of burn-in steps discarded before collecting samples (default 200).

    Returns
    -------
    results_spec : astropy Table
        For method='aperture', new columns are keyed with str(perc):
            'Wave [um] <perc>', 'Flux [uJy] <perc>', 'Flux err [uJy] <perc>',
            'r aperture <perc>', 'r in <perc>', 'r out <perc>'
        For method='psf', new columns are keyed with 'psf':
            'Wave [um] psf', 'Flux [uJy] psf', 'Flux err [uJy] psf',
            'x centroid psf', 'y centroid psf', 'background psf', 'fit chi2 psf'
            and, if Gaia neighbours were found, one column pair per neighbour:
            'Flux [uJy] neighbour_<k>'  and  'Flux err [uJy] neighbour_<k>'
        If mcmc_use=True, additional columns are added:
            'Flux err lo [uJy] psf'  (16th percentile uncertainty, asymmetric lower)
            'Flux err hi [uJy] psf'  (84th percentile uncertainty, asymmetric upper)
            'log_f psf'              (median fitted noise-inflation log_f)
            'Flux err lo [uJy] neighbour_<k>'
            'Flux err hi [uJy] neighbour_<k>'
    """
    if method not in ('aperture', 'psf'):
        raise ValueError(f"method must be 'aperture' or 'psf', got '{method}'")

    # Gaia DR3 proper motions and coordinates are at epoch J2016.0.
    # Used when propagating neighbour positions to each observation epoch.
    GAIA_EPOCH = 2016.0

    wave     = []
    flux     = []
    flux_err = []

    # aperture-only output arrays
    r_ap_array  = []
    r_in_array  = []
    r_out_array = []

    # psf-only output arrays
    x_cen_arr = []
    y_cen_arr = []
    bkg_arr   = []
    chi2_arr  = []

    # neighbours is discovered on slice i=0 and reused for all slices.
    # n_neighbours and the flux accumulator lists are set after the query.
    neighbours             = None   # populated on first slice (PSF method only)
    n_neighbours           = 0
    neighbour_flux_arr     = []
    neighbour_flux_err_arr = []

    # MCMC-only accumulators (populated only when mcmc_use=True)
    flux_err_lo_arr    = []   # 16th-percentile lower uncertainty on target flux
    flux_err_hi_arr    = []   # 84th-percentile upper uncertainty on target flux
    log_f_arr          = []   # fitted noise-inflation parameter per slice
    nb_flux_err_lo_arr = []   # list-of-lists, one per neighbour
    nb_flux_err_hi_arr = []

    for i in range(len(results)):

        spectral_image_url = results['access_url'][i]
        print(f'{i} out of {len(results)} [{method}]: {spectral_image_url}')

        # ------------------------------------------------------------------
        # Download image
        # ------------------------------------------------------------------
        output_path = path_data + "temporary.fits"
        downloaded = 0
        while downloaded == 0:
            try:
                with requests.get(spectral_image_url, stream=True) as r:
                    r.raise_for_status()
                    with open(output_path, "wb") as f:
                        for chunk in r.iter_content(chunk_size=2 * 2 * 1024 * 1024):
                            if chunk:
                                f.write(chunk)
                downloaded = 1
            except:
                downloaded = 0

        hdulist = fits.open(output_path)
        os.remove(output_path)

        # ------------------------------------------------------------------
        # Set up diagnostic figure
        # ------------------------------------------------------------------
        fig = plt.figure(figsize=(15, 8))
        gs  = GridSpec(2, 4, figure=fig)
        ax0 = fig.add_subplot(gs[0, 0])
        ax1 = fig.add_subplot(gs[0, 1])
        ax2 = fig.add_subplot(gs[0, 2])
        ax3 = fig.add_subplot(gs[0, 3])
        ax4 = fig.add_subplot(gs[1, :2])
        ax5 = fig.add_subplot(gs[1, 2:])

        # ------------------------------------------------------------------
        # Extract relevant information
        # ------------------------------------------------------------------
        spectral_image_header = hdulist[1].header
        psf_header   = hdulist[5].header
        psfcube      = hdulist[5].data
        flags_header = hdulist[2].header
        flags_data   = hdulist[2].data
        image_data     = hdulist[1].data.astype(float)
        image_data_err = np.sqrt(np.abs(hdulist[3].data)).astype(float)

        # ------------------------------------------------------------------
        # Proper-motion corrected pixel position of the target
        # ------------------------------------------------------------------
        new_ra, new_dec = move_forward_position(ra_deg, dec_deg, pmra, pmdec, epoch,
                                                results['t_min'][i])
        wcs = WCS(spectral_image_header)
        x_rel, y_rel = wcs.world_to_pixel(SkyCoord(ra=new_ra, dec=new_dec, unit="deg"))
        x_rel, y_rel = float(x_rel), float(y_rel)

        # ------------------------------------------------------------------
        # On the first slice: query Gaia for neighbours using the WCS that
        # is already in memory -- no extra download required.
        # The neighbours table is reused for all subsequent slices; only
        # their pixel positions are recomputed via proper-motion propagation.
        # ------------------------------------------------------------------
        if i == 0 and method == 'psf':
            neighbours = _query_gaia_neighbours(ra_deg, dec_deg, wcs)
            n_neighbours           = len(neighbours) if neighbours is not None else 0
            neighbour_flux_arr     = [[] for _ in range(n_neighbours)]
            neighbour_flux_err_arr = [[] for _ in range(n_neighbours)]
            nb_flux_err_lo_arr     = [[] for _ in range(n_neighbours)]
            nb_flux_err_hi_arr     = [[] for _ in range(n_neighbours)]

        # ------------------------------------------------------------------
        # Proper-motion corrected pixel positions of neighbours (PSF only).
        # Gaia DR3 coordinates are at epoch J2016.0, so we propagate from
        # GAIA_EPOCH regardless of the target's reference epoch.
        # ------------------------------------------------------------------
        neighbour_pixels = []    # list of (x_n, y_n) in image pixels
        if neighbours is not None and method == 'psf':
            for nb in neighbours:
                nb_ra, nb_dec = move_forward_position(
                    nb['ra'], nb['dec'], nb['pmra'], nb['pmdec'],
                    GAIA_EPOCH, results['t_min'][i])
                x_nb, y_nb = wcs.world_to_pixel(
                    SkyCoord(ra=nb_ra, dec=nb_dec, unit="deg"))
                neighbour_pixels.append((float(x_nb), float(y_nb)))

        # ------------------------------------------------------------------
        # Nearest PSF (needed by both methods)
        # ------------------------------------------------------------------
        psf = get_closest_psf(psf_header, psfcube, x_rel, y_rel)
        oversample = float(psf_header['OVERSAMP'])

        # ------------------------------------------------------------------
        # Wavelength at the target pixel position
        # ------------------------------------------------------------------
        spectral_wcs = WCS(header=hdulist["IMAGE"].header, fobj=hdulist, key="W")
        spectral_wcs.sip = None
        wl, bw = spectral_wcs.pixel_to_world(x_rel, y_rel)
        wave.append(wl.value)

        # ==================================================================
        # APERTURE PHOTOMETRY BRANCH
        # ==================================================================
        if method == 'aperture':

            r_aperture, r_in, r_out = study_different_apertures(
                x_rel, y_rel, image_data, image_data_err,
                flags_header, flags_data,
                psf, psf.shape[0] / 2 - 1, psf.shape[0] / 2 - 1,
                oversample, perc, increase_ring, ax4, ax5)
            r_ap_array.append(r_aperture)
            r_in_array.append(r_in)
            r_out_array.append(r_out)

            aperture_corr = get_aperture_corr(
                psf, psf.shape[0] / 2 - 1, psf.shape[0] / 2 - 1,
                r_aperture * oversample, ax0)

            if test_external_ring:
                study_different_rings_image(
                    x_rel, y_rel, image_data, image_data_err,
                    flags_header, flags_data, r_aperture, r_in, r_out,
                    path_figures + name + '/' + str(np.round(wl.value, 2)) + '_phot_test.png')

            aper_flux, aper_flux_err, total_bkg, total_bkg_err = calculate_flux_and_bkg(
                x_rel, y_rel, image_data, image_data_err,
                flags_header, flags_data, r_aperture, r_in, r_out,
                ax1, ax2, ax3)
            while str(total_bkg) == 'nan':
                r_in, r_out = 2 * r_in, 2 * r_out
                aper_flux, aper_flux_err, total_bkg, total_bkg_err = calculate_flux_and_bkg(
                    x_rel, y_rel, image_data, image_data_err,
                    flags_header, flags_data, r_aperture, r_in, r_out,
                    ax1, ax2, ax3)

            apersum_bkgsub     = aper_flux - total_bkg
            apersum_bkgsub_err = np.sqrt(aper_flux_err ** 2 + total_bkg_err ** 2)

            omega_arcsec2     = hdulist[1].header['HIERARCH OMEGA_MEDIAN']
            omega_err_arcsec2 = hdulist[1].header['HIERARCH OMEGA_STDDEV']
            omega_st     = omega_arcsec2     / (206265 ** 2)
            omega_st_err = omega_err_arcsec2 / (206265 ** 2)

            flux_ujy = apersum_bkgsub * omega_st * 1e12 * aperture_corr
            flux_ujy_err = np.sqrt(
                (apersum_bkgsub_err * omega_st     * 1e12 * aperture_corr) ** 2 +
                (apersum_bkgsub     * omega_st_err * 1e12 * aperture_corr) ** 2)

            flux.append(float(flux_ujy))
            flux_err.append(float(flux_ujy_err))
            plot_suffix = str(perc)

        # ==================================================================
        # PSF FITTING BRANCH
        # ==================================================================
        elif method == 'psf':

            psf_interp = _build_psf_interpolator(psf, oversample)

            flags_fit = ['SUR_ERROR', 'NONFUNC', 'MISSING_DATA',
                         'HOT', 'COLD', 'NONLINEAR', 'PERSIST']
            mask_fit = define_mask(flags_fit, flags_header, flags_data)

            try:
                (A_fit, A_err, x_fit, y_fit, bkg_const, red_chi2,
                 nb_amplitudes, nb_amp_errs) = _fit_psf(
                    image_data, image_data_err, mask_fit,
                    x_rel, y_rel, psf_interp,
                    half_size=psf_half_size,
                    centroid_slack=psf_centroid_slack,
                    neighbour_pixels=neighbour_pixels)
            except RuntimeError as exc:
                print(f'  WARNING: PSF fit failed for slice {i}: {exc}')
                A_fit, A_err  = np.nan, np.nan
                x_fit, y_fit  = x_rel, y_rel
                bkg_const     = np.nan
                red_chi2      = np.nan
                nb_amplitudes = [np.nan] * n_neighbours
                nb_amp_errs   = [np.nan] * n_neighbours

            x_cen_arr.append(x_fit)
            y_cen_arr.append(y_fit)
            bkg_arr.append(bkg_const)
            chi2_arr.append(red_chi2)

            # MJy/sr -> uJy  (PSF normalised to unit integral -> A = total flux)
            omega_arcsec2     = hdulist[1].header['HIERARCH OMEGA_MEDIAN']
            omega_err_arcsec2 = hdulist[1].header['HIERARCH OMEGA_STDDEV']
            omega_st     = omega_arcsec2     / (206265 ** 2)
            omega_st_err = omega_err_arcsec2 / (206265 ** 2)

            # ------------------------------------------------------------------
            # MCMC branch: replaces least_squares uncertainties with posterior
            # percentiles.  The least_squares solution is used as the starting
            # point so walkers are already near the posterior peak.
            # ------------------------------------------------------------------
            if mcmc_use and np.isfinite(A_fit):
                try:
                    (A_med, A_lo, A_hi,
                     nb_A_meds, nb_A_los, nb_A_his,
                     log_f_med) = _fit_psf_mcmc(
                        image_data, image_data_err, mask_fit,
                        x_fit, y_fit, psf_interp,
                        A_fit, nb_amplitudes, bkg_const,
                        half_size=psf_half_size,
                        centroid_slack=psf_centroid_slack,
                        neighbour_pixels=neighbour_pixels,
                        nwalkers=mcmc_nwalkers,
                        nsteps=mcmc_nsteps,
                        nburn=mcmc_nburn)

                    # Use MCMC median as the flux point estimate; asymmetric errs
                    flux_ujy    = A_med  * omega_st * 1e12
                    # Symmetric component from omega uncertainty
                    omega_unc   = A_med * omega_st_err * 1e12
                    flux_ujy_err_lo = np.sqrt((A_lo  * omega_st * 1e12) ** 2 + omega_unc ** 2)
                    flux_ujy_err_hi = np.sqrt((A_hi  * omega_st * 1e12) ** 2 + omega_unc ** 2)
                    # Symmetric summary error = average of the two wings
                    flux_ujy_err = 0.5 * (flux_ujy_err_lo + flux_ujy_err_hi)

                    flux_err_lo_arr.append(float(flux_ujy_err_lo))
                    flux_err_hi_arr.append(float(flux_ujy_err_hi))
                    log_f_arr.append(float(log_f_med))

                    for k, (nb_med, nb_lo, nb_hi) in enumerate(
                            zip(nb_A_meds, nb_A_los, nb_A_his)):
                        nb_flux     = nb_med * omega_st * 1e12
                        nb_omega_unc = nb_med * omega_st_err * 1e12
                        nb_err_lo = np.sqrt((nb_lo * omega_st * 1e12) ** 2 + nb_omega_unc ** 2)
                        nb_err_hi = np.sqrt((nb_hi * omega_st * 1e12) ** 2 + nb_omega_unc ** 2)
                        neighbour_flux_arr[k].append(float(nb_flux))
                        neighbour_flux_err_arr[k].append(0.5 * (float(nb_err_lo) + float(nb_err_hi)))
                        nb_flux_err_lo_arr[k].append(float(nb_err_lo))
                        nb_flux_err_hi_arr[k].append(float(nb_err_hi))

                except Exception as exc:
                    print(f'  WARNING: MCMC failed for slice {i}: {exc}. '
                          f'Falling back to least_squares uncertainties.')
                    mcmc_use_this_slice = False
                else:
                    mcmc_use_this_slice = True
            else:
                mcmc_use_this_slice = False

            if not mcmc_use_this_slice:
                # least_squares uncertainties (symmetric, formal only)
                flux_ujy = A_fit * omega_st * 1e12
                flux_ujy_err = np.sqrt(
                    (A_err * omega_st     * 1e12) ** 2 +
                    (A_fit * omega_st_err * 1e12) ** 2)
                # Fill MCMC columns with NaN so arrays stay the same length
                if mcmc_use:
                    flux_err_lo_arr.append(np.nan)
                    flux_err_hi_arr.append(np.nan)
                    log_f_arr.append(np.nan)
                    for k in range(n_neighbours):
                        nb_flux_err_lo_arr[k].append(np.nan)
                        nb_flux_err_hi_arr[k].append(np.nan)

                for k, (nb_A, nb_A_err) in enumerate(zip(nb_amplitudes, nb_amp_errs)):
                    nb_flux_ujy     = nb_A     * omega_st * 1e12
                    nb_flux_ujy_err = np.sqrt(
                        (nb_A_err * omega_st     * 1e12) ** 2 +
                        (nb_A     * omega_st_err * 1e12) ** 2)
                    neighbour_flux_arr[k].append(float(nb_flux_ujy))
                    neighbour_flux_err_arr[k].append(float(nb_flux_ujy_err))

            flux.append(float(flux_ujy))
            flux_err.append(float(flux_ujy_err))

            # Diagnostic plots
            _plot_psf_fit_diag(image_data, image_data_err, mask_fit,
                               x_fit, y_fit, x_rel, y_rel,
                               psf_interp, psf, psf_half_size, psf_centroid_slack,
                               neighbour_pixels, nb_amplitudes,
                               ax0, ax1, ax2, ax3)

            ax4.plot(list(range(len(chi2_arr))), chi2_arr, 'ko-')
            ax4.axhline(1.0, color='r', ls='--', label='chi2=1')
            ax4.set_xlabel('Slice index')
            ax4.set_ylabel('Reduced chi2')
            ax4.set_title('PSF fit quality')
            ax4.legend()

            ax5.errorbar(wave, np.array(flux), yerr=np.array(flux_err),
                         fmt='.', color='tab:blue', label='Target')
            nb_colors = cm.get_cmap('Reds')(np.linspace(0.4, 0.9, max(n_neighbours, 1)))
            for k in range(n_neighbours):
                if len(neighbour_flux_arr[k]) > 0:
                    ax5.errorbar(wave[:len(neighbour_flux_arr[k])],
                                 np.array(neighbour_flux_arr[k]),
                                 yerr=np.array(neighbour_flux_err_arr[k]),
                                 fmt='.', color=nb_colors[k], label=f'Neighbour {k}')
            ax5.set_xscale('log')
            ax5.set_xlabel('Wavelength [um]')
            ax5.set_ylabel('Flux [uJy]')
            ax5.set_title('Spectrum so far')
            ax5.legend(fontsize=7)

            plot_suffix = 'psf'

        # ------------------------------------------------------------------
        # Save per-slice diagnostic figure
        # ------------------------------------------------------------------
        plt.tight_layout()
        plt.savefig(
            path_figures + name + '/' + str(np.round(wl.value, 2)) +
            '_phot_' + plot_suffix + '.png',
            dpi=100, bbox_inches='tight')
        plt.close(fig)

        astropy.utils.data.clear_download_cache(hashorurl=None, pkgname='astropy')

    # ----------------------------------------------------------------------
    # Store per-slice results
    # ----------------------------------------------------------------------
    key = str(perc) if method == 'aperture' else 'psf'

    results_spec['Wave [um] '      + key] = np.array(wave)
    results_spec['Flux [uJy] '     + key] = np.array(flux)
    results_spec['Flux err [uJy] ' + key] = np.array(flux_err)

    if method == 'aperture':
        results_spec['r aperture ' + key] = np.array(r_ap_array)
        results_spec['r in '       + key] = np.array(r_in_array)
        results_spec['r out '      + key] = np.array(r_out_array)
    else:
        results_spec['x centroid psf'] = np.array(x_cen_arr)
        results_spec['y centroid psf'] = np.array(y_cen_arr)
        results_spec['background psf'] = np.array(bkg_arr)
        results_spec['fit chi2 psf']   = np.array(chi2_arr)
        for k in range(n_neighbours):
            results_spec[f'Flux [uJy] neighbour_{k}']     = np.array(neighbour_flux_arr[k])
            results_spec[f'Flux err [uJy] neighbour_{k}'] = np.array(neighbour_flux_err_arr[k])
        if mcmc_use:
            results_spec['Flux err lo [uJy] psf'] = np.array(flux_err_lo_arr)
            results_spec['Flux err hi [uJy] psf'] = np.array(flux_err_hi_arr)
            results_spec['log_f psf']              = np.array(log_f_arr)
            for k in range(n_neighbours):
                results_spec[f'Flux err lo [uJy] neighbour_{k}'] = np.array(nb_flux_err_lo_arr[k])
                results_spec[f'Flux err hi [uJy] neighbour_{k}'] = np.array(nb_flux_err_hi_arr[k])

    # ----------------------------------------------------------------------
    # Final comparison plot
    # ----------------------------------------------------------------------
    try:
        known_data_spec = Table.read('../Data/' + known_data['File'][i_known_data], format='ascii')
        wave_spex      = known_data_spec['col1']
        flux_spex      = known_data_spec['col2']
        flux_spex_err  = known_data_spec['col3']
        fil_phot_synt  = ['J', 'H', 'K']
        phot_measurement     = [known_data['J'][i_known_data],
                                 known_data['H'][i_known_data],
                                 known_data['K'][i_known_data]]
        phot_err_measurement = [known_data['J_err'][i_known_data],
                                 known_data['H_err'][i_known_data],
                                 known_data['H_err'][i_known_data]]
        scaling, _, _, _, _, _, _, _, _ = renormalize_spectrum(
            fil_phot_synt, phot_measurement, phot_err_measurement,
            wave_spex, flux_spex, flux_spex_err,
            filter_path='/Users/rociokiman/Hunter College Dropbox/Rocio Kiman/papers/_Published/Coconuts2/Data/Filters/')
        flux_spex_norm     = scaling * flux_spex
        flux_spex_norm     = flux_spex_norm * (3.33564095e+04 * (wave_spex * 1e4) ** 2) * 1e6
        flux_spex_err_norm = scaling * flux_spex_err * (3.33564095e+04 * (wave_spex * 1e4) ** 2) * 1e6
        plt.errorbar(wave_spex, flux_spex_norm, yerr=flux_spex_err_norm, fmt='', color='lightgray')
        plt.errorbar(wave_spex, flux_spex_norm, fmt='-', label='Spex')
    except:
        pass

    label = 'SPHEREx' if method == 'aperture' else 'SPHEREx (PSF)'
    plt.errorbar(wave, np.array(flux), yerr=np.array(flux_err), fmt='.', label=label)
    plt.legend(title=name)
    plt.xscale('log')
    plt.xlabel('Wavelength [um]')
    plt.ylabel('Flux [uJy]')
    plt.savefig(path_figures + name + '/_comparison_spec_' + key + '.png', dpi=100)
    plt.show()

    return results_spec


# ==========================================================================
# PSF fitting helpers
# ==========================================================================

def _query_gaia_neighbours(ra_deg, dec_deg, wcs):
    """
    Query Gaia DR3 for sources within 2x the SPHEREx pixel scale of the target.

    The pixel scale is derived from *wcs*, which must be the already-parsed
    WCS of the current image slice.  No FITS download is performed here.

    The target itself (any Gaia source within 1 arcsec of ra_deg, dec_deg) is
    excluded.  Sources with missing proper motions are kept with pmra=pmdec=0
    so their positions can still be propagated to each observation epoch.

    Parameters
    ----------
    ra_deg, dec_deg : float
        Target coordinates (degrees).
    wcs : astropy.wcs.WCS
        WCS of the current image slice, already in memory.

    Returns
    -------
    neighbours : astropy Table with columns
        'ra', 'dec', 'pmra', 'pmdec', 'source_id', 'phot_g_mean_mag'
        or None if no neighbours are found.
    """
    # Pixel scale in arcsec from the WCS pixel-scale matrix
    pixel_scale_deg    = np.sqrt(np.abs(np.linalg.det(wcs.pixel_scale_matrix)))
    pixel_scale_arcsec = pixel_scale_deg * 3600.0
    search_radius_arcsec = 2.0 * pixel_scale_arcsec
    print(f'  Gaia neighbour search: pixel scale = {pixel_scale_arcsec:.2f} arcsec, '
          f'search radius = {search_radius_arcsec:.2f} arcsec')

    # Cone search in Gaia DR3
    coord  = SkyCoord(ra=ra_deg * u.deg, dec=dec_deg * u.deg)
    radius = search_radius_arcsec * u.arcsec

    try:
        Gaia.MAIN_GAIA_TABLE = 'gaiadr3.gaia_source'
        Gaia.ROW_LIMIT = -1
        job = Gaia.cone_search_async(coord, radius=radius,
                                     columns=['source_id', 'ra', 'dec',
                                              'pmra', 'pmdec', 'phot_g_mean_mag'])
        gaia_table = job.get_results()
    except Exception as exc:
        print(f'  WARNING: Gaia query failed: {exc}. Fitting target alone.')
        return None

    if len(gaia_table) == 0:
        print('  Gaia: no sources found in search radius. Fitting target alone.')
        return None

    # Exclude the target (within 1 arcsec)
    gaia_coords  = SkyCoord(ra=gaia_table['ra'], dec=gaia_table['dec'], unit='deg')
    target_coord = SkyCoord(ra=ra_deg * u.deg, dec=dec_deg * u.deg)
    is_target    = target_coord.separation(gaia_coords).arcsec < 2.0
    neighbours_raw = gaia_table[~is_target]

    if len(neighbours_raw) == 0:
        print('  Gaia: no neighbours after excluding the target. Fitting target alone.')
        return None

    # Fill missing proper motions with 0
    pmra_vals  = np.where(np.isfinite(neighbours_raw['pmra'].filled(np.nan)),
                          neighbours_raw['pmra'].filled(0.0), 0.0)
    pmdec_vals = np.where(np.isfinite(neighbours_raw['pmdec'].filled(np.nan)),
                          neighbours_raw['pmdec'].filled(0.0), 0.0)

    neighbours = Table({
        'ra':              np.array(neighbours_raw['ra']),
        'dec':             np.array(neighbours_raw['dec']),
        'pmra':            pmra_vals,
        'pmdec':           pmdec_vals,
        'source_id':       np.array(neighbours_raw['source_id']),
        'phot_g_mean_mag': np.array(neighbours_raw['phot_g_mean_mag']),
    })

    print(f'  Gaia: found {len(neighbours)} neighbour(s) within '
          f'{search_radius_arcsec:.2f} arcsec:')
    for row in neighbours:
        print(f'    source_id={row["source_id"]}  '
              f'G={row["phot_g_mean_mag"]:.2f}  '
              f'ra={row["ra"]:.6f}  dec={row["dec"]:.6f}  '
              f'pmra={row["pmra"]:.2f}  pmdec={row["pmdec"]:.2f}')

    return neighbours


def _build_psf_interpolator(psf_oversampled, oversample):
    """
    Return a callable psf_model(dx_image, dy_image) that evaluates the PSF
    at arbitrary offsets in image pixels from its centre.

    The PSF is normalised so its integral equals 1 in image-pixel units,
    meaning the fitted amplitude A equals the total integrated source flux
    in the same units as the image (MJy/sr).
    """
    n = psf_oversampled.shape[0]
    cx, cy = (n - 1) / 2.0, (n - 1) / 2.0
    x_os = (np.arange(n) - cx) / oversample
    y_os = (np.arange(n) - cy) / oversample

    pixel_size_image = 1.0 / oversample
    norm     = np.sum(psf_oversampled) * pixel_size_image ** 2
    psf_norm = psf_oversampled / norm

    interp = RectBivariateSpline(y_os, x_os, psf_norm, kx=3, ky=3)

    def psf_model(dx_image, dy_image):
        return interp(dy_image, dx_image, grid=False)

    return psf_model


def _fit_psf(data, data_err, mask, x0, y0, psf_interp,
             half_size=10, centroid_slack=1.5, neighbour_pixels=None):
    """
    Simultaneously fit the target + any catalogue neighbours + a tilted-plane
    background to a pixel cutout.

    Model
    -----
    image(x, y) = A_target * PSF(x - x0 - dx, y - y0 - dy)
                + sum_k  A_k * PSF(x - x_k, y - y_k)
                + B*x + C*y + D

    where:
      A_target, dx, dy  - target amplitude and allowed centroid shift
      A_k               - amplitude of k-th neighbour (>= 0, centroid fixed)
      B, C, D           - tilted-plane background

    The fitting box is automatically expanded beyond psf_half_size so that
    every neighbour centroid is covered, giving the fit enough pixels to
    constrain all sources simultaneously.

    Parameters
    ----------
    data, data_err    : 2-D ndarrays (full image)
    mask              : 2-D bool array (True = good pixel)
    x0, y0            : initial target centroid estimate (image pixels, 0-indexed)
    psf_interp        : callable from _build_psf_interpolator
    half_size         : base half-width of fitting box in pixels
    centroid_slack    : max centroid shift for the target (pixels)
    neighbour_pixels  : list of (x_n, y_n) image-pixel positions, or None

    Returns
    -------
    A_fit, A_err      : target amplitude and 1-sigma formal uncertainty
    x_fit, y_fit      : fitted target centroid
    bkg_const         : fitted background constant D
    red_chi2          : reduced chi-squared
    nb_amplitudes     : list of fitted neighbour amplitudes (empty if no neighbours)
    nb_amp_errs       : list of neighbour amplitude 1-sigma uncertainties
    """
    if neighbour_pixels is None:
        neighbour_pixels = []
    n_nb = len(neighbour_pixels)

    ny, nx = data.shape
    ix0, iy0 = int(round(x0)), int(round(y0))

    # Expand the fitting box to enclose every source with half_size padding
    all_x = [x0] + [p[0] for p in neighbour_pixels]
    all_y = [y0] + [p[1] for p in neighbour_pixels]
    pad = half_size
    xlo = max(int(np.floor(min(all_x))) - pad, 0)
    xhi = min(int(np.ceil( max(all_x))) + pad + 1, nx)
    ylo = max(int(np.floor(min(all_y))) - pad, 0)
    yhi = min(int(np.ceil( max(all_y))) + pad + 1, ny)

    xi = np.arange(xlo, xhi)
    yi = np.arange(ylo, yhi)
    XX, YY = np.meshgrid(xi, yi)

    data_cut = data[ylo:yhi, xlo:xhi]
    err_cut  = data_err[ylo:yhi, xlo:xhi]
    mask_cut = mask[ylo:yhi, xlo:xhi]

    valid   = mask_cut & np.isfinite(data_cut) & np.isfinite(err_cut) & (err_cut > 0)
    min_pix = 6 + n_nb
    if valid.sum() < min_pix:
        raise RuntimeError(
            f"Not enough valid pixels for PSF fit ({valid.sum()} < {min_pix}).")

    Xv = XX[valid].astype(float)
    Yv = YY[valid].astype(float)
    Dv = data_cut[valid].astype(float)
    Ev = err_cut[valid].astype(float)

    # ------------------------------------------------------------------
    # Parameter vector layout:
    #   p[0]               = A_target
    #   p[1]               = dx  (target centroid shift x)
    #   p[2]               = dy  (target centroid shift y)
    #   p[3 .. 3+n_nb-1]   = A_neighbour_0 .. A_neighbour_{n_nb-1}
    #   p[3+n_nb]          = B  (background x-gradient)
    #   p[4+n_nb]          = C  (background y-gradient)
    #   p[5+n_nb]          = D  (background constant)
    # ------------------------------------------------------------------
    local_med = float(np.nanmedian(Dv))
    A_init    = float(np.nanmax(Dv) - local_med)
    if A_init <= 0:
        A_init = float(np.nanstd(Dv))

    p0 = ([A_init, 0.0, 0.0]
          + [A_init * 0.1] * n_nb
          + [0.0, 0.0, local_med])

    bounds_lo = ([0.0, -centroid_slack, -centroid_slack]
                 + [0.0] * n_nb
                 + [-np.inf, -np.inf, -np.inf])
    bounds_hi = ([np.inf, centroid_slack, centroid_slack]
                 + [np.inf] * n_nb
                 + [np.inf, np.inf, np.inf])

    # Pre-compute fixed neighbour PSF footprints (centroids never move)
    nb_psf_vals = [psf_interp(Xv - p[0], Yv - p[1]) for p in neighbour_pixels]

    def residuals(p):
        A, dx, dy = p[0], p[1], p[2]
        nb_amps   = p[3: 3 + n_nb]
        B, C, D   = p[3 + n_nb], p[4 + n_nb], p[5 + n_nb]
        model = A * psf_interp(Xv - (x0 + dx), Yv - (y0 + dy)) + B * Xv + C * Yv + D
        for k, nb_A in enumerate(nb_amps):
            model = model + nb_A * nb_psf_vals[k]
        return (model - Dv) / Ev

    result = least_squares(residuals, p0, bounds=(bounds_lo, bounds_hi),
                           method='trf', ftol=1e-10, xtol=1e-10, gtol=1e-10,
                           max_nfev=5000 + 1000 * n_nb)

    p_fit     = result.x
    A_fit     = float(p_fit[0])
    dx_fit    = float(p_fit[1])
    dy_fit    = float(p_fit[2])
    nb_A_fits = list(p_fit[3: 3 + n_nb])
    D_fit     = float(p_fit[5 + n_nb])

    # Formal uncertainties from SVD of the Jacobian
    try:
        _, s, Vt  = np.linalg.svd(result.jac, full_matrices=False)
        threshold = np.finfo(float).eps * max(result.jac.shape) * s[0]
        s_inv     = np.where(s > threshold, 1.0 / s, 0.0)
        cov       = (Vt.T * s_inv ** 2) @ Vt
        A_err     = float(np.sqrt(cov[0, 0]))
        nb_A_errs = [float(np.sqrt(cov[3 + k, 3 + k])) for k in range(n_nb)]
    except np.linalg.LinAlgError:
        A_err     = np.nan
        nb_A_errs = [np.nan] * n_nb

    ndof     = int(valid.sum()) - len(p0)
    red_chi2 = float(np.sum(result.fun ** 2) / max(ndof, 1))

    return (A_fit, A_err,
            float(x0 + dx_fit), float(y0 + dy_fit),
            D_fit, red_chi2,
            nb_A_fits, nb_A_errs)


def _fit_psf_mcmc(data, data_err, mask, x0, y0, psf_interp,
                  A_init, nb_A_inits, bkg_init,
                  half_size=10, centroid_slack=1.5, neighbour_pixels=None,
                  nwalkers=32, nsteps=500, nburn=200):
    """
    MCMC PSF fit using emcee, starting from the least_squares solution.

    Model
    -----
    Same multi-source + tilted-plane model as _fit_psf, with one extra free
    parameter log_f (log of an additive noise floor in the same units as the
    image).  The log-likelihood is:

        log L = -0.5 * sum[ (model - data)^2 / sigma_tot^2 + log(2*pi*sigma_tot^2) ]

    where  sigma_tot^2 = sigma_pixel^2 + exp(2*log_f)

    This formulation (Foreman-Mackey et al. 2013, emcee paper, Section 3.3)
    lets the data themselves determine how much extra noise is needed to bring
    chi2 to ~1, rather than assuming the per-pixel error array is perfect.
    When the residuals are random but chi2 >> 1 (noise underestimation), log_f
    settles at a positive value and the returned flux uncertainties are
    automatically inflated to match the true scatter.

    Parameter vector layout (length = 3 + n_nb + 3 + 1)
    -----------------------------------------------------------
      p[0]               A_target          (>= 0)
      p[1]               dx                (|dx| <= centroid_slack)
      p[2]               dy                (|dy| <= centroid_slack)
      p[3..3+n_nb-1]     A_neighbour_k     (>= 0)
      p[3+n_nb]          B  (bkg x-slope)
      p[4+n_nb]          C  (bkg y-slope)
      p[5+n_nb]          D  (bkg constant)
      p[6+n_nb]          log_f             (unconstrained)

    Parameters
    ----------
    data, data_err : 2-D arrays (full image)
    mask           : 2-D bool array (True = good pixel)
    x0, y0         : target centroid from least_squares (image pixels)
    psf_interp     : callable from _build_psf_interpolator
    A_init         : target amplitude from least_squares
    nb_A_inits     : list of neighbour amplitudes from least_squares
    bkg_init       : background constant D from least_squares
    half_size      : fitting box half-width (pixels)
    centroid_slack : max centroid displacement for the target
    neighbour_pixels : list of (x_n, y_n), or None
    nwalkers       : number of emcee walkers (must be even, > 2*n_params)
    nsteps         : production steps per walker (after burn-in)
    nburn          : burn-in steps discarded

    Returns
    -------
    A_med, A_lo, A_hi         : target amplitude median, 16th- and 84th-pct half-widths
    nb_A_meds, nb_A_los, nb_A_his : same for each neighbour (lists)
    log_f_med                 : median of the noise-inflation posterior
    """
    try:
        import emcee
    except ImportError:
        raise ImportError("emcee is required for MCMC fitting: pip install emcee")

    if neighbour_pixels is None:
        neighbour_pixels = []
    n_nb = len(neighbour_pixels)

    # ------------------------------------------------------------------
    # Build pixel cutout (same box logic as _fit_psf)
    # ------------------------------------------------------------------
    ny, nx = data.shape
    all_x = [x0] + [p[0] for p in neighbour_pixels]
    all_y = [y0] + [p[1] for p in neighbour_pixels]
    pad = half_size
    xlo = max(int(np.floor(min(all_x))) - pad, 0)
    xhi = min(int(np.ceil( max(all_x))) + pad + 1, nx)
    ylo = max(int(np.floor(min(all_y))) - pad, 0)
    yhi = min(int(np.ceil( max(all_y))) + pad + 1, ny)

    xi = np.arange(xlo, xhi);  yi = np.arange(ylo, yhi)
    XX, YY = np.meshgrid(xi, yi)

    data_cut = data[ylo:yhi, xlo:xhi]
    err_cut  = data_err[ylo:yhi, xlo:xhi]
    mask_cut = mask[ylo:yhi, xlo:xhi]

    valid = mask_cut & np.isfinite(data_cut) & np.isfinite(err_cut) & (err_cut > 0)
    Xv = XX[valid].astype(float)
    Yv = YY[valid].astype(float)
    Dv = data_cut[valid].astype(float)
    Sv = err_cut[valid].astype(float)          # per-pixel sigma
    S2v = Sv ** 2                               # per-pixel variance

    # Pre-compute fixed neighbour PSF footprints
    nb_psf_vals = [psf_interp(Xv - p[0], Yv - p[1]) for p in neighbour_pixels]

    # ------------------------------------------------------------------
    # Log-probability = log-likelihood + log-prior
    # ------------------------------------------------------------------
    def _log_prior(p):
        A       = p[0]
        dx, dy  = p[1], p[2]
        nb_amps = p[3: 3 + n_nb]
        # log_f is the last parameter
        log_f   = p[6 + n_nb]

        # Hard bounds
        if A < 0:
            return -np.inf
        if abs(dx) > centroid_slack or abs(dy) > centroid_slack:
            return -np.inf
        if any(a < 0 for a in nb_amps):
            return -np.inf

        # Weakly informative prior on log_f:
        # Gaussian centred on log(median pixel sigma), width 3 dex.
        # This keeps log_f from wandering to absurd values while still
        # being essentially uninformative over any reasonable noise range.
        log_f_prior_mu = np.log(float(np.median(Sv)))
        return -0.5 * ((log_f - log_f_prior_mu) / 3.0) ** 2

    def _log_likelihood(p):
        A, dx, dy = p[0], p[1], p[2]
        nb_amps   = p[3: 3 + n_nb]
        B, C, D   = p[3 + n_nb], p[4 + n_nb], p[5 + n_nb]
        log_f     = p[6 + n_nb]

        model = A * psf_interp(Xv - (x0 + dx), Yv - (y0 + dy)) + B * Xv + C * Yv + D
        for k, nb_A in enumerate(nb_amps):
            model = model + nb_A * nb_psf_vals[k]

        sigma2_tot = S2v + np.exp(2.0 * log_f)
        return -0.5 * np.sum((Dv - model) ** 2 / sigma2_tot + np.log(sigma2_tot))

    def _log_prob(p):
        lp = _log_prior(p)
        if not np.isfinite(lp):
            return -np.inf
        ll = _log_likelihood(p)
        return lp + ll if np.isfinite(ll) else -np.inf

    # ------------------------------------------------------------------
    # Starting positions: least_squares solution + tiny ball scatter
    # ------------------------------------------------------------------
    n_params = 3 + n_nb + 3 + 1      # A, dx, dy, nb_amps, B, C, D, log_f

    # Ensure nwalkers is large enough and even
    nwalkers = max(nwalkers, 2 * n_params + 2)
    if nwalkers % 2 != 0:
        nwalkers += 1

    log_f_init   = np.log(float(np.median(Sv)))  # start: noise = typical pixel sigma
    p_center = np.array([A_init, 0.0, 0.0]
                        + list(nb_A_inits)
                        + [0.0, 0.0, float(bkg_init), log_f_init])

    # Scatter walkers in a tiny ball around the MAP, clipping to valid range
    rng   = np.random.default_rng(42)
    scale = np.abs(p_center) * 1e-3 + 1e-6
    p0    = p_center + scale * rng.standard_normal((nwalkers, n_params))
    # Enforce positivity for amplitudes
    p0[:, 0]         = np.abs(p0[:, 0])
    for k in range(n_nb):
        p0[:, 3 + k] = np.abs(p0[:, 3 + k])

    # ------------------------------------------------------------------
    # Run sampler
    # ------------------------------------------------------------------
    sampler = emcee.EnsembleSampler(nwalkers, n_params, _log_prob)

    # Burn-in
    state = sampler.run_mcmc(p0, nburn, progress=False)
    sampler.reset()

    # Production
    sampler.run_mcmc(state, nsteps, progress=False)

    # Acceptance fraction check
    mean_af = float(np.mean(sampler.acceptance_fraction))
    if mean_af < 0.1:
        print(f'  WARNING: low MCMC acceptance fraction ({mean_af:.2f}). '
              f'Consider increasing nwalkers or nsteps.')

    # ------------------------------------------------------------------
    # Extract posterior summaries
    # ------------------------------------------------------------------
    # Flatten chain: shape (nsteps * nwalkers, n_params)
    flat = sampler.get_chain(flat=True)

    def _percentiles(col):
        med = float(np.percentile(flat[:, col], 50))
        lo  = med - float(np.percentile(flat[:, col], 16))   # lower half-width
        hi  = float(np.percentile(flat[:, col], 84)) - med   # upper half-width
        return med, lo, hi

    A_med, A_lo, A_hi = _percentiles(0)

    nb_A_meds, nb_A_los, nb_A_his = [], [], []
    for k in range(n_nb):
        med, lo, hi = _percentiles(3 + k)
        nb_A_meds.append(med);  nb_A_los.append(lo);  nb_A_his.append(hi)

    log_f_med = float(np.percentile(flat[:, 6 + n_nb], 50))

    return A_med, A_lo, A_hi, nb_A_meds, nb_A_los, nb_A_his, log_f_med


def _plot_psf_fit_diag(data, data_err, mask, x_fit, y_fit, x_rel, y_rel,
                       psf_interp, psf_oversampled, half_size, centroid_slack,
                       neighbour_pixels, nb_amplitudes,
                       ax_psf, ax_data, ax_model, ax_resid):
    """
    Fill four diagnostic axes:
        ax_psf   - oversampled PSF image
        ax_data  - masked data cutout; target = red '+', neighbours = orange 'x'
        ax_model - full multi-source + background model
        ax_resid - residuals (data - model), colour-scaled symmetrically

    All source positions are marked on every panel so you can immediately
    judge whether the fit has correctly identified each source.
    """
    n_nb = len(neighbour_pixels)
    ny, nx = data.shape

    # Same box logic as _fit_psf
    all_x = [x_fit] + [p[0] for p in neighbour_pixels]
    all_y = [y_fit] + [p[1] for p in neighbour_pixels]
    pad = half_size
    xlo = max(int(np.floor(min(all_x))) - pad, 0)
    xhi = min(int(np.ceil( max(all_x))) + pad + 1, nx)
    ylo = max(int(np.floor(min(all_y))) - pad, 0)
    yhi = min(int(np.ceil( max(all_y))) + pad + 1, ny)

    xi = np.arange(xlo, xhi);  yi = np.arange(ylo, yhi)
    XX, YY = np.meshgrid(xi, yi)

    data_cut = data[ylo:yhi, xlo:xhi].copy()
    err_cut  = data_err[ylo:yhi, xlo:xhi].copy()
    mask_cut = mask[ylo:yhi, xlo:xhi]
    data_cut[~mask_cut] = np.nan

    valid = mask_cut & np.isfinite(data_cut) & np.isfinite(err_cut) & (err_cut > 0)
    Xv = XX[valid].astype(float);  Yv = YY[valid].astype(float)
    Dv = data_cut[valid].astype(float);  Ev = err_cut[valid].astype(float)

    nb_psf_vals = [psf_interp(Xv - p[0], Yv - p[1]) for p in neighbour_pixels]

    local_med = float(np.nanmedian(Dv)) if valid.any() else 0.0
    A_init    = float(np.nanmax(Dv) - local_med) if valid.any() else 1.0
    if A_init <= 0:
        A_init = 1.0
    nb_A_init = [max(float(a), 0.0) for a in nb_amplitudes]

    p0  = ([A_init, x_fit - x_rel, y_fit - y_rel] + nb_A_init + [0.0, 0.0, local_med])
    blo = ([0.0, -centroid_slack, -centroid_slack] + [0.0] * n_nb + [-np.inf, -np.inf, -np.inf])
    bhi = ([np.inf, centroid_slack, centroid_slack] + [np.inf] * n_nb + [np.inf, np.inf, np.inf])

    # Defaults if quick re-fit fails
    A_plot    = A_init;  B_plot = C_plot = 0.0;  D_plot = local_med
    nb_A_plot = nb_A_init

    try:
        def res(p):
            A, dx, dy = p[0], p[1], p[2]
            nb_amps   = p[3: 3 + n_nb]
            B, C, D   = p[3 + n_nb], p[4 + n_nb], p[5 + n_nb]
            model = (A * psf_interp(Xv - (x_rel + dx), Yv - (y_rel + dy))
                     + B * Xv + C * Yv + D)
            for k, nb_A in enumerate(nb_amps):
                model = model + nb_A * nb_psf_vals[k]
            return (model - Dv) / Ev

        r2        = least_squares(res, p0, bounds=(blo, bhi), method='trf', max_nfev=500)
        A_plot    = float(r2.x[0])
        nb_A_plot = list(r2.x[3: 3 + n_nb])
        B_plot    = float(r2.x[3 + n_nb])
        C_plot    = float(r2.x[4 + n_nb])
        D_plot    = float(r2.x[5 + n_nb])
    except Exception:
        pass

    # Build model image over the full cutout grid
    psf_grid  = psf_interp(XX.ravel() - x_fit, YY.ravel() - y_fit).reshape(XX.shape)
    model_cut = A_plot * psf_grid + B_plot * XX + C_plot * YY + D_plot
    for k, (p, nb_A) in enumerate(zip(neighbour_pixels, nb_A_plot)):
        nb_grid   = psf_interp(XX.ravel() - p[0], YY.ravel() - p[1]).reshape(XX.shape)
        model_cut = model_cut + nb_A * nb_grid
    resid_cut = data_cut - model_cut

    ext         = [xlo, xhi, ylo, yhi]
    finite_data = data_cut[np.isfinite(data_cut)]
    vmin, vmax  = (np.nanpercentile(finite_data, [5, 99])
                   if finite_data.size else (0.0, 1.0))

    def _mark_sources(ax):
        """Overplot target (red +) and neighbours (orange x + label) on ax."""
        ax.plot(x_fit, y_fit, 'r+', ms=12, mew=2, label='Target')
        for k, (x_nb, y_nb) in enumerate(neighbour_pixels):
            ax.plot(x_nb, y_nb, 'x', color='orange', ms=10, mew=2)
            ax.annotate(f'N{k}', xy=(x_nb, y_nb),
                        xytext=(4, 4), textcoords='offset points',
                        color='orange', fontsize=7)

    # PSF panel
    ax_psf.imshow(psf_oversampled, origin='lower', cmap='viridis')
    ax_psf.set_title('PSF (oversampled)')

    # Data panel
    ax_data.imshow(data_cut, origin='lower', extent=ext,
                   vmin=vmin, vmax=vmax, cmap='viridis')
    _mark_sources(ax_data)
    if n_nb > 0:
        ax_data.legend(fontsize=7, loc='upper right')
    ax_data.set_title('Data')

    # Model panel
    ax_model.imshow(model_cut, origin='lower', extent=ext,
                    vmin=vmin, vmax=vmax, cmap='viridis')
    _mark_sources(ax_model)
    ax_model.set_title('PSF model')

    # Residuals panel
    finite_resid = resid_cut[np.isfinite(resid_cut)]
    vlim = float(np.percentile(np.abs(finite_resid), 95)) if finite_resid.size else 1.0
    ax_resid.imshow(resid_cut, origin='lower', extent=ext,
                    vmin=-vlim, vmax=vlim, cmap='RdBu_r')
    _mark_sources(ax_resid)
    ax_resid.set_title('Residuals')


# ==========================================================================
# Shared helper functions (unchanged from original)
# ==========================================================================

def move_forward_position(ra, dec, pmra, pmdec, epoch, new_epoch):
    t = Time(new_epoch, format='mjd', scale='utc')
    decimal_year = t.decimalyear
    c = SkyCoord(
        ra=ra * u.deg,
        dec=dec * u.deg,
        pm_ra_cosdec=pmra * u.mas/u.yr,
        pm_dec=pmdec * u.mas/u.yr,
        frame="icrs",
        obstime=Time(f"J{epoch}")
    )
    c2025 = c.apply_space_motion(Time(f"J{decimal_year}"))
    return c2025.ra.deg, c2025.dec.deg

def get_min_max_plot(data):
    data_flat = data.flatten()
    median    = np.nanmedian(data_flat)
    perc_low  = np.nanpercentile(data_flat, 16)
    perc_high = np.nanpercentile(data_flat, 84)
    return median - perc_low, median + perc_high

def define_mask(flags, flags_header, flags_data):
    flags_num = []
    for x in flags_header['MP*']:
        if x.replace('MP_', '') in flags:
            flags_num.append(flags_header['MP*'][x])
    mask = np.zeros_like(flags_data, dtype=bool)
    for b in flags_num:
        mask |= ((flags_data >> b) & 1) != 0
    good = ~mask
    return good

def source_aperture(data, position_center, r_aperture, data_err=[]):
    aperture_full = CircularAperture(position_center, r=r_aperture)
    if len(data_err) > 0:
        aper_stats_full = ApertureStats(data, aperture_full, error=data_err, sigma_clip=None)
    else:
        aper_stats_full = ApertureStats(data, aperture_full, sigma_clip=None)
    return aperture_full, aper_stats_full

def background_aperture(data, position_center, r_in, r_out, data_err=[]):
    annulus_aperture = CircularAnnulus(position_center, r_in=r_in, r_out=r_out)
    sigclip = SigmaClip(sigma=3.0, maxiters=10)
    if len(data_err) > 0:
        bkg_stats = ApertureStats(data, annulus_aperture, error=data_err, sigma_clip=sigclip)
    else:
        bkg_stats = ApertureStats(data, annulus_aperture, sigma_clip=sigclip)
    return annulus_aperture, bkg_stats

def get_closest_psf(psf_header, psfcube, x_rel, y_rel):
    xctr = {}
    yctr = {}
    for key, val in psf_header.items():
        xm = re.match(r'(XCTR*)', key)
        if xm:
            xplane = int(key.split("_")[1])
            xctr[xplane] = val
        ym = re.match(r'(YCTR*)', key)
        if ym:
            yplane = int(key.split("_")[1])
            yctr[xplane] = val

    tab = Table(names=["zone_id", "x", "y"], dtype=[int, float, float])
    for zone_id in xctr.keys():
        tab.add_row([zone_id, xctr[zone_id], yctr[zone_id]])

    tab["distance"] = np.sqrt((tab["x"] - 1 - x_rel) ** 2 + (tab["y"] - 1 - y_rel) ** 2)
    tab.sort("distance")

    psf_cube_plane = tab[0]["zone_id"]
    distance_min   = tab[0]["distance"]
    psf = psfcube[psf_cube_plane - 1]
    return psf

def get_aperture_corr(psf, x_center, y_center, r_aperture, ax_plot=''):
    positions_psf_center = [(x_center, y_center)]
    aperture_psf, aper_stats_psf = source_aperture(psf, positions_psf_center,
                                                    r_aperture=r_aperture)
    apersum_bkgsub_psf = aper_stats_psf.sum
    aperture_corr = 1.0 / apersum_bkgsub_psf
    if ax_plot != '':
        ax_plot.imshow(psf)
        aperture_psf.plot(ax=ax_plot, color='tab:green', lw=2, label='Photometry aperture')
        ax_plot.set_title('PSF')
    return aperture_corr

def calculate_flux_and_bkg(x_rel, y_rel, image_data, image_data_err, flags_header, flags_data,
                            r_aperture, r_in, r_out, ax_plot1='', ax_plot2='', ax_plot3=''):
    flags = ['SUR_ERROR', 'NONFUNC', 'MISSING_DATA', 'HOT', 'COLD', 'NONLINEAR', 'PERSIST']
    mask_aperture  = define_mask(flags, flags_header, flags_data)
    data_clean     = np.where(mask_aperture, image_data, np.nan)
    data_err_clean = np.where(mask_aperture, image_data_err, np.nan)

    positions = [(x_rel, y_rel)]
    aperture, aper_stats = source_aperture(data_clean, positions, r_aperture, data_err_clean)

    flags = ['OVERFLOW', 'SUR_ERROR', 'NONFUNC', 'MISSING_DATA', 'HOT', 'COLD', 'NONLINEAR',
             'PERSIST', 'OUTLIER', 'SOURCE', 'TRANSIENT']
    mask_bkg           = define_mask(flags, flags_header, flags_data)
    data_bkg_clean     = np.where(mask_bkg, image_data, np.nan)
    data_bkg_err_clean = np.where(mask_bkg, image_data_err, np.nan)

    annulus_aperture, bkg_stats = background_aperture(
        data_bkg_clean, positions, r_in, r_out, data_bkg_err_clean)
    total_bkg     = bkg_stats.median * aper_stats.sum_aper_area.value
    total_bkg_err = np.abs(bkg_stats.std) * aper_stats.sum_aper_area.value

    if ax_plot1 != '':
        delta = r_out + 3
        vmin, vmax = get_min_max_plot(data_clean)
        ax_plot1.imshow(data_clean, interpolation='nearest', vmin=vmin, vmax=vmax, cmap='Grays')
        ax_plot1.set_xlim(x_rel - delta, x_rel + delta)
        ax_plot1.set_ylim(y_rel - delta, y_rel + delta)
        aperture.plot(ax=ax_plot1, color='tab:green', lw=2)
        annulus_aperture.plot(ax=ax_plot1, color='tab:red', lw=2)
        ax_plot1.set_title('Source flux')

        vmin, vmax = get_min_max_plot(data_bkg_clean)
        ax_plot2.imshow(data_bkg_clean, interpolation='nearest', vmin=vmin, vmax=vmax, cmap='Grays')
        ax_plot2.set_xlim(x_rel - delta, x_rel + delta)
        ax_plot2.set_ylim(y_rel - delta, y_rel + delta)
        aperture.plot(ax=ax_plot2, color='tab:green', lw=2)
        annulus_aperture.plot(ax=ax_plot2, color='tab:red', lw=2)
        ax_plot2.set_title('Bkg flux')

        vmin, vmax = get_min_max_plot(image_data)
        ax_plot3.imshow(image_data, interpolation='nearest', vmin=vmin, vmax=vmax, cmap='Grays')
        ax_plot3.set_xlim(x_rel - delta, x_rel + delta)
        ax_plot3.set_ylim(y_rel - delta, y_rel + delta)
        ax_plot3.set_title('Original')

    return aper_stats.sum, aper_stats.sum_err, total_bkg, total_bkg_err

def study_different_apertures(x_rel, y_rel, image_data, image_data_err, flags_header, flags_data,
                              psf, x_center, y_center, oversample, perc, increase_ring,
                              ax_plot, ax_plot2):
    r_aperture_test_image = np.linspace(0.1, 5)
    r_aperture_test_psf   = r_aperture_test_image * oversample

    ap_corr_array = []
    for r_aperture_i in r_aperture_test_psf:
        apcorr = get_aperture_corr(psf, x_center, y_center, r_aperture_i)
        ap_corr_array.append(apcorr)
    ap_corr_array = np.array(ap_corr_array).flatten()

    f_aperture      = interp1d(1 / ap_corr_array, r_aperture_test_image)
    r_aperture_perc = np.float64(f_aperture(perc))
    r_in_perc       = r_aperture_perc * 8 * increase_ring
    r_out_perc      = r_aperture_perc * 10 * increase_ring

    ax_plot.plot(r_aperture_test_image, ap_corr_array, '.-', color='k')
    ax_plot.axvline(x=r_aperture_perc, color='k', linewidth=1, linestyle='--')
    ax_plot.set_xlabel('raperture')
    ax_plot.set_ylabel('apcorr')
    ax_plot.set_yscale('log')

    flux_array = []
    for r_aperture_i in r_aperture_test_image:
        aper_flux, _, total_bkg, _ = calculate_flux_and_bkg(
            x_rel, y_rel, image_data, image_data_err,
            flags_header, flags_data, r_aperture_i, r_in_perc, r_out_perc)
        flux_array.append(aper_flux - total_bkg)
    flux_array = np.array(flux_array).flatten()

    ax_plot2.plot(r_aperture_test_image, flux_array, '.-', color='k',
                  label=f'rin={r_in_perc:.1f}, rout={r_out_perc:.1f}')
    ax_plot2.axvline(x=r_aperture_perc, color='k', linewidth=1, linestyle='--')
    ax_plot2.set_xlabel('raperture')
    ax_plot2.set_ylabel('flux')

    return r_aperture_perc, r_in_perc, r_out_perc

def study_different_rings_image(x_rel, y_rel, image_data, image_data_err, flags_header, flags_data,
                                r_aperture, r_in, r_out, test_plot, ax_plot=''):
    f_test, axs_test    = plt.subplots(1, 2, figsize=(10, 5))
    r_aperture_test_all = np.linspace(r_aperture, 5)
    r_in_test_all       = np.linspace(10, 20)
    r_out_test_all      = np.linspace(r_in + 1, 25)

    cmap   = cm.get_cmap('viridis')
    colors = cmap(np.linspace(0, 1, len(r_in_test_all)))
    for r_in_i, color_i in zip(r_in_test_all, colors):
        r_aperture_test = r_aperture_test_all[r_aperture_test_all < r_in_i]
        r_out_i = r_in_i + 5
        flux_array = []
        for r_aperture_i in r_aperture_test:
            aper_flux, _, total_bkg, _ = calculate_flux_and_bkg(
                x_rel, y_rel, image_data, image_data_err,
                flags_header, flags_data, r_aperture_i, r_in_i, r_out_i)
            flux_array.append(aper_flux - total_bkg)
        flux_array = np.array(flux_array).flatten()
        c_i = 'r' if r_in_i == r_in else color_i
        axs_test[0].plot(r_aperture_test, flux_array, '.-', color=c_i,
                         label=f'rin={r_in_i:.1f}, rout={r_out_i:.1f}')
        axs_test[0].axvline(x=r_aperture, color='k', linewidth=1, linestyle='--')
    axs_test[0].set_xlabel('raperture')
    axs_test[0].set_ylabel('flux')
    axs_test[0].set_title('Changing rin, rout=rin+5')
    axs_test[0].legend(ncols=3)

    r_out_test      = r_out_test_all[r_out_test_all > r_in]
    r_aperture_test = r_aperture_test_all[r_aperture_test_all < r_in]
    colors          = cmap(np.linspace(0, 1, len(r_out_test)))
    for r_out_i, color_i in zip(r_out_test, colors):
        flux_array = []
        for r_aperture_i in r_aperture_test:
            aper_flux, _, total_bkg, _ = calculate_flux_and_bkg(
                x_rel, y_rel, image_data, image_data_err,
                flags_header, flags_data, r_aperture_i, r_in, r_out_i)
            flux_array.append(aper_flux - total_bkg)
        flux_array = np.array(flux_array).flatten()
        if r_out_i == r_out:
            c_i = 'r'
            if ax_plot != '':
                ax_plot.plot(r_aperture_test, flux_array, '.-', color='k',
                             label=f'rin={r_in:.1f}, rout={r_out:.1f}')
                ax_plot.axvline(x=r_aperture, color='k', linewidth=1, linestyle='--')
                ax_plot.set_xlabel('raperture')
                ax_plot.set_ylabel('flux')
        else:
            c_i = color_i
        axs_test[1].plot(r_aperture_test, flux_array, '.-', color=c_i,
                         label=f'rin={r_in:.1f}, rout={r_out_i:.1f}')
        axs_test[1].axvline(x=r_aperture, color='k', linewidth=1, linestyle='--')
    axs_test[1].set_xlabel('raperture')
    axs_test[1].set_ylabel('flux')
    axs_test[1].set_title('Changing rout')
    axs_test[1].legend(ncols=3)
    f_test.savefig(test_plot, dpi=300)