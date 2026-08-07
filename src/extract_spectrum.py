from astropy.time import Time
from astropy.coordinates import SkyCoord
from astropy import units as u
import numpy as np
from photutils.aperture import CircularAperture,aperture_photometry, CircularAnnulus
from photutils.aperture import ApertureStats
from astropy.stats import SigmaClip
import re
from astropy.table import Table
import matplotlib.pyplot as plt
from scipy.interpolate import interp1d
import matplotlib.cm as cm
from astropy.wcs import WCS
from astropy.io import fits
from matplotlib.gridspec import GridSpec
import astropy
import requests
import os

from .flux_renormalize import *
from .extras import *

def extract_spectrum(name,results_irsa,ra_deg,dec_deg,pmra,pmdec,epoch,perc,
                     path_figures,path_data):
    '''
    Main function to extract spectra from SPHEREx
    name (str) Objects name
    results_irsa (table) Results from IRSA query for the position of the object
    ra_deg (float) Right ascension of the object
    dec_deg (float) Declination of the object
    pmra (float) Proper motion in right ascension 
    pmdec (float) Proper motion in declination 
    epoch (float) Epoch of the ra and dec
    perc (float) Percentage of flux included in the aperture
    path_figures (str) Path where to save output figures
    path_data (str) Path where to save extracted spectrum
    '''

    # Prepare to extract spectrum
    wave = [] # Wavelength 
    flux = [] # Flux
    flux_err = [] # Flux error
    r_ap_array = [] # Aperture radius used in each image
    r_in_array = [] # Inner radius from background ring used in each image
    r_out_array = [] # Outer radius from background ring used in each image
    obs_date_mjd_array = [] # Epoch for each image

    results_spec = Table() # Final table with full spectrum

    # Loop over all the images for each object
    for i in range(len(results_irsa)):

        spectral_image_url = results_irsa['access_url'][i]
        print(f'{i} out of {len(results_irsa)}:' + spectral_image_url)

        # Download image
        # Repeat the task until it works given that IRSA fails sometimes
        output_path = path_data + "temporary.fits"
        downloaded = 0
        while downloaded == 0:
            try:
                with requests.get(spectral_image_url, stream=True) as r:
                    r.raise_for_status()
                    with open(output_path, "wb") as f:
                        for chunk in r.iter_content(chunk_size= 
                                                    2 * 2 * 1024 * 1024):  #1MB  
                            if chunk:  # filter out keep-alive chunks
                                f.write(chunk)
                downloaded = 1
            except:
                downloaded = 0

        # Open image and remove downloaded temporary file
        hdulist = fits.open(output_path)
        os.remove(output_path)
        hdulist = update_psf_header(hdulist)

        #set up diagnostic plots
        fig = plt.figure(figsize=(15, 8))
        gs = GridSpec(2, 4, figure=fig) 
        ax0 = fig.add_subplot(gs[0, 0])
        ax1 = fig.add_subplot(gs[0, 1])
        ax2 = fig.add_subplot(gs[0, 2])
        ax3 = fig.add_subplot(gs[0, 3])
        ax4 = fig.add_subplot(gs[1, :2])
        ax5 = fig.add_subplot(gs[1, 2:])

        # Extract relevant information
        spectral_image_header = hdulist[1].header
        psf_header = hdulist[5].header
        psfcube = hdulist[5].data
        flags_header = hdulist[2].header
        flags_data = hdulist[2].data
        image_data = hdulist[1].data
        image_data_err = np.sqrt(hdulist[3].data)
        obs_date_mjd = hdulist[1].header['MJD-OBS']

        # Get coordinates in real world transformation
        new_ra, new_dec = move_forward_position(ra_deg,dec_deg,pmra,pmdec,
                                                epoch,results_irsa['t_min'][i])
        wcs = WCS(spectral_image_header)
        # Get pixel position for relevant coord
        x_rel, y_rel = wcs.world_to_pixel(SkyCoord(ra=new_ra, dec=new_dec, 
                                                   unit="deg"))

        # Get wavelength for the position of the star
        spectral_wcs = WCS(header=hdulist["IMAGE"].header, fobj=hdulist, 
                           key="W")
        spectral_wcs.sip = None
        wl, bw = spectral_wcs.pixel_to_world(x_rel, y_rel)
        wave.append(wl.value)

        obs_date_mjd_array.append(obs_date_mjd)

        if not isinstance(perc, list):
            perc = [perc]

        cmap = plt.colormaps['plasma'] 
        colors_perc = cmap(np.linspace(0, 1, len(perc)))

        # Estimate aperture radius so perc of flux is included
        psf = get_closest_psf(psf_header, psfcube, x_rel, y_rel)
        r = study_different_apertures(x_rel, y_rel, image_data, image_data_err, 
                                      flags_header, flags_data, 
                                      psf, psf.shape[0]/2-1, psf.shape[0]/2-1, 
                                      psf_header['OVERSAMP'], perc, ax4, ax5,
                                      colors_perc)
        
        r_aperture, r_in, r_out = r
        r_ap_array.append(r_aperture)
        r_in_array.append(r_in)
        r_out_array.append(r_out)



        flux_j = []
        flux_err_j = []
        for j in range(len(perc)):

            # Calculate aperture correction for that position
            aperture_corr = get_aperture_corr(psf, psf.shape[0]/2-1, 
                                            psf.shape[0]/2-1, 
                                            r_aperture[j]*psf_header['OVERSAMP'], 
                                            ax0,colors_perc[j])


            
            r =  calculate_flux_and_bkg(x_rel, y_rel, image_data, image_data_err, 
                                        flags_header, flags_data, r_aperture[j], 
                                        r_in[j], r_out[j], ax1, ax2, ax3,
                                        colors_perc[j])
            aper_flux, aper_flux_err, total_bkg, total_bkg_err = r


            # Subtract background from aperture
            apersum_bkgsub = aper_flux - total_bkg
            apersum_bkgsub_err = np.sqrt(aper_flux_err**2+total_bkg_err**2)

            # Turn flux from MJy/sr->uJy
            omega_arcsec2 = hdulist[1].header['HIERARCH OMEGA_MEDIAN'] #arcsec2
            omega_err_arcsec2 = hdulist[1].header['HIERARCH OMEGA_STDDEV'] #arcsec2

            omega_st = omega_arcsec2 / (206265 ** 2) #st
            omega_st_err = omega_err_arcsec2 / (206265 ** 2) #st
            aperture_skysub_ujy = apersum_bkgsub * omega_st * 1e12 * aperture_corr 
            aperture_skysub_ujy_err = np.sqrt((apersum_bkgsub_err * omega_st * 1e12 * aperture_corr)**2 + 
                                            (apersum_bkgsub * omega_st_err * 1e12 * aperture_corr )**2)
            flux_j.append(aperture_skysub_ujy[0])
            flux_err_j.append(aperture_skysub_ujy_err[0])

        flux.append(flux_j)
        flux_err.append(flux_err_j)

        plt.tight_layout()
        plt.savefig(path_figures+name+'/'+str(np.round(wl.value,2))+'_phot.png',
                    dpi=100, bbox_inches = 'tight')
        plt.close(fig)

        # Delete temporary file created to download the full data that is not needed
        astropy.utils.data.clear_download_cache(hashorurl=None, pkgname='astropy')

    # Save results
    
    results_spec['Wave'] = np.array(wave)
    results_spec['Wave'].unit = 'um'
    results_spec['Wave'].description = 'Observed wavelength'

    for j in range(len(perc)):
        results_spec['Flux_'+str(perc[j])] = np.array([flux_i[j] for flux_i in flux])
        results_spec['Flux_'+str(perc[j])].unit = 'uJy'
        results_spec['Flux_'+str(perc[j])].description = 'Flux density with ' + str(perc[j]) + ' percent aperture'
        results_spec['Fluxerr_'+str(perc[j])] = np.array([flux_err_i[j] for flux_err_i in flux_err])
        results_spec['Fluxerr_'+str(perc[j])].unit = 'uJy'
        results_spec['Fluxerr_'+str(perc[j])].description = '1-sigma flux uncertainty with ' + str(perc[j]) + ' percent aperture'

        results_spec['r_aperture_'+str(perc[j])] = np.array([r_ap_i[j] for r_ap_i in r_ap_array])
        results_spec['r_aperture_'+str(perc[j])].unit = 'pix'
        results_spec['r_aperture_'+str(perc[j])].description = 'Photometric aperture radius with ' + str(perc[j]) + ' percent aperture'
        results_spec['r_in_'+str(perc[j])] = np.array([r_in_i[j] for r_in_i in r_in_array])
        results_spec['r_in_'+str(perc[j])].unit = 'pix'
        results_spec['r_in_'+str(perc[j])].description = 'Sky annulus inner radius with ' + str(perc[j]) + ' percent aperture'
        results_spec['r_out_'+str(perc[j])] = np.array([r_out_i[j] for r_out_i in r_out_array]) 
        results_spec['r_out_'+str(perc[j])].unit = 'pix'
        results_spec['r_out_'+str(perc[j])].description = 'Sky annulus outer radius with ' + str(perc[j]) + ' percent aperture'

    results_spec['OBSMJD'] = np.array(obs_date_mjd_array)
    results_spec['OBSMJD'].description = 'Observation date (MJD)'

    results_spec.meta['NAME'] = name
    results_spec.meta['RA'] = ra_deg
    results_spec.meta['DEC'] = dec_deg
    results_spec.meta['PMRA'] = pmra
    results_spec.meta['PMDEC'] =pmdec
    results_spec.meta['EPOCH'] = epoch
    perc_str = ','.join(str(p) for p in perc)
    results_spec.meta['PERC'] = perc_str

    results_spec.meta['comments'] = [
    'NAME: Target identifier',
    'RA, DEC: ICRS coordinates in degrees',
    'PMRA, PMDEC: proper motion in mas/yr',
    'EPOCH: epoch of the position',
    'PERC: aperture correction percentage',
    ]
    
    # Plot final spectrum
    for j in range(len(perc)):
        flux_perc = np.array([flux_i[j] for flux_i in flux])
        flux_err_perc = np.array([flux_err_i[j] for flux_err_i in flux_err])
        plt.errorbar(wave,np.array(flux_perc),yerr=np.array(flux_err_perc),
                    fmt='.',label='Percentage aperture '+str(perc[j]),
                    color=colors_perc[j])
    plt.legend(title=name)
    plt.xscale('log')
    plt.xlabel('Wavelength (um)')
    plt.ylabel('Flux (uJy)')
    plt.savefig(path_figures+name+'/'+'_comparison_spec.png',dpi=100)
    #plt.show()
    plt.close()

    return results_spec



def move_forward_position(ra,dec,pmra,pmdec,epoch,new_epoch):
    '''
    Propagate position from the known epoch to the new_epoch from each image
    from SPHEREx
    '''
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
    '''
    Estimate min and max from plot to adjust colorbar
    '''
    data_flat = data.flatten()
    median = np.nanmedian(data_flat)
    perc_low = np.nanpercentile(data_flat,16)
    perc_high = np.nanpercentile(data_flat,84)
    return median-perc_low, median+perc_high

def define_mask(flags, flags_header, flags_data):
    '''
    Define mask to remove bad pixel and known sources from the image
    '''
    flags_num = []
    for x in flags_header['MP*']:
        if x.replace('MP_','') in flags:
            flags_num.append(flags_header['MP*'][x])

    mask = np.zeros_like(flags_data, dtype=bool)
    for b in flags_num:
        mask |= ((flags_data >> b) & 1) != 0
    good = ~mask
    return good

def source_aperture(data, position_center, r_aperture, data_err=[]):
    '''
    Calculate flux in the aperture
    '''
    aperture_full = CircularAperture(position_center, r=r_aperture)
    if len(data_err)>0:
        aper_stats_full = ApertureStats(data, aperture_full, 
                                        error=data_err, sigma_clip=None)
    else:
        aper_stats_full = ApertureStats(data, aperture_full, sigma_clip=None)
    return aperture_full, aper_stats_full

def background_aperture(data, position_center, r_in, r_out, data_err=[]):
    '''
    Estimate background flux for the aperture correction
    '''
    annulus_aperture = CircularAnnulus(position_center, r_in=r_in, r_out=r_out)
    sigclip = SigmaClip(sigma=3.0, maxiters=10)
    if len(data_err) > 0:
        bkg_stats = ApertureStats(data, annulus_aperture, error=data_err, sigma_clip=sigclip)
    else: 
        bkg_stats = ApertureStats(data, annulus_aperture, sigma_clip=sigclip)
    return annulus_aperture, bkg_stats
     
def get_closest_psf(psf_header, psfcube, x_rel, y_rel):
    '''
    Look for the positions of the psfs in the data file
    '''
    xctr = {}
    yctr = {}
    for key, val in psf_header.items():
        # Look for keys like XCTR* or YCTR*
        xm = re.match(r'(XCTR*)', key)
        if xm:
            xplane = int(key.split("_")[1])
            xctr[xplane] = val
        ym = re.match(r'(YCTR*)', key)
        if ym:
            yplane = int(key.split("_")[1])
            yctr[yplane] = val  

    # Sort all psf according to distance to relevant coordinates
    tab = Table(names=["zone_id" , "x" , "y"], dtype=[int, float, float])
    for zone_id in xctr.keys():
        tab.add_row([zone_id , xctr[zone_id] , yctr[zone_id]])

    tab["distance"] = np.sqrt((tab["x"]-1 - x_rel)**2 + (tab["y"]-1 - y_rel)**2)
    tab.sort("distance")

    # Select the psf that is closest to the coordinates of interest
    psf_cube_plane = tab[0]["zone_id"]
    distance_min = tab[0]["distance"]
    psf = psfcube[psf_cube_plane-1]

    return psf

def get_aperture_corr(psf, x_center, y_center, r_aperture, ax_plot='', color=''):
    '''
    Estimate aperture correction
    '''
    positions_psf_center = [(x_center, y_center)]
    aperture_psf, aper_stats_psf = source_aperture(psf, positions_psf_center, 
                                                   r_aperture=r_aperture)

    apersum_bkgsub_psf = aper_stats_psf.sum 

    aperture_corr = 1.0/(apersum_bkgsub_psf)

    # Plot results
    if ax_plot!='':
        ax_plot.imshow(psf)
        ap_patches = aperture_psf.plot(ax=ax_plot,color=color, lw=2,
                                label='Photometry aperture')
        ax_plot.set_title('PSF')

    return aperture_corr

def calculate_flux_and_bkg(x_rel, y_rel, image_data, image_data_err, 
                           flags_header, flags_data, r_aperture, 
                           r_in, r_out, ax_plot1='', ax_plot2='', ax_plot3='',
                           color_i=''):
    #flags for aperture
    flags = ['SUR_ERROR', 'NONFUNC', 'MISSING_DATA', 'HOT', 'COLD','NONLINEAR',
             'PERSIST','BLOOM','SNOWBALL','HALO','SATELLITE_HALO']
    mask_aperture = define_mask(flags, flags_header, flags_data)
    data_clean = np.where(mask_aperture, image_data, np.nan)
    data_err_clean = np.where(mask_aperture, image_data_err, np.nan)

    # Calculate flux in the selected aperture
    positions = [(x_rel, y_rel)]

    aperture, aper_stats = source_aperture(data_clean, positions, r_aperture, data_err_clean)

    #flags for photometry background
    flags = ['OVERFLOW', 'SUR_ERROR', 'NONFUNC', 'MISSING_DATA', 'HOT', 
             'COLD', 'NONLINEAR', 'PERSIST', 'OUTLIER', 'SOURCE', 'TRANSIENT',
             'BLOOM','SNOWBALL','HALO','SATELLITE_HALO']
    mask_bkg = define_mask(flags, flags_header, flags_data)
    data_bkg_clean = np.where(mask_bkg, image_data, np.nan)
    data_bkg_err_clean = np.where(mask_bkg, image_data_err, np.nan)

    # Calculate background flux
    annulus_aperture, bkg_stats = background_aperture(data_bkg_clean, positions, r_in, r_out, data_bkg_err_clean)
    total_bkg = bkg_stats.median * aper_stats.sum_aper_area.value
    n_bkg = bkg_stats.sum_aper_area.value
    total_bkg_err = (np.sqrt(np.pi / 2.0) * bkg_stats.std / np.sqrt(n_bkg)) * aper_stats.sum_aper_area.value

    if ax_plot1 != '':
        delta = r_out+3
        vmin, vmax = get_min_max_plot(data_clean)
        ax_plot1.imshow(data_clean, interpolation='nearest', 
                        vmin=vmin, vmax=vmax, cmap='Grays')
        ax_plot1.set_xlim(x_rel-delta,x_rel+delta)
        ax_plot1.set_ylim(y_rel-delta,y_rel+delta)
        ap_patches = aperture.plot(ax=ax_plot1, color=color_i, lw=2)
        ann_patches = annulus_aperture.plot(ax=ax_plot1, color=color_i, lw=2)
        ax_plot1.set_title('Source flux')

        vmin, vmax = get_min_max_plot(data_bkg_clean)
        ax_plot2.imshow(data_bkg_clean, interpolation='nearest', 
                        vmin=vmin,vmax=vmax,cmap='Grays')
        ax_plot2.set_xlim(x_rel-delta,x_rel+delta)
        ax_plot2.set_ylim(y_rel-delta,y_rel+delta)
        ap_patches = aperture.plot(ax=ax_plot2, color=color_i, lw=2)
        ann_patches = annulus_aperture.plot(ax=ax_plot2, color=color_i, lw=2)
        ax_plot2.set_title('Bkg flux')

        vmin, vmax = get_min_max_plot(image_data)
        ax_plot3.imshow(image_data, interpolation='nearest', 
                        vmin=vmin,vmax=vmax,cmap='Grays')
        ax_plot3.set_xlim(x_rel-delta,x_rel+delta)
        ax_plot3.set_ylim(y_rel-delta,y_rel+delta)
        ax_plot3.set_title('Original')

    return aper_stats.sum, aper_stats.sum_err, total_bkg, total_bkg_err

def study_different_apertures(x_rel, y_rel, image_data, image_data_err, 
                              flags_header, flags_data, psf, x_center, 
                              y_center, oversample, perc, ax_plot, ax_plot2,
                              colors):
    '''
    Compare flux using different apertures
    '''
    r_aperture_test_image = np.linspace(0.1,5,100)
    r_aperture_test_psf = r_aperture_test_image*oversample

    ap_corr_array = []
    for r_aperture_i in r_aperture_test_psf:
        apcorr = get_aperture_corr(psf, x_center, y_center, r_aperture_i)
        ap_corr_array.append(apcorr)
    ap_corr_array = np.array(ap_corr_array).flatten()

    # Add values of aperture correction to plot
    ax_plot.plot(r_aperture_test_image, ap_corr_array, '.-', color='k')

    # Interpolate aperture correction
    f_aperture = interp1d(1/ap_corr_array,r_aperture_test_image)

    r_aperture_perc = []
    r_in_perc = []
    r_out_perc = []
    for perc_i,color_i in zip(perc,colors):
        r_aperture_perc_i = f_aperture(perc_i)
        r_aperture_perc_i = np.float64(r_aperture_perc_i)

        r_in_perc_i = r_aperture_perc_i*8
        r_out_perc_i = r_aperture_perc_i*10

        r_aperture_perc.append(r_aperture_perc_i)
        r_in_perc.append(r_in_perc_i)
        r_out_perc.append(r_out_perc_i)

        ax_plot.axvline(x=r_aperture_perc_i,color=color_i,linewidth=1,
                        linestyle='--',label='Aperture: '+str(perc_i))
    ax_plot.set_xlabel('raperture')
    ax_plot.set_ylabel('apcorr')
    ax_plot.set_yscale('log')

    flux_array = []
    for r_aperture_i in r_aperture_test_image:
        r =  calculate_flux_and_bkg(x_rel, y_rel, 
                                    image_data, image_data_err, 
                                    flags_header, flags_data, 
                                    r_aperture_i, 
                                    r_aperture_i*8, r_aperture_i*10)
        aper_flux,_,total_bkg,_  = r
        apersum_bkgsub = aper_flux - total_bkg
        flux_array.append(apersum_bkgsub)
    flux_array = np.array(flux_array).flatten()
        
    ax_plot2.plot(r_aperture_test_image, flux_array, '.-', color='k')
    for i in range(len(perc)):
        ax_plot2.axvline(x=r_aperture_perc[i],color=colors[i],
                         linewidth=1,linestyle='--')
    ax_plot2.set_xlabel('raperture')
    ax_plot2.set_ylabel('flux')

    return r_aperture_perc, r_in_perc, r_out_perc

def study_different_rings_image(x_rel, y_rel, image_data, image_data_err, 
                                flags_header, flags_data, 
                                r_aperture, r_in, r_out, test_plot, 
                                ax_plot=''):

    f_test,axs_test = plt.subplots(1,2,figsize=(10,5))
    r_aperture_test_all = np.linspace(r_aperture,5)
    r_in_test_all = np.linspace(10,20)
    r_out_test_all = np.linspace(r_in+1,25)

    cmap = cm.get_cmap('viridis')
    colors = cmap(np.linspace(0, 1, len(r_in_test_all)))
    for r_in_i,color_i in zip(r_in_test_all,colors):
        r_aperture_test = r_aperture_test_all[r_aperture_test_all<r_in_i]
        r_out_i = r_in_i + 5
        flux_array = []
        for r_aperture_i in r_aperture_test:
            r =  calculate_flux_and_bkg(x_rel, y_rel, 
                                        image_data, image_data_err, 
                                        flags_header, flags_data, 
                                        r_aperture_i, r_in_i, r_out_i)
            aper_flux,_,total_bkg,_ = r
            apersum_bkgsub = aper_flux - total_bkg
            flux_array.append(apersum_bkgsub)
        flux_array = np.array(flux_array).flatten()
        if r_in_i == r_in:
            c_i = 'r'
        else:
            c_i = color_i
        axs_test[0].plot(r_aperture_test, flux_array, '.-', color=c_i, 
                         label=f'rin={r_in_i}, rout={r_out_i}')
        axs_test[0].axvline(x=r_aperture,color='k',linewidth=1,linestyle='--')
    axs_test[0].set_xlabel('raperture')
    axs_test[0].set_ylabel('flux')
    axs_test[0].set_title('Changing rin, rout=rin+5')
    axs_test[0].legend(ncols=3)

    r_out_test=r_out_test_all[r_out_test_all>r_in]
    r_aperture_test = r_aperture_test_all[r_aperture_test_all<r_in]
    colors = cmap(np.linspace(0, 1, len(r_out_test)))
    for r_out_i,color_i in zip(r_out_test,colors):
        flux_array = []
        for r_aperture_i in r_aperture_test:
            r =  calculate_flux_and_bkg(x_rel, y_rel, 
                                        image_data, image_data_err, 
                                        flags_header, flags_data, 
                                        r_aperture_i, r_in, r_out_i)
            aper_flux,_,total_bkg,_ = r
            apersum_bkgsub = aper_flux - total_bkg
            flux_array.append(apersum_bkgsub)
        flux_array = np.array(flux_array).flatten()
        if r_out_i == r_out:
            c_i = 'r'
            ax_plot.plot(r_aperture_test, flux_array, '.-', color='k', 
                         label=f'rin={r_in}, rout={r_out}')
            ax_plot.axvline(x=r_aperture,color='k',linewidth=1,linestyle='--')
            ax_plot.set_xlabel('raperture')
            ax_plot.set_ylabel('flux')
        else:
            c_i = color_i
        axs_test[1].plot(r_aperture_test, flux_array, '.-', color=c_i, 
                         label=f'rin={r_in}, rout={r_out_i}')
        axs_test[1].axvline(x=r_aperture,color='k',linewidth=1,linestyle='--')
    axs_test[1].set_xlabel('raperture')
    axs_test[1].set_ylabel('flux')
    axs_test[1].set_title('Changing rout')
    axs_test[1].legend(ncols=3)
    f_test.savefig(test_plot,dpi=300)

