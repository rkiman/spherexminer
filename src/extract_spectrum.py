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
from .flux_renormalize import *
import astropy
import requests
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry
import os


def extrac_spectrum(name,results,results_spec,ra_deg,dec_deg,pmra,pmdec,epoch,perc,test_external_ring,
                    known_data,i_known_data,path_figures,path_data,increase_ring = 1):
    wave = []
    flux = []
    flux_err = []
    r_ap_array = []
    r_in_array = []
    r_out_array = []
    idx_nan_array = []

    for i in range(len(results)):

        spectral_image_url = results['access_url'][i]
        print(f'{i} out of {len(results)}:' + spectral_image_url)

        # Download image
        output_path = path_data + "temporary.fits"
        downloaded = 0
        while downloaded == 0:
            try:
                with requests.get(spectral_image_url, stream=True) as r:
                    r.raise_for_status()
                    with open(output_path, "wb") as f:
                        for chunk in r.iter_content(chunk_size= 2 * 2 * 1024 * 1024):  # 1 MB chunks  
                            if chunk:  # filter out keep-alive chunks
                                f.write(chunk)
                downloaded = 1
            except:
                downloaded = 0

        hdulist = fits.open(output_path)

        os.remove(output_path)


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

        # Get coordinates in real world transformation
        new_ra, new_dec = move_forward_position(ra_deg,dec_deg,pmra,pmdec,epoch,results['t_min'][i])
        wcs = WCS(spectral_image_header)
        # Get pixel position for relevant coord
        x_rel, y_rel = wcs.world_to_pixel(SkyCoord(ra=new_ra, dec=new_dec, unit="deg"))

        # Estimate aperture radius so perc of flux is included
        psf = get_closest_psf(psf_header, psfcube, x_rel, y_rel)
        r_aperture, r_in, r_out = study_different_apertures(x_rel, y_rel, image_data, image_data_err, flags_header, flags_data, 
                                                        psf, psf.shape[0]/2-1, psf.shape[0]/2-1, psf_header['OVERSAMP'], perc, increase_ring, ax4, ax5)
        r_ap_array.append(r_aperture)
        r_in_array.append(r_in)
        r_out_array.append(r_out)

        # Calculate aperture correction for that position
        aperture_corr = get_aperture_corr(psf, psf.shape[0]/2-1, psf.shape[0]/2-1, r_aperture*psf_header['OVERSAMP'], ax0)

        if test_external_ring:
            study_different_rings_image(x_rel, y_rel, image_data, image_data_err, 
                                        flags_header, flags_data, r_aperture, r_in, r_out,
                                        '../Figures/'+name+'/'+str(np.round(wl.value,2))+'_phot_test.png')
        
        

        # Get wavelength for the position of the star
        spectral_wcs = WCS(header=hdulist["IMAGE"].header, fobj=hdulist, key="W")
        spectral_wcs.sip = None
        wl, bw = spectral_wcs.pixel_to_world(x_rel, y_rel)
        wave.append(wl.value)
        
        aper_flux, aper_flux_err, total_bkg, total_bkg_err =  calculate_flux_and_bkg(x_rel, y_rel, image_data, image_data_err, flags_header, flags_data, r_aperture, r_in, r_out, ax1, ax2, ax3)
        while str(total_bkg)=='nan':
            r_in, r_out = 2*r_in, 2*r_out
            aper_flux, aper_flux_err, total_bkg, total_bkg_err =  calculate_flux_and_bkg(x_rel, y_rel, image_data, image_data_err, flags_header, flags_data, r_aperture, r_in, r_out, ax1, ax2, ax3)


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
        flux.append(aperture_skysub_ujy[0])
        flux_err.append(aperture_skysub_ujy_err[0])
        
        plt.tight_layout()
        plt.savefig(path_figures+name+'/'+str(np.round(wl.value,2))+'_phot_'+str(perc)+'.png',dpi=100, bbox_inches = 'tight')
        plt.close(fig)

        # Delete temporary file created to download the full data that is not needed
        astropy.utils.data.clear_download_cache(hashorurl=None, pkgname='astropy')

    
    results_spec['Wave [um] '+str(perc)] = np.array(wave)
    results_spec['Flux [uJy] '+str(perc)] = np.array(flux)
    results_spec['Flux err [uJy] '+str(perc)] = np.array(flux_err)
    results_spec['r aperture '+str(perc)] = np.array(r_ap_array)
    results_spec['r in '+str(perc)] = np.array(r_in_array)
    results_spec['r out '+str(perc)] = np.array(r_out_array)
    #results_spec['Idx nan '+str(perc)] = np.array(idx_nan_array)
    
    try:
        known_data_spec = Table.read('../Data/'+known_data['File'][i_known_data], format='ascii')
        wave_spex = known_data_spec['col1']
        flux_spex = known_data_spec['col2']
        flux_spex_err = known_data_spec['col3']
        fil_phot_synt = ['J','H','K']
        phot_measurement = [known_data['J'][i_known_data],known_data['H'][i_known_data],known_data['K'][i_known_data]]
        phot_err_measurement = [known_data['J_err'][i_known_data],known_data['H_err'][i_known_data],known_data['H_err'][i_known_data]]
        scaling, _, _, _, _, _, _, _, _ = renormalize_spectrum(fil_phot_synt, phot_measurement, phot_err_measurement, wave_spex, flux_spex, flux_spex_err,
                                                            filter_path='/Users/rociokiman/Hunter College Dropbox/Rocio Kiman/papers/_Published/Coconuts2/Data/Filters/')
        flux_spex_norm = scaling*flux_spex #erg/s/cm2/A
        flux_spex_norm =flux_spex_norm * (3.33564095e+04*(wave_spex*1e4)**2) * 1e6 #uJy
        flux_spex_err_norm = scaling * flux_spex_err * (3.33564095e+04*(wave_spex*1e4)**2) * 1e6

        plt.errorbar(wave_spex,flux_spex_norm,yerr=flux_spex_err_norm, fmt='',color='lightgray')
        plt.errorbar(wave_spex,flux_spex_norm,fmt='-',label='Spex')
    except:
        a = 0

    plt.errorbar(wave,np.array(flux),yerr=np.array(flux_err),fmt='.',label='SPHEREx')
    plt.legend(title=name)
    plt.xscale('log')
    plt.xlabel('Wavelength [um]')
    plt.ylabel('Flux [uJy]')
    plt.savefig(path_figures+name+'/'+'_comparison_spec_'+str(perc)+'.png',dpi=100)
    plt.show()

    return results_spec



def move_forward_position(ra,dec,pmra,pmdec,epoch,new_epoch):
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
    median = np.nanmedian(data_flat)
    perc_low = np.nanpercentile(data_flat,16)
    perc_high = np.nanpercentile(data_flat,84)
    return median-perc_low, median+perc_high

def define_mask(flags, flags_header, flags_data):
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

    aperture_full = CircularAperture(position_center, r=r_aperture)
    if len(data_err)>0:
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
    # Look for the positions of the psfs in the data file
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
            yctr[xplane] = val  

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

def get_aperture_corr(psf, x_center, y_center, r_aperture, ax_plot=''):
    positions_psf_center = [(x_center, y_center)]

    #aperture_psf_full, aper_stats_psf_full = source_aperture(psf, positions_psf_center, r_aperture=60.0)
    aperture_psf, aper_stats_psf = source_aperture(psf, positions_psf_center, r_aperture=r_aperture)

    #annulus_aperture_psf, bkg_stats_psf = background_aperture(psf, positions_psf_center, r_in, r_out)
    #total_bkg_psf = bkg_stats_psf.median * aper_stats_psf.sum_aper_area.value

    apersum_bkgsub_psf = aper_stats_psf.sum #- total_bkg_psf

    aperture_corr = 1.0/(apersum_bkgsub_psf)#/aper_stats_psf_full.sum)

    # Plot results
    if ax_plot!='':
        ax_plot.imshow(psf)
        ap_patches = aperture_psf.plot(ax=ax_plot,color='tab:green', lw=2,
                                label='Photometry aperture')
        ax_plot.set_title('PSF')

    return aperture_corr

def calculate_flux_and_bkg(x_rel, y_rel, image_data, image_data_err, flags_header, flags_data, r_aperture, r_in, r_out, ax_plot1='', ax_plot2='', ax_plot3=''):
    #flags for aperture
    flags = ['SUR_ERROR', 'NONFUNC', 'MISSING_DATA', 'HOT', 'COLD','NONLINEAR','PERSIST']
    mask_aperture = define_mask(flags, flags_header, flags_data)
    data_clean = np.where(mask_aperture, image_data, np.nan)
    data_err_clean = np.where(mask_aperture, image_data_err, np.nan)

    # Calculate flux in the selected aperture
    positions = [(x_rel, y_rel)]
    
    aperture, aper_stats = source_aperture(data_clean, positions, r_aperture, data_err_clean)

    #flags for photometry background
    flags = ['OVERFLOW', 'SUR_ERROR', 'NONFUNC', 'MISSING_DATA', 'HOT', 'COLD', 'NONLINEAR',
            'PERSIST', 'OUTLIER', 'SOURCE', 'TRANSIENT']
    mask_bkg = define_mask(flags, flags_header, flags_data)
    data_bkg_clean = np.where(mask_bkg, image_data, np.nan)
    data_bkg_err_clean = np.where(mask_bkg, image_data_err, np.nan)

    # Calculate background flux
    annulus_aperture, bkg_stats = background_aperture(data_bkg_clean, positions, r_in, r_out, data_bkg_err_clean)
    total_bkg = bkg_stats.median * aper_stats.sum_aper_area.value
    total_bkg_err = np.abs(bkg_stats.std) * aper_stats.sum_aper_area.value

    if ax_plot1 != '':
        delta = r_out+3
        vmin, vmax = get_min_max_plot(data_clean)
        ax_plot1.imshow(data_clean, interpolation='nearest', vmin=vmin, vmax=vmax, cmap='Grays')
        ax_plot1.set_xlim(x_rel-delta,x_rel+delta)
        ax_plot1.set_ylim(y_rel-delta,y_rel+delta)
        ap_patches = aperture.plot(ax=ax_plot1, color='tab:green', lw=2)
        ann_patches = annulus_aperture.plot(ax=ax_plot1, color='tab:red', lw=2)
        ax_plot1.set_title('Source flux')

        vmin, vmax = get_min_max_plot(data_bkg_clean)
        ax_plot2.imshow(data_bkg_clean, interpolation='nearest', vmin=vmin,vmax=vmax,cmap='Grays')
        ax_plot2.set_xlim(x_rel-delta,x_rel+delta)
        ax_plot2.set_ylim(y_rel-delta,y_rel+delta)
        ap_patches = aperture.plot(ax=ax_plot2, color='tab:green', lw=2)
        ann_patches = annulus_aperture.plot(ax=ax_plot2, color='tab:red', lw=2)
        ax_plot2.set_title('Bkg flux')

        vmin, vmax = get_min_max_plot(image_data)
        ax_plot3.imshow(image_data, interpolation='nearest', vmin=vmin,vmax=vmax,cmap='Grays')
        ax_plot3.set_xlim(x_rel-delta,x_rel+delta)
        ax_plot3.set_ylim(y_rel-delta,y_rel+delta)
        ax_plot3.set_title('Original')

    return aper_stats.sum, aper_stats.sum_err, total_bkg, total_bkg_err

def study_different_apertures(x_rel, y_rel, image_data, image_data_err, flags_header, flags_data, 
                              psf, x_center, y_center, oversample, perc , increase_ring, ax_plot, ax_plot2):
    r_aperture_test_image = np.linspace(0.1,5)
    r_aperture_test_psf = r_aperture_test_image*oversample

    ap_corr_array = []
    for r_aperture_i in r_aperture_test_psf:
        apcorr = get_aperture_corr(psf, x_center, y_center, r_aperture_i)
        ap_corr_array.append(apcorr)
    ap_corr_array = np.array(ap_corr_array).flatten()


    f_aperture = interp1d(1/ap_corr_array,r_aperture_test_image)
    r_aperture_perc = f_aperture(perc)
    r_aperture_perc = np.float64(r_aperture_perc)

    r_in_perc = r_aperture_perc*8*increase_ring
    r_out_perc = r_aperture_perc*10*increase_ring

    ax_plot.plot(r_aperture_test_image, ap_corr_array, '.-', color='k')
    ax_plot.axvline(x=r_aperture_perc,color='k',linewidth=1,linestyle='--')
    ax_plot.set_xlabel('raperture')
    ax_plot.set_ylabel('apcorr')
    ax_plot.set_yscale('log')


    flux_array = []
    for r_aperture_i in r_aperture_test_image:
        aper_flux,_,total_bkg,_ =  calculate_flux_and_bkg(x_rel, y_rel, image_data, image_data_err, flags_header, flags_data, r_aperture_i, r_in_perc, r_out_perc)
        apersum_bkgsub = aper_flux - total_bkg
        flux_array.append(apersum_bkgsub)
    flux_array = np.array(flux_array).flatten()
        
    ax_plot2.plot(r_aperture_test_image, flux_array, '.-', color='k', label=f'rin={r_in_perc}, rout={r_out_perc}')
    ax_plot2.axvline(x=r_aperture_perc,color='k',linewidth=1,linestyle='--')
    ax_plot2.set_xlabel('raperture')
    ax_plot2.set_ylabel('flux')

    return r_aperture_perc, r_in_perc, r_out_perc

def study_different_rings_image(x_rel, y_rel, image_data, image_data_err, flags_header, flags_data, 
                                r_aperture, r_in, r_out, test_plot, ax_plot=''):

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
            aper_flux,_,total_bkg,_ =  calculate_flux_and_bkg(x_rel, y_rel, image_data, image_data_err, flags_header, flags_data, r_aperture_i, r_in_i, r_out_i)
            apersum_bkgsub = aper_flux - total_bkg
            flux_array.append(apersum_bkgsub)
        flux_array = np.array(flux_array).flatten()
        if r_in_i == r_in:
            c_i = 'r'
        else:
            c_i = color_i
        axs_test[0].plot(r_aperture_test, flux_array, '.-', color=c_i, label=f'rin={r_in_i}, rout={r_out_i}')
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
            aper_flux,_,total_bkg,_ =  calculate_flux_and_bkg(x_rel, y_rel, image_data, image_data_err, flags_header, flags_data, r_aperture_i, r_in, r_out_i)
            apersum_bkgsub = aper_flux - total_bkg
            flux_array.append(apersum_bkgsub)
        flux_array = np.array(flux_array).flatten()
        if r_out_i == r_out:
            c_i = 'r'
            ax_plot.plot(r_aperture_test, flux_array, '.-', color='k', label=f'rin={r_in}, rout={r_out}')
            ax_plot.axvline(x=r_aperture,color='k',linewidth=1,linestyle='--')
            ax_plot.set_xlabel('raperture')
            ax_plot.set_ylabel('flux')
        else:
            c_i = color_i
        axs_test[1].plot(r_aperture_test, flux_array, '.-', color=c_i, label=f'rin={r_in}, rout={r_out_i}')
        axs_test[1].axvline(x=r_aperture,color='k',linewidth=1,linestyle='--')
    axs_test[1].set_xlabel('raperture')
    axs_test[1].set_ylabel('flux')
    axs_test[1].set_title('Changing rout')
    axs_test[1].legend(ncols=3)
    f_test.savefig(test_plot,dpi=300)

