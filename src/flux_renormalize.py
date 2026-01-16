## Code based on flux_calib.py by Genaro Suarez

import numpy as np
from astropy.io import ascii

def get_zero_point(fil_phot_synt_i):
	# from Jy to mag
    if (fil_phot_synt_i == 'J'): 
        phot_F0, ephot_F0 = 1544.03, 27.8 # Jy
    elif (fil_phot_synt_i == 'H'): 
        phot_F0, ephot_F0 = 1034.80, 20.0 # Jy
    elif (fil_phot_synt_i == 'K'): 
        phot_F0, ephot_F0 = 638.19, 12.6 # Jy
    elif (fil_phot_synt_i == 'F1000W'): 
        phot_F0, ephot_F0 = 38.51, 38.51/50 # Jy
    elif (fil_phot_synt_i == 'F1280W'): 
        phot_F0, ephot_F0 = 23.52, 23.52/50 # Jy
    elif (fil_phot_synt_i == 'F1800W'): 
        phot_F0, ephot_F0 = 11.99, 11.99/50 # Jy
    elif (fil_phot_synt_i == 'I1'): 
        phot_F0, ephot_F0 = 274.53, 274.53/50 # Jy
    elif (fil_phot_synt_i == 'I2'): 
        phot_F0, ephot_F0 = 177.66, 177.66/50 # Jy
    elif (fil_phot_synt_i == 'W1'): 
        phot_F0, ephot_F0 = 303.56, 303.56/50 # Jy
    elif (fil_phot_synt_i == 'W2'): 
        phot_F0, ephot_F0 = 168.46, 168.46/50 # Jy
    elif (fil_phot_synt_i == 'W3'): 
        phot_F0, ephot_F0 = 31.07, 31.07/50 # Jy
    elif (fil_phot_synt_i == 'Y'): 
        phot_F0, ephot_F0 = 2070.63, 2070.63/50 # Jy
    else:
    	print('Zero point for ' + fil_phot_synt_i + ' not included.')

    return phot_F0, ephot_F0


def get_filter_response(fil_phot_synt_i, filter_path = '/Users/rociokiman/Dropbox (Hunter College)/papers/Coconuts2/Data/Filters/'):

	if (fil_phot_synt_i == 'J'): 
		filter_response = ascii.read(filter_path + 'MKO_NSFCam.J.dat')
	elif (fil_phot_synt_i == 'H'): 
		filter_response = ascii.read(filter_path + 'MKO_NSFCam.H.dat')
	elif (fil_phot_synt_i == 'K'): 
		filter_response = ascii.read(filter_path + 'MKO_NSFCam.K.dat')
	elif (fil_phot_synt_i == 'F1000W'): 
		filter_response = ascii.read(filter_path + 'JWST_MIRI.F1000W.dat')
	elif (fil_phot_synt_i == 'F1280W'): 
		filter_response = ascii.read(filter_path + 'JWST_MIRI.F1280W.dat')
	elif (fil_phot_synt_i == 'F1800W'): 
		filter_response = ascii.read(filter_path + 'JWST_MIRI.F1800W.dat')
	elif (fil_phot_synt_i == 'I1'):
	    filter_response = ascii.read(filter_path + 'Spitzer_IRAC.I1.dat')
	elif (fil_phot_synt_i == 'I2'):
	    filter_response = ascii.read(filter_path + 'Spitzer_IRAC.I2.dat')
	elif (fil_phot_synt_i == 'W1'):
	    filter_response = ascii.read(filter_path + 'WISE_WISE.W1.dat')
	elif (fil_phot_synt_i == 'W2'):
	    filter_response = ascii.read(filter_path + 'WISE_WISE.W2.dat')
	elif (fil_phot_synt_i == 'W3'):
	    filter_response = ascii.read(filter_path + 'WISE_WISE.W3.dat')
	elif (fil_phot_synt_i == 'Y'):
	    filter_response = ascii.read(filter_path + 'Gemini_Flamingos2.Y.dat')
	else:
		print("Filter " + fil_phot_synt_i + " not included")

	return filter_response

def calculate_synt_phot(fil_phot_synt, wl_spectrum, flux_spectrum, eflux_spectrum, 
	                    filter_path = '/Users/rociokiman/Dropbox (Hunter College)/papers/Coconuts2/Data/Filters/'):
	'''
	Filters available: 'J', 'H', 'K', 'F1000W', 'F1280W', 'F1800W'

	INPUT:
	fil_phot_synt: list of strings. Filter to obtain the synthetic photometry (NOT FULL COVERAGE OF THE J FILTER)

	OUTPUT:
	phot_synt_mag: array, synthetic magnitude 
	ephot_synt_mag: array, error synthetic magnitude 
	phot_synt: array, synthetic flux in erg/s/cm2/A
	ephot_synt: array, synthetic error flux in erg/s/cm2/A
	
	'''

	# synthetic photometry
	phot_synt = np.zeros(len(fil_phot_synt)) # array to save the synthetic photometry in erg/s/cm2/A
	ephot_synt = np.zeros(len(fil_phot_synt)) # array to save the errors of the synthetic photometry in erg/s/cm2/A
	phot_synt_mag = np.zeros(len(fil_phot_synt)) # array to save the synthetic photometry in mag
	ephot_synt_mag = np.zeros(len(fil_phot_synt)) # array to save the synthetic photometry in mag
	lambda_eff = np.zeros(len(fil_phot_synt)) # to save the effective wavelength of each filter
	width_eff = np.zeros(len(fil_phot_synt)) # to save the effective width of each filter

	for i in range(len(fil_phot_synt)):
		try:

		    filter_response = get_filter_response(fil_phot_synt[i], filter_path)
		    
		    filter_response_wl = filter_response['col1'] / 1e4 # in um
		    filter_response_flux = filter_response['col2']
		    
		    # wavelength dispersion of the spectrum in the wavelength range of the filter
		    ind_spectrum_filter_response = np.where((wl_spectrum>=filter_response_wl.min()) & (wl_spectrum<=filter_response_wl.max()))[0]
		    wl_spectrum_filter_response = wl_spectrum[ind_spectrum_filter_response] # wavelength of the spectrum in the filter passband
		    flux_spectrum_filter_response = flux_spectrum[ind_spectrum_filter_response] # fluxes of the spectrum in the filter passband
		    eflux_spectrum_filter_response = eflux_spectrum[ind_spectrum_filter_response] # flux errors of the spectrum in the filter passband
		    

		    # dispersion of spectrum spectra (um)
		    # This is the delta wavelegth to make the integral. We just calculate the delta at each step.
		    wl_dis_spectrum_filter_response = wl_spectrum_filter_response[1:] - wl_spectrum_filter_response[:-1] 
		    # add an element equal to the last to keep the same shape as the wl_spectrum array
		    wl_dis_spectrum_filter_response = np.insert(wl_dis_spectrum_filter_response, 
		                                        wl_dis_spectrum_filter_response.size, 
		                                        wl_dis_spectrum_filter_response[-1]) 
		    
		    # synthetic photometry for each filter
		    # interpolate the filter response to the wavelength points of the spectrum
		    #flux_spectrum_filter_response = np.interp(wl_spectrum_filter_response, filter_response_wl, filter_response_flux) 
		    # filter response for the wavelength resolution of the spectrum spectra (erg/s/cm2/A)
		    filter_response_flux_int = np.interp(wl_spectrum_filter_response, filter_response_wl, filter_response_flux) # filter response for the wavelength resolution of the spectrum spectra (erg/s/cm2/A)
		    
		    # normalize the response curve (it was dimensionless but now it has 1/um units)
		    filter_response_flux_int_norm = filter_response_flux_int / sum(filter_response_flux_int*wl_dis_spectrum_filter_response) # 1/um
		    
		    # synthetic flux density
		    phot_synt[i] = sum(flux_spectrum_filter_response*filter_response_flux_int_norm*wl_dis_spectrum_filter_response) # erg/s/cm2/A
		    ephot_synt[i] = np.median(eflux_spectrum_filter_response/flux_spectrum_filter_response) * phot_synt[i] # keep median fractional flux uncertainties in the filter passband
		    
		    # compute the effective wavelength and effective width of each filter
		    lambda_eff[i] = sum(wl_spectrum_filter_response*filter_response_flux_int*flux_spectrum_filter_response*wl_dis_spectrum_filter_response) / sum(filter_response_flux_int*flux_spectrum_filter_response*wl_dis_spectrum_filter_response) # um
		    width_eff[i] = sum(filter_response_flux_int*wl_dis_spectrum_filter_response) / filter_response_flux_int.max() # um
		    
		    # convert flux (in erg/s/cm2/A) to magnitudes
		    # first from erg/s/cm2/A to Jy
		    #phot_synt_Jy = phot_synt[i]*(lambda_eff[i]*1e-4)**2 / 3e-21 # Jy
		    phot_synt_Jy = 3.33564095e+04*phot_synt[i]*(lambda_eff[i]*1e4)**2 # Jy (the above option gives the same result
		    ephot_synt_Jy = phot_synt_Jy * ephot_synt[i]/phot_synt[i] # Jy
		    
		    phot_F0, ephot_F0 = get_zero_point(fil_phot_synt[i])

		    phot_synt_mag[i] = -2.5*np.log10(phot_synt_Jy/phot_F0) # in mag
		    ephot_synt_mag[i] = (2.5/np.log(10))*np.sqrt((ephot_synt_Jy/phot_synt_Jy)**2+(ephot_F0/phot_F0)**2) # in mag
		except:
			phot_synt[i] = np.nan
			ephot_synt[i] = np.nan
			lambda_eff[i] = np.nan
			width_eff[i] = np.nan
			phot_synt_mag[i] = np.nan
			ephot_synt_mag[i] = np.nan
	    
	return lambda_eff, width_eff, phot_synt_mag, ephot_synt_mag, phot_synt, ephot_synt


def renormalize_spectrum(fil_phot_synt, phot_measurement, phot_err_measurement, wl_spectrum, flux_spectrum, eflux_spectrum,
	                     filter_path = '/Users/rociokiman/Dropbox (Hunter College)/papers/Coconuts2/Data/Filters/'):
	'''
	Filters available: 'J', 'H', 'K', 'F1000W', 'F1280W', 'F1800W'

	INPUT:
	fil_phot_synt: list of strings. Filter to obtain the synthetic photometry (NOT FULL COVERAGE OF THE J FILTER)
	phot_measurement: list of photometry measurements to use to renormilize
	phot_err_measurement: list of photometry error measurements to use to renormilize

	OUTPUT:
	
	'''

	twoMASS_erg = np.zeros(len(fil_phot_synt)) # to store the 2MASS photometry in erg/s/cm2/A
	etwoMASS_erg = np.zeros(len(fil_phot_synt)) # to store the 2MASS photometry in erg/s/cm2/A

	res = np.zeros(len(fil_phot_synt))
	eres = np.zeros(len(fil_phot_synt))

	lambda_eff, width_eff, phot_synt_mag, ephot_synt_mag, phot_synt, ephot_synt = calculate_synt_phot(fil_phot_synt, wl_spectrum, flux_spectrum, eflux_spectrum, filter_path)
	
	for i in range(len(fil_phot_synt)):

		phot_F0, ephot_F0 = get_zero_point(fil_phot_synt[i])

		# 2MASS photometry in erg/s/cm2/A
		# first from mag to Jy
		twoMASS_Jy = phot_F0 * 10**(phot_measurement[i]/(-2.5))
		etwoMASS_Jy = 10**(-phot_measurement[i]//2.5)*np.sqrt(ephot_F0**2+((-1*np.log(10)/2.5)*phot_F0*phot_err_measurement[i])**2)

		# from Jy to erg/s/cm2/A
		twoMASS_erg[i] = twoMASS_Jy / (3.33564095e+04*(lambda_eff[i]*1e4)**2)
		etwoMASS_erg[i] = twoMASS_erg[i] * etwoMASS_Jy/twoMASS_Jy # in erg/s/cm2/A

		# residual between the observed magnitudes and the synthetic ones
		res[i] = phot_measurement[i] - phot_synt_mag[i]
		eres[i] = np.sqrt(phot_err_measurement[i]**2 + ephot_synt_mag[i]**2)
	    

	# mean residuals
	res_JHK = np.mean((res))

	# scaling that must be applied to the spectrum
	scaling = 10**(-res_JHK/2.5)

	print('scaling applied to the spectrum to be in erg/s/cm2/A: '+str(scaling)) 

	return scaling, lambda_eff, width_eff, phot_synt_mag, ephot_synt_mag, phot_synt, ephot_synt, twoMASS_erg, etwoMASS_erg

