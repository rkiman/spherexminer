## Code based on flux_calib.py by Genaro Suarez
## See https://seda.readthedocs.io/en/latest/

import numpy as np
from astropy.io import ascii

def get_zero_point(fil_phot_synt_i):
	# from Jy to mag
	if (fil_phot_synt_i == 'J_MKO'): 
		phot_F0, ephot_F0 = 1544.03, 27.8 # Jy
		wave0 = 12417.04
	elif (fil_phot_synt_i == 'H_MKO'): 
		phot_F0, ephot_F0 = 1034.80, 20.0 # Jy
		wave0 = 16140.31
	elif (fil_phot_synt_i == 'K_MKO'): 
		phot_F0, ephot_F0 = 638.19, 12.6 # Jy
		wave0 = 21840.23
	elif (fil_phot_synt_i == 'J_2MASS'): 
		phot_F0, ephot_F0 = 1594, 1594/50 # Jy
		wave0 = 12350
	elif (fil_phot_synt_i == 'H_2MASS'): 
		phot_F0, ephot_F0 = 1024, 1024/50 # Jy
		wave0 = 16620
	elif (fil_phot_synt_i == 'K_2MASS'): 
		phot_F0, ephot_F0 = 666.8, 666.8/50 # Jy
		wave0 = 21590
	elif (fil_phot_synt_i == 'F1000W'): 
		phot_F0, ephot_F0 = 38.51, 38.51/50 # Jy
		wave0 = 98793.45
	elif (fil_phot_synt_i == 'F1280W'): 
		phot_F0, ephot_F0 = 23.52, 23.52/50 # Jy
		wave0 = 127059.68
	elif (fil_phot_synt_i == 'F1800W'): 
		phot_F0, ephot_F0 = 11.99, 11.99/50 # Jy
		wave0 = 178734.17
	elif (fil_phot_synt_i == 'I1'): 
		phot_F0, ephot_F0 = 274.53, 274.53/50 # Jy
		wave0 = 35074.83
	elif (fil_phot_synt_i == 'I2'): 
		phot_F0, ephot_F0 = 177.66, 177.66/50 # Jy
		wave0 = 44365.56
	elif (fil_phot_synt_i == 'W1'): 
		phot_F0, ephot_F0 = 309.54, 309.54/50 # Jy
		wave0 = 33526
	elif (fil_phot_synt_i == 'W2'): 
		phot_F0, ephot_F0 = 171.787, 171.787/50 # Jy
		wave0 = 46028
	elif (fil_phot_synt_i == 'W3'): 
		phot_F0, ephot_F0 = 31.674, 31.674/50 # Jy
		wave0 = 115608
	elif (fil_phot_synt_i == 'Y'): 
		phot_F0, ephot_F0 = 2070.63, 2070.63/50 # Jy
		wave0 = 10182.69
	elif (fil_phot_synt_i == 'F460M'): 
		phot_F0, ephot_F0 = 164.11, 164.11/50 # Jy
		wave0 = 46269.86
	elif (fil_phot_synt_i == 'F444W'): 
		phot_F0, ephot_F0 = 184.10, 184.10/50 # Jy
		wave0 = 43504.26
	elif (fil_phot_synt_i == 'F430M'): 
		phot_F0, ephot_F0 = 190.70, 190.70/50 # Jy
		wave0 = 42784.79
	elif (fil_phot_synt_i == 'F410M'): 
		phot_F0, ephot_F0 = 208.75, 208.75/50 # Jy
		wave0 = 40723.18
	elif (fil_phot_synt_i == 'F360M'): 
		phot_F0, ephot_F0 = 260.31, 260.31/50 # Jy
		wave0 = 36148.93
	elif (fil_phot_synt_i == 'F356W'): 
		phot_F0, ephot_F0 = 271.39, 271.39/50 # Jy
		wave0 = 35287.04
	elif (fil_phot_synt_i == 'F335M'): 
		phot_F0, ephot_F0 = 298.91, 298.91/50 # Jy
		wave0 = 33537.23
	elif (fil_phot_synt_i == 'F300M'): 
		phot_F0, ephot_F0 = 369.65, 369.65/50 # Jy
		wave0 = 29818.32
	elif (fil_phot_synt_i == 'F277W'): 
		phot_F0, ephot_F0 = 430.14, 430.14/50 # Jy
		wave0 = 27278.58
	elif (fil_phot_synt_i == 'F250M'): 
		phot_F0, ephot_F0 = 504.65, 504.65/50 # Jy
		wave0 = 25005.80
	elif (fil_phot_synt_i == 'F210M'): 
		phot_F0, ephot_F0 = 688.03, 688.03/50 # Jy
		wave0 = 20908.35
	elif (fil_phot_synt_i == 'F200W'): 
		phot_F0, ephot_F0 = 757.65, 757.65/50 # Jy
		wave0 = 19680.41
	elif (fil_phot_synt_i == 'F182M'): 
		phot_F0, ephot_F0 = 844.94, 844.94/50 # Jy
		wave0 = 18388.83
	elif (fil_phot_synt_i == 'F162M'): 
		phot_F0, ephot_F0 = 1023.04, 1023.04/50 # Jy
		wave0 = 16243.33
	elif (fil_phot_synt_i == 'F150W'): 
		phot_F0, ephot_F0 = 1172.06, 1172.06/50 # Jy
		wave0 = 14872.56
	elif (fil_phot_synt_i == 'F140M'): 
		phot_F0, ephot_F0 = 1288.65, 1288.65/50 # Jy
		wave0 = 14023.62	
	elif (fil_phot_synt_i == 'F115W'): 
		phot_F0, ephot_F0 = 1746.12, 1746.12/50 # Jy
		wave0 = 11433.62
	elif (fil_phot_synt_i == 'Lp_MKO'): 
		phot_F0, ephot_F0 = 245.20, 245.20/50 # Jy
		wave0 = 37302.87
	elif (fil_phot_synt_i == 'Mp_MKO'): 
		phot_F0, ephot_F0 = 160.62, 160.62/50 # Jy
		wave0 = 46803.78
	else:
		print('Zero point for ' + fil_phot_synt_i + ' not included.')
	
	return wave0, phot_F0, ephot_F0


def get_filter_response(fil_phot_synt_i, filter_path):

	if (fil_phot_synt_i == 'J_MKO'): 
		filter_response = ascii.read(filter_path + 'MKO_NSFCam.J.dat')
	elif (fil_phot_synt_i == 'H_MKO'): 
		filter_response = ascii.read(filter_path + 'MKO_NSFCam.H.dat')
	elif (fil_phot_synt_i == 'K_MKO'): 
		filter_response = ascii.read(filter_path + 'MKO_NSFCam.K.dat')
	elif (fil_phot_synt_i == 'J_2MASS'): 
		filter_response = ascii.read(filter_path + '2MASS_2MASS.J.dat')
	elif (fil_phot_synt_i == 'H_2MASS'): 
		filter_response = ascii.read(filter_path + '2MASS_2MASS.H.dat')
	elif (fil_phot_synt_i == 'K_2MASS'): 
		filter_response = ascii.read(filter_path + '2MASS_2MASS.Ks.dat')
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
	elif (fil_phot_synt_i == 'F460M'): 
		filter_response = ascii.read(filter_path + 'JWST_NIRCam.F460M.dat')
	elif (fil_phot_synt_i == 'F444W'): 
		filter_response = ascii.read(filter_path + 'JWST_NIRCam.F444W.dat')
	elif (fil_phot_synt_i == 'F430M'): 
		filter_response = ascii.read(filter_path + 'JWST_NIRCam.F430M.dat')
	elif (fil_phot_synt_i == 'F410M'): 
		filter_response = ascii.read(filter_path + 'JWST_NIRCam.F410M.dat')
	elif (fil_phot_synt_i == 'F360M'): 
		filter_response = ascii.read(filter_path + 'JWST_NIRCam.F360M.dat')
	elif (fil_phot_synt_i == 'F356W'): 
		filter_response = ascii.read(filter_path + 'JWST_NIRCam.F356W.dat')
	elif (fil_phot_synt_i == 'F335M'): 
		filter_response = ascii.read(filter_path + 'JWST_NIRCam.F335M.dat')
	elif (fil_phot_synt_i == 'F300M'): 
		filter_response = ascii.read(filter_path + 'JWST_NIRCam.F300M.dat')
	elif (fil_phot_synt_i == 'F277W'): 
		filter_response = ascii.read(filter_path + 'JWST_NIRCam.F277W.dat')
	elif (fil_phot_synt_i == 'F250M'):
		filter_response = ascii.read(filter_path + 'JWST_NIRCam.F250M.dat')
	elif (fil_phot_synt_i == 'F210M'):
		filter_response = ascii.read(filter_path + 'JWST_NIRCam.F210M.dat')
	elif (fil_phot_synt_i == 'F200W'):
		filter_response = ascii.read(filter_path + 'JWST_NIRCam.F200W.dat')
	elif (fil_phot_synt_i == 'F182M'):
		filter_response = ascii.read(filter_path + 'JWST_NIRCam.F182M.dat')
	elif (fil_phot_synt_i == 'F162M'): 
		filter_response = ascii.read(filter_path + 'JWST_NIRCam.F162M.dat')
	elif (fil_phot_synt_i == 'F150W'): 
		filter_response = ascii.read(filter_path + 'JWST_NIRCam.F150W.dat')
	elif (fil_phot_synt_i == 'F140M'): 
		filter_response = ascii.read(filter_path + 'JWST_NIRCam.F140M.dat')
	elif (fil_phot_synt_i == 'F115W'): 
		filter_response = ascii.read(filter_path + 'JWST_NIRCam.F115W.dat')
	elif (fil_phot_synt_i == 'Lp_MKO'): 
		filter_response = ascii.read(filter_path + 'MKO_NSFCam.Lp.dat')
	elif (fil_phot_synt_i == 'Mp_MKO'): 
		filter_response = ascii.read(filter_path + 'MKO_NSFCam.Mp.dat')
	else:
		print("Filter " + fil_phot_synt_i + " not included")

	wave_col, flux_col = filter_response.colnames
	wave_first = filter_response[wave_col][0] - 0.001
	wave_last = filter_response[wave_col][-1] + 0.001

	filter_response.insert_row(0, [wave_first, 0])
	filter_response.add_row([wave_last, 0])

	return filter_response

def calculate_synt_phot(fil_phot_synt, wl_spectrum, flux_spectrum, eflux_spectrum, 
	                    filter_path):

	# synthetic photometry
	phot_synt = np.zeros(len(fil_phot_synt)) 
	ephot_synt = np.zeros(len(fil_phot_synt)) 
	phot_synt_mag = np.zeros(len(fil_phot_synt)) 
	ephot_synt_mag = np.zeros(len(fil_phot_synt)) 
	lambda_eff = np.zeros(len(fil_phot_synt)) 
	width_eff = np.zeros(len(fil_phot_synt)) 

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
		    # filter response for the wavelength resolution of the spectrum spectra (erg/s/cm2/A)
			filter_response_flux_int = np.interp(wl_spectrum_filter_response, filter_response_wl, filter_response_flux) # filter response for the wavelength resolution of the spectrum spectra (erg/s/cm2/A)
			
		    # normalize the response curve (it was dimensionless but now it has 1/um units)
			filter_response_flux_int_norm = (filter_response_flux_int * wl_spectrum_filter_response) / sum(filter_response_flux_int * wl_spectrum_filter_response * wl_dis_spectrum_filter_response) # 1/um
		
		    # synthetic flux density
			phot_synt[i] = sum(flux_spectrum_filter_response*filter_response_flux_int_norm*wl_dis_spectrum_filter_response) # erg/s/cm2/A
			ephot_synt[i] = np.sqrt(sum((eflux_spectrum_filter_response * filter_response_flux_int_norm * wl_dis_spectrum_filter_response)**2))
		    
		    # compute the effective wavelength and effective width of each filter
			lambda_eff[i] = sum(wl_spectrum_filter_response**2*filter_response_flux_int*flux_spectrum_filter_response*wl_dis_spectrum_filter_response) / sum(wl_spectrum_filter_response*filter_response_flux_int*flux_spectrum_filter_response*wl_dis_spectrum_filter_response) # um
			width_eff[i] = sum(filter_response_flux_int*wl_dis_spectrum_filter_response) / filter_response_flux_int.max() # um

		    # convert flux (in erg/s/cm2/A) to magnitudes
		    # first from erg/s/cm2/A to Jy
		    #phot_synt_Jy = phot_synt[i]*(lambda_eff[i]*1e-4)**2 / 3e-21 # Jy
			phot_synt_Jy = 3.33564095e+04*phot_synt[i]*(lambda_eff[i]*1e4)**2 # Jy (the above option gives the same result
			ephot_synt_Jy = phot_synt_Jy * ephot_synt[i]/phot_synt[i] # Jy
			
			_, phot_F0, ephot_F0 = get_zero_point(fil_phot_synt[i])
			
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


def renormalize_spectrum(fil_phot_synt, phot_measurement, phot_err_measurement, 
						 wl_spectrum, flux_spectrum, eflux_spectrum,
	                     filter_path):

	twoMASS_erg = np.zeros(len(fil_phot_synt)) # to store the 2MASS photometry in erg/s/cm2/A
	etwoMASS_erg = np.zeros(len(fil_phot_synt)) # to store the 2MASS photometry in erg/s/cm2/A

	res = np.zeros(len(fil_phot_synt))

	lambda_eff, width_eff, phot_synt_mag, ephot_synt_mag, phot_synt, ephot_synt = calculate_synt_phot(fil_phot_synt, wl_spectrum, flux_spectrum, eflux_spectrum, filter_path)
	
	for i in range(len(fil_phot_synt)):

		_, phot_F0, ephot_F0 = get_zero_point(fil_phot_synt[i])

		# 2MASS photometry in erg/s/cm2/A
		# first from mag to Jy
		twoMASS_Jy = phot_F0 * 10**(phot_measurement[i]/(-2.5))
		etwoMASS_Jy = 10**(-phot_measurement[i]/2.5)*np.sqrt(ephot_F0**2+((-1*np.log(10)/2.5)*phot_F0*phot_err_measurement[i])**2)

		# from Jy to erg/s/cm2/A
		twoMASS_erg[i] = twoMASS_Jy / (3.33564095e+04*(lambda_eff[i]*1e4)**2)
		etwoMASS_erg[i] = twoMASS_erg[i] * etwoMASS_Jy/twoMASS_Jy # in erg/s/cm2/A

		# residual between the observed magnitudes and the synthetic ones
		res[i] = phot_measurement[i] - phot_synt_mag[i]
	    

	# mean residuals
	res_JHK = np.mean((res))

	# scaling that must be applied to the spectrum
	scaling = 10**(-res_JHK/2.5)

	return scaling, lambda_eff, width_eff, phot_synt_mag, ephot_synt_mag, phot_synt, ephot_synt, twoMASS_erg, etwoMASS_erg

