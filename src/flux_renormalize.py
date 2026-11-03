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


# np.trapz was renamed np.trapezoid in numpy 2
_trapezoid = getattr(np, 'trapezoid', None) or getattr(np, 'trapz')


def load_vega_fnu(vega_path):
	"""
	Read a CALSPEC Vega spectrum (e.g. alpha_lyr_stis_011.fits; WAVELENGTH in A,
	FLUX in erg/s/cm2/A) and return wavelength (um) and F_nu (Jy).
	"""
	from astropy.table import Table
	vega = Table.read(vega_path)
	wl_A = np.asarray(vega['WAVELENGTH'], dtype=float)
	flam = np.asarray(vega['FLUX'], dtype=float)
	fnu_jy = 3.33564095e+04 * flam * wl_A**2
	return wl_A / 1e4, fnu_jy


def calculate_synt_phot_v2(fil_phot_synt, wl_spectrum, fnu_spectrum, efnu_spectrum,
	                       bandwidth, filter_path, min_coverage=0.9, vega=None,
	                       zero_point_err=True, n_grid=5000):
	"""
	Synthetic photometry from SPHEREx narrow-band measurements.

	Each SPHEREx measurement j is treated as a constant F_nu over a top-hat of
	width bandwidth_j centered on wl_j (not as a sample of a continuous
	spectrum). For a photon-counting detector with response T(lambda) the
	band-averaged flux density is

	    F_syn = sum_j F_nu,j A_j / sum_j A_j,   A_j = int_{dlambda_j} T(lambda)/lambda dlambda

	The whole calculation is done in F_nu, so no (source-dependent) effective
	wavelength is needed to convert F_lambda to F_nu.

	Parameters
	----------
	fil_phot_synt : list of filter names (see get_filter_response)
	wl_spectrum : wavelengths of the SPHEREx measurements (um)
	fnu_spectrum, efnu_spectrum : F_nu and its error (Jy)
	bandwidth : SPHEREx bandwidth of each measurement (um), e.g. the BANDWIDTH
	    column of the official channel table
	filter_path : directory with the filter response files
	min_coverage : minimum fraction of int T/lambda dlambda covered by the
	    SPHEREx intervals; otherwise the magnitude is NaN
	vega : None or (wl_um, fnu_jy) of a Vega spectrum (see load_vega_fnu). If
	    given, the zero point is the Vega flux integrated through the same
	    filter (self-consistent Vega system); otherwise get_zero_point is used.
	zero_point_err : include the zero-point uncertainty in the magnitude error
	    (only for the get_zero_point zero points)

	Returns
	-------
	lambda_pivot (um), coverage, phot_synt_mag, ephot_synt_mag,
	phot_synt_Jy, ephot_synt_Jy, phot_F0_Jy
	"""
	wl_spectrum = np.asarray(wl_spectrum, dtype=float)
	fnu_spectrum = np.asarray(fnu_spectrum, dtype=float)
	efnu_spectrum = np.asarray(efnu_spectrum, dtype=float)
	bandwidth = np.asarray(bandwidth, dtype=float) * np.ones_like(wl_spectrum)

	n = len(fil_phot_synt)
	lambda_pivot = np.ones(n) * np.nan
	coverage = np.ones(n) * np.nan
	phot_synt_mag = np.ones(n) * np.nan
	ephot_synt_mag = np.ones(n) * np.nan
	phot_synt_Jy = np.ones(n) * np.nan
	ephot_synt_Jy = np.ones(n) * np.nan
	phot_F0_Jy = np.ones(n) * np.nan

	good = np.isfinite(wl_spectrum) & np.isfinite(fnu_spectrum) & \
	       np.isfinite(efnu_spectrum) & np.isfinite(bandwidth) & (bandwidth > 0)
	wl, fnu, efnu, bw = wl_spectrum[good], fnu_spectrum[good], efnu_spectrum[good], bandwidth[good]
	lo, hi = wl - bw / 2, wl + bw / 2

	for i in range(n):
		try:
			filter_response = get_filter_response(fil_phot_synt[i], filter_path)
			wave_col, flux_col = filter_response.colnames
			f_wl = np.asarray(filter_response[wave_col], dtype=float) / 1e4 # um
			f_T = np.asarray(filter_response[flux_col], dtype=float)

			# Fine grid over the filter and cumulative integral of T/lambda
			grid = np.linspace(f_wl.min(), f_wl.max(), n_grid)
			T = np.clip(np.interp(grid, f_wl, f_T), 0, None)
			integrand = T / grid
			cum = np.concatenate([[0], np.cumsum(0.5 * (integrand[1:] + integrand[:-1]) * np.diff(grid))])
			norm = cum[-1]

			# Filter-only pivot wavelength (for reference)
			lambda_pivot[i] = np.sqrt(_trapezoid(T * grid, grid) / norm)

			# Weight of each SPHEREx interval
			A = np.interp(hi, grid, cum) - np.interp(lo, grid, cum)
			use = A > 0
			if use.sum() == 0:
				raise ValueError('No SPHEREx data in filter ' + fil_phot_synt[i])

			# Fraction of the filter covered by the union of the intervals
			covered = np.zeros(grid.size, dtype=bool)
			for a, b in zip(lo[use], hi[use]):
				covered |= (grid >= a) & (grid <= b)
			coverage[i] = _trapezoid(integrand * covered, grid) / norm

			phot_synt_Jy[i] = np.sum(A[use] * fnu[use]) / np.sum(A[use])
			ephot_synt_Jy[i] = np.sqrt(np.sum((A[use] * efnu[use])**2)) / np.sum(A[use])

			# Zero point
			if vega is not None:
				vega_fnu = np.interp(grid, vega[0], vega[1])
				phot_F0 = _trapezoid(vega_fnu * integrand, grid) / norm
				ephot_F0 = 0.0
			else:
				_, phot_F0, ephot_F0 = get_zero_point(fil_phot_synt[i])
				if not zero_point_err:
					ephot_F0 = 0.0
			phot_F0_Jy[i] = phot_F0

			if coverage[i] < min_coverage or phot_synt_Jy[i] <= 0:
				continue
			phot_synt_mag[i] = -2.5 * np.log10(phot_synt_Jy[i] / phot_F0)
			ephot_synt_mag[i] = (2.5 / np.log(10)) * np.sqrt((ephot_synt_Jy[i] / phot_synt_Jy[i])**2 + (ephot_F0 / phot_F0)**2)
		except Exception as e:
			print('calculate_synt_phot_v2: ' + str(fil_phot_synt[i]) + ': ' + str(e))

	return lambda_pivot, coverage, phot_synt_mag, ephot_synt_mag, phot_synt_Jy, ephot_synt_Jy, phot_F0_Jy


# Narrow-band (per-channel) synthetic photometry for v2 spectra, merged from spherex_photometry_v2.py.

def synthetic_photometry_v2(spectrum, fil_phot_synt, filter_path, min_coverage=0.9, vega=None,
                            zero_point_err=True, drop_nonpositive=True):
    """Synthetic magnitudes from a v2 SPHEREx spectrum (see check_spectra.clean_spectrum_v2)."""
    flux = np.asarray(spectrum['Flux'], dtype=float)
    mask = np.isfinite(flux)
    if drop_nonpositive:
        mask &= flux > 0
    _, coverage, mag, emag, _, _, _ = calculate_synt_phot_v2(
        fil_phot_synt, np.asarray(spectrum['Wave'], dtype=float)[mask], flux[mask] * 1e-6,
        np.asarray(spectrum['Fluxerr'], dtype=float)[mask] * 1e-6,
        np.asarray(spectrum['Bandwidth'], dtype=float)[mask], filter_path,
        min_coverage=min_coverage, vega=vega, zero_point_err=zero_point_err)
    return np.array(mag), np.array(emag), np.array(coverage)


def literature_photometry(known_data, i, fil_phot_synt):
    """Literature magnitudes and errors (columns <filter> and <filter>_err) of row i; masked -> NaN."""
    mag = np.array([float(np.ma.filled(known_data[f][i], np.nan)) for f in fil_phot_synt])
    err = np.array([float(np.ma.filled(known_data[f + '_err'][i], np.nan)) for f in fil_phot_synt])
    return mag, err


def plot_syn_minus_lit(data_phot, data_phot_err, syn_phot, syn_phot_err, fil_phot_synt,
                       xlim=(10, 18), ylim=(-0.4, 0.4), n_mc=10000, seed=42, title=None):
    """Synthetic minus literature photometry versus literature magnitude, one panel per filter (same
    figure as in standards_new / youngstandards)."""
    import matplotlib.pyplot as plt
    from matplotlib.ticker import AutoMinorLocator
    from astropy.stats import sigma_clipped_stats
    from astropy.table import Table

    rng = np.random.default_rng(seed)
    n = data_phot.shape[0]
    f, axs = plt.subplots(len(fil_phot_synt), 1, figsize=(6, len(fil_phot_synt) * 1.5), sharex=True)
    axs = np.atleast_1d(axs)
    rows = []
    for j, fil_phot_i in enumerate(fil_phot_synt):
        syn_mag_dist = rng.normal(syn_phot[:, j], syn_phot_err[:, j], (n_mc, n))
        lit_mag_dist = rng.normal(data_phot[:, j], data_phot_err[:, j], (n_mc, n))
        delta = syn_mag_dist - lit_mag_dist
        delta_j = np.nanmedian(delta, axis=0)
        delta_std_j = np.nanstd(delta, axis=0)

        good = np.isfinite(delta_j)
        if good.sum() > 0:
            mean, median_delta_all, std_delta_all = sigma_clipped_stats(delta_j[good], sigma=3.0, maxiters=5, cenfunc='median')
        else:
            median_delta_all, std_delta_all = np.nan, np.nan
        rows.append([fil_phot_i.replace('I1', 'ch1').replace('I2', 'ch2'), int(good.sum()), median_delta_all, std_delta_all])

        axs[j].fill_between(np.arange(xlim[0], xlim[1] + 1), median_delta_all - std_delta_all,
                            median_delta_all + std_delta_all, color='lightgray')
        label = fil_phot_i.replace('I1', 'ch1').replace('I2', 'ch2')
        axs[j].errorbar(data_phot[:, j], delta_j, yerr=delta_std_j, fmt='.', zorder=1,
                        label=label + ': ' + str(np.round(median_delta_all, 2)) + r'$\pm$' + str(np.round(std_delta_all, 2)))
        axs[j].legend(loc=3)
        axs[j].axhline(0, linestyle='--', color='k')
        axs[j].set_xlim(*xlim)
        axs[j].set_ylim(*ylim)
        axs[j].set_ylabel('Syn - Lit (mag)')
        axs[j].xaxis.set_minor_locator(AutoMinorLocator())
        axs[j].yaxis.set_minor_locator(AutoMinorLocator())
        axs[j].tick_params('both', direction='in', top=True, right=True)
        axs[j].tick_params('both', which='minor', direction='in', top=True, right=True)
    axs[-1].set_xlabel('Literature Magnitude (mag)')
    if title:
        axs[0].set_title(title)
    plt.tight_layout()

    stats = Table(rows=rows, names=['Filter', 'N', 'Offset', 'Scatter'])
    stats['Offset'].format = '.3f'
    stats['Scatter'].format = '.3f'
    return f, stats
