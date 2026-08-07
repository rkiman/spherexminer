from astroquery.ipac.irsa import Irsa
import matplotlib.pyplot as plt
import numpy as np
from astropy.table import Table
import os
import sys
from src import *
import astropy
from astropy.coordinates import SkyCoord
from astropy import units as u
import warnings
from astropy.wcs import WCS, FITSFixedWarning
import logging
logging.getLogger('astropy.wcs').setLevel(logging.ERROR)
from erfa import ErfaWarning
warnings.filterwarnings(
    "ignore",
    category=ErfaWarning,
    message='.*pmsafe.*distance overridden.*'
)


### Starndards L/T dwarfs
known_data = Table.read('Data/tables/standards.csv')
path_figures = 'Figures/'
path_data = 'Data/spectra/'

### Standards young L/T dwarfs
#known_data = Table.read('Data/tables/youngstandards.csv')
#path_figures = 'Figures/'
#path_data = 'Data/spectra/'


perc = [0.9,0.95,0.98]  # Percentage of flux to choose r_aperture


for i_known_data in range(1,len(known_data)):
    name = known_data['Name'][i_known_data].replace(' ','')
    ra_deg = known_data['ra'][i_known_data]
    dec_deg = known_data['dec'][i_known_data]
    pmra = known_data['pmra'][i_known_data]
    pmdec = known_data['pmdec'][i_known_data]
    epoch = known_data['epoch'][i_known_data]

    file_name = path_data+'Spherex_spec_'+name+'.fits' 

    if not os.path.exists(file_name):
        print(f'{i_known_data} out of {len(known_data)}')

        coord = SkyCoord(ra_deg, dec_deg, unit='deg')
        search_radius = 3 * u.arcsec
        print(name)

        # Following tutorial https://caltech-ipac.github.io/irsa-tutorials/spherex-intro/
        Irsa.list_collections(filter='spherex')

        results = Irsa.query_sia(pos=(coord, search_radius), collection='spherex_qr2')

        if not os.path.exists(path_figures + name+'/'):
            os.makedirs(path_figures + name+'/')

        results_spec = extract_spectrum(name,results,ra_deg,dec_deg,pmra,pmdec,epoch,perc,
                                       path_figures,path_data)
        results_spec.sort('Wave')
        results_spec.write(file_name,format='fits',overwrite=True)
        