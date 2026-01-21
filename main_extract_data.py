from astroquery.ipac.irsa import Irsa
import matplotlib.pyplot as plt
import numpy as np
from astropy.table import Table
import os
import sys
sys.path.append('../')
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
#known_data = Table.read('../Data/standards/standards.csv')
#path_figures = '../Figures/standards/'
#path_data = '../Data/standards/'

### Standards young L/T dwarfs
known_data = Table.read('../Data/tables/youngstandards.csv')
path_figures = '../Figures/'
path_data = '../Data/spectra/'

### Disk Claire
#known_data = Table.read('../Data/disk_claire/disk_claire.csv')
#mask = np.array([str(x)!='-' and str(x)!='--' for x in known_data['pmra']])
#known_data = known_data[mask]
#ra,dec = [],[]
#for ra_i,dec_i in zip(known_data['ra'],known_data['dec']):
#    c = SkyCoord(ra_i+' '+dec_i, unit=(u.hourangle, u.deg))
#    ra.append(c.ra.deg)
#    dec.append(c.dec.deg)
#known_data['ra'] = ra
#known_data['dec'] = dec
#for column in ['pmra','pmdec','epoch']:
#    known_data[column] = np.array([float(x) for x in known_data[column]])

### Young LT dwarfs
#youngLTdwarfs = Table.read('../Data/youngLTdwarfs/youngLTdwarfs.csv')
#youngLTdwarfs = youngLTdwarfs[np.array(youngLTdwarfs['sptnum_ir_formula']) >= 10]
#known_data = Table()
#known_data['Name'] = youngLTdwarfs['name']
#known_data['ra'] = youngLTdwarfs['ra_j2000_P1']
#known_data['dec'] = youngLTdwarfs['dec_j2000_P1']
#known_data['pmra'] = youngLTdwarfs['pmra_formula']
#known_data['pmdec'] = youngLTdwarfs['pmdec_formula']
#known_data['epoch'] = np.ones(len(youngLTdwarfs))*2000
#path_figures = '../Figures/youngLTdwarfs/'
#path_data = '../Data/youngLTdwarfs/'

###subdwarfs
#subdwarfs = Table.read('../Data/subdwarfs/subdwarfs.csv')
#subdwarfs = subdwarfs[np.array(subdwarfs['sptnum_ir_formula']) < -10]
#known_data = Table()
#known_data['Name'] = subdwarfs['name']
#known_data['ra'] = subdwarfs['ra_j2000_WISE']
#known_data['dec'] = subdwarfs['dec_j2000_WISE']
#known_data['pmra'] = subdwarfs['pmra_formula']
#known_data['pmdec'] = subdwarfs['pmdec_formula']
#known_data['epoch'] = np.ones(len(subdwarfs))*2000
#path_figures = '../Figures/subdwarfs/'
#path_data = '../Data/subdwarfs/'



perc = 0.9 # Percentage of flux to choose r_aperture

for i_known_data in range(len(known_data)):
    name = known_data['Name'][i_known_data]
    ra_deg = known_data['ra'][i_known_data]
    dec_deg = known_data['dec'][i_known_data]
    pmra = known_data['pmra'][i_known_data]
    pmdec = known_data['pmdec'][i_known_data]
    epoch = known_data['epoch'][i_known_data]

    if not os.path.exists(path_data+'Spherex_spec_'+name+'.csv'):
        print(f'{i_known_data} out of {len(known_data)}')

        coord = SkyCoord(ra_deg, dec_deg, unit='deg')
        search_radius = 1 * u.arcsec
        print(name)

        results_spec = Table()

        # Following tutorial https://caltech-ipac.github.io/irsa-tutorials/spherex-intro/
        Irsa.list_collections(filter='spherex')

        results = Irsa.query_sia(pos=(coord, search_radius), collection='spherex_qr2')

        if not os.path.exists(path_figures + name+'/'):
            os.makedirs(path_figures + name+'/')

        test_external_ring = False

        results_spec = extrac_spectrum(name,results,results_spec,ra_deg,dec_deg,pmra,pmdec,epoch,perc,
                                       test_external_ring,known_data,i_known_data,path_figures,path_data)#,increase_ring=5)
        results_spec.sort('Wave [um] '+str(perc))
        results_spec.write(path_data+'Spherex_spec_'+name+'.csv',format='csv',overwrite=True)
        