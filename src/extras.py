from astropy.table import Table
import numpy as np
import astropy.units as u

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
    path_standard = path_data + 'spectra/Spherex_spec_'+name_standard+'.csv'

    return path_standard,find_standard

def get_not_good_young_standard():

    the_ones_that_dont_look_good = ['Gaia DR2 6273537856003401728','V* DH Tau B','2MASS J07410404-0359495','2MASS J16212830-2529558',
    'Cl* Melotte   25   CFHT      19','2MASS J02292794-0053282','2MASS J02292794-0053282','2MASS J20135152-2806020','HD  44627B',
    'ROXs 12B','LP  944-20','2MASS J15165651-2640251','2MASS J06272161-5308428','2MASS J11271382-3735076','2MASS J00464841+0715177',
    'PSO J318.4243+35.1277','2MASS J09175035+2944455','Cl* Melotte   22   PLIZ      35','2MASS J19350976-6200473','WISE J052857.68+090104.4',
    '* bet Pic b','Cl* Melotte   22    BRB      17','2MASS J04373705+2331080','2MASS J15575011-2952431','CD-52   381B',
    'PSO J078.9904+31.0171','2MASS J07123786-6155528','HD 106906B','LSPM J1108+6830','2MASS J13595510-4034582','2MASS J05082480+2422518',
    'DENIS J065219.7-253450','2MASS J02495436-0558015','2MASS J04144158+2809583','2MASS J20544332-2745512','2MASS J20113196-5048112',
    '2MASS J01531463-6744181','CD-35  2722B','G 196-3B','2MASS J14252798-3650229','2MASS J12563961-2718455','2MASS J15382417-1953116',
    'LEHPM 1481B','RX J1609.5-2105B','Cl* Melotte   22    BRB      23','NAME 2MASS J02192210-3925225b','2MASS J09512690-8023553',
    '2MASSI J0103320+193536','2MASS J08132232-1522043','HD 116434C','SIPS J1256-1257B','NAME AF Lep b','2MASS J00470038+6803543',
    '2MASS J17410280-4642218','HD 203030B','CWISE J050626.96+073842.4','WISE J124332.17+600126.6','*  51 Eri b','PSO J039.6352-21.7746',
    '* kap And B','2MASS J23224684-3133231']

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

