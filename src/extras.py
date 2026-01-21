from astropy.table import Table
import numpy as np

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


def get_closest_youngstandard_spectrum(spt_object, path_data):
    standards = Table.read(path_data + 'tables/youngstandards.csv')
    standards_spt_num = standards['SpT']

    find_standard = np.argmin(abs(standards_spt_num-spt_object))
    name_standard = standards['Name'][find_standard]
    path_standard = path_data + 'spectra/Spherex_spec_'+name_standard+'.csv'

    return path_standard,find_standard

