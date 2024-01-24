from pandas import read_csv
import toml
from scipy.optimize import curve_fit as fit
import numpy as np
import ctypes

from correlations import *
from read_in_data import *

lib_path = "/home/matteo/Downloads/ac/build/"

class Dataset:
    
    # mode can be either 0 (no rescaling of params) or 1 (rescaling params / block spin)
    def add_data(self, folder, param, mode):
        fields_data = read_csv(folder + '/traces.csv')
        mass_data = read_csv(folder + '/data.csv')
        S_t = get_time_slices_from_timeslicefile(folder + "/slice.dat", field_axis=0, return_all=False)
        data_Sq_t = read_csv(folder + "/data.csv")
        Sq_t = data_Sq_t['corr'].to_numpy(np.dtype('f8')).reshape((-1, self.Nt))
        toml_params = self.get_toml_params(folder + '/input.toml')

        volume = (self.Nt, self.Nx)
        
        # magnetisation
        blob = ComputeStatistics(fields_data['sigma'])
        self.phi.append((blob.average, blob.sigmaF))
        avgxxx = blob.average
        avgxxxerr = blob.sigmaF
        # condensate
        blob = ComputeStatistics(fields_data['tr'])
        self.condensate.append((blob.average, blob.sigmaF))
        # susceptibility
        blob = ComputeStatistics((fields_data['sigma'] - avgxxx)**2)
        self.chi2.append((blob.average, blob.sigmaF))
        chi2 = blob.average
        chi2err = blob.sigmaF
        # renormalised bosonic mass
        val, err = get_ren_mass_right_via_timeslices(S_t, volume)
        #val, err = get_ren_mass_right_via_timeslices2(S_t, volume, chi2, chi2err)
        self.m_phi_r.append((val, err))
        # physical quark mass
        val, err = get_phys_quark_mass_via_timeslices(Sq_t, volume)
        self.m_q_phys.append((val, err))
        # parameter value
        p = toml_params[param[0]][param[1]]
        s = toml_params['physics']['cutFraction']
        if mode == 1:
            if param[1] == "mass":
                p /= s*s
        self.parameters.append(p)
       
    def clear_data(self):
        self.phi = [] # <phi>
        self.abs_phi = [] # <|phi|>
        self.condensate = [] # <psibar psi>
        self.chi2 = [] # susceptiblity or second connected moment
        self.m_phi_r = [] # renormalised mesons mass
        self.m_q_phys = [] # physical quark mass
        self.parameters = [] # x parameter
    
    def sort_data(self, data):
        arr = [(p, val) for p, val in zip(self.parameters, data)]
        sorted_arr = sorted(arr, key = lambda x: x[0])
        sorted_p = []
        sorted_val = []
        for x in sorted_arr:
            sorted_p.append(x[0])
            sorted_val.append(x[1])
        return sorted_p, sorted_val
    
    def get_toml_params(self, filename):
        params = toml.load(filename)
        return params
        
    def __init__(self, Nt, Nx):
        self.Nt = Nt
        self.Nx = Nx
        self.clear_data()
    


class ACreturn(ctypes.Structure):
	_fields_ = [ ("average", ctypes.c_double), ("sigmaF", ctypes.c_double), ("sigmaF_error", ctypes.c_double), ("ACtime", ctypes.c_double), ("ACtimeError", ctypes.c_double) ]


def ComputeStatistics(data):
	ft_lib = ctypes.cdll.LoadLibrary(lib_path + "libac.so")

	arg = (ctypes.c_double * len(data))(*data)
	avg = ft_lib.Wrapper
	avg.restype = ACreturn

	return avg(arg, len(data))

def expectedM(m0, g, sigma, pi):
    r2 = sigma**2 + pi[0]**2 + pi[1]**2 + pi[2]**2
    denom = 2*(g*sigma + m0 + 1)
    sqrroot = np.sqrt((g**2*r2 + 2*m0*(g*sigma + 1) + 2*g*sigma + m0**2 + 2)**2 - 4*(g*sigma+m0+1)**2)
    num = -sqrroot + g**2*r2 + 2*g*m0*sigma + 2*g*sigma + m0**2 + 2*m0 + 2
    return -np.log(num/denom)

def fitfuncSinh(x, m_re, A):
    return A * np.sinh(m_re*(Nt/2-x))

def fitToSinh(ydata, startidx, endidx, plot=False):
    yvals = ydata[startidx:endidx]
    xvals = np.array(range(startidx, endidx))

    fitparams = fit(fitfuncSinh, xvals, yvals)
        
    return fitparams[0]

def get_phys_quark_mass_via_timeslices(Sq_t, volume):
    global Nt 
    Nt = int(volume[0])
    corr = np.average(Sq_t, axis=0)
    try:
    	val, err = fitToSinh(corr, 1, Nt, plot=False)
    except:
    	val = 0
    	err = 0
    del Nt
    return [val, err]


def get_ren_mass_right_via_timeslices2(S_t, volume, chi2, chi_err):

    Nt = volume[0]
    Nx = volume[1]
    
    a = np.arange(0, Nt)
    a[a>int(Nt/2)] = Nt-a[a>int(Nt/2)]


    connected_corr_t, connected_corr_t_err = get_connected_2pt_fct(S_t)
    #chi2 = Nx * np.sum( connected_corr_t)
    mu2 = 2 * Nx * np.sum( connected_corr_t * a**2)
    
    ren_mass2 = 2*2 * chi2/mu2
    ren_mass = np.sqrt( ren_mass2 )
        
    #chi_err = Nx * np.sqrt( np.sum( connected_corr_t_err**2 ))
    mu_err = 2*Nx * np.sqrt( np.sum( (a**2 * connected_corr_t_err)**2  )   )
    mass2_err = ren_mass2 * np.sqrt( (chi_err/chi2)**2 + (mu_err/mu2)**2 )
    mass_err = ren_mass *0.5 *mass2_err/ren_mass2

    return ren_mass, mass_err
