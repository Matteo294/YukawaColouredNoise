#Evaluation functions

import h5py
import numpy as np
import matplotlib.pyplot as plt
import os, fnmatch
import scipy
from tqdm import tqdm
import multiprocessing as mp
import subprocess

from scipy.optimize import curve_fit

from scipy.odr import *


def get_mag(cfgs: np.ndarray):
    """Return mean and error of magnetization."""
    axis = tuple([i+1 for i in range(len(cfgs.shape)-1)])
    return jackknife(cfgs.mean(axis=axis))

def get_abs_mag(cfgs: np.ndarray):
    """Return mean and error of absolute magnetization."""
    axis = tuple([i+1 for i in range(len(cfgs.shape)-1)])
    return jackknife(np.abs(cfgs.mean(axis=axis)))

def get_chi2(cfgs: np.ndarray):
    """Return mean and error of suceptibility."""
    V = np.prod(cfgs.shape[1:])
    axis = tuple([i+1 for i in range(len(cfgs.shape)-1)])
    mags = cfgs.mean(axis=axis)
    #return jackknife(V * (mags**2 - mags.mean()**2))
    return jackknife(((mags - mags.mean())**2))
    #return jackknife(V * (mags**2))

#only valid for 2D, since it only accounts for correlations in the same dimensions
#phi(x)phi(x+r) with r=0,0,0,1... but not r=1,1,0,0
def get_corr_func(cfgs: np.ndarray):
    """Return connected two-point correlation function with errors for symmetric lattices.""" #only valid for 2D lattices!!!
    mag_sq = np.mean(cfgs)**2
    corr_func = []
    axis = tuple([i+1 for i in range(len(cfgs.shape)-1)])

    for i in range(1, cfgs.shape[1], 1):
        corrs = []

        for mu in range(len(cfgs.shape)-1):
            corrs.append(np.mean(cfgs * np.roll(cfgs, i, mu+1), axis=axis))

        corrs = np.array(corrs).mean(axis=0)
        
        corr_mean, corr_err = jackknife(corrs - mag_sq)
        corr_func.append([i, corr_mean, corr_err])

    return np.array(corr_func)



def better_jackknife(samples: np.ndarray):                           
    """Return mean and estimated lower error bound."""
    N = samples.shape[0]
    original_mean = samples.mean(axis=0)

    subset_means = 1/(N-1) * (N * original_mean - samples)   #nice formel von Felipe

    subset_mean = subset_means.mean(axis=0)
    
    unbiased_estimator = N * original_mean - (N-1) * subset_mean   
    error = np.sqrt(  (N - 1) * np.mean(np.square(subset_means - original_mean), axis=0)   )
        
    return unbiased_estimator, error
        



def get_autocorrelation_cpp(data, file_to_save, exe_path):

    np.savetxt(file_to_save, data, newline='\n')

    bashCommand = [exe_path, str(len(data)), file_to_save]

    process = subprocess.Popen(bashCommand, stdout=subprocess.PIPE)
    output, error = process.communicate()

    results_from_cpp = np.array(output.decode().split(), dtype=np.double)
    labels = ["average", "sigmaF","SigmaF_error", "ACtime", "ACtimeError"]
    return labels, results_from_cpp
    


def jackknife(samples: np.ndarray):                           
    """Return mean and estimated lower error bound."""
    N = samples.shape[0]
    original_mean = samples.mean(axis=0)

    subset_means = []
    for i in tqdm(range(samples.shape[0])):
        subset_means.append(np.delete(samples, i, axis=0).mean(axis=0))

    subset_means = np.asarray(subset_means)
    subset_mean = subset_means.mean(axis=0)
    
    unbiased_estimator = N * original_mean - (N-1) * subset_mean   
    error = np.sqrt((N - 1) * np.mean(np.square(subset_means - original_mean), axis=0))
        
    return unbiased_estimator, error



def jackknife_para_auxiliary(input_pool):
    i = input_pool[0]
    samples = input_pool[1]
    """Return mean and estimated lower error bound."""
    return np.delete(samples, i, axis=0).mean(axis=0)


def jackknife_para(samples: np.ndarray, number_processors):
    N = samples.shape[0]
    original_mean = samples.mean(axis=0)   

    list_iterations = np.arange(0,N,1)
    input_pool = [ [k, samples] for k in list_iterations] 
            
    pool = mp.Pool(number_processors)
    subset_means =pool.map(jackknife_para_auxiliary, input_pool)
    pool.close()    
    subset_means = np.asarray(subset_means)
    subset_mean = subset_means.mean(axis=0)

    unbiased_estimator = N * original_mean - (N-1) * subset_mean   
    error = np.sqrt((N - 1) * np.mean(np.square(subset_means - original_mean), axis=0))
    
    return unbiased_estimator, error




# finally renormalized mass diggaaahhhhhhh

def get_ren_mass_right_via_hdf(field_configs, lattice_size):
    axis = tuple([i+1 for i in range(len(configs.shape)-1)])
    spatial_vol = np.prod(lattice_size[1:])
    d = len(lattice_size)
    
    a = np.arange(0,lattice_size[0])
    a[a>int(lattice_size[0]/2)] = lattice_size[0]-a[a>int(lattice_size[0]/2)]

    S_t = np.mean(field_configs, axis = axis)
    
    connected_corr_t, connected_corr_t_err = get_connected_2pt_fct(S_t)
    chi2 = spatial_vol * np.sum( connected_corr_t)
    mu2 = d * spatial_vol * np.sum( connected_corr_t * a**2)
    
    ren_mass2 = 2*d * chi2/mu2
    ren_mass = np.sqrt( ren_mass2 )
    
    chi_err = spatial_vol * np.sqrt( np.sum( connected_corr_t_err**2 ))
    mu_err = 2*spatial_vol * np.sqrt( np.sum( (a**2 * connected_corr_t_err)**2  )   )
    mass2_err = ren_mass2 * np.sqrt( (chi_err/chi2)**2 + (mu_err/mu2)**2 )
    mass_err = ren_mass *0.5 *mass2_err/ren_mass2

    
    return ren_mass, mass_err


def get_ren_mass_right_via_timeslices(S_t, lattice_size):

    spatial_vol = np.prod(lattice_size[1:])
    d = len(lattice_size)
    
    a = np.arange(0,lattice_size[0])
    a[a>int(lattice_size[0]/2)] = lattice_size[0]-a[a>int(lattice_size[0]/2)]


    connected_corr_t, connected_corr_t_err = get_connected_2pt_fct(S_t)
    chi2 = spatial_vol * np.sum( connected_corr_t)
    mu2 = d * spatial_vol * np.sum( connected_corr_t * a**2)
    
    ren_mass2 = 2*d * chi2/mu2
    ren_mass = np.sqrt( ren_mass2 )
        
    chi_err = spatial_vol * np.sqrt( np.sum( connected_corr_t_err**2 ))
    mu_err = 2*spatial_vol * np.sqrt( np.sum( (a**2 * connected_corr_t_err)**2  )   )
    mass2_err = ren_mass2 * np.sqrt( (chi_err/chi2)**2 + (mu_err/mu2)**2 )
    mass_err = ren_mass *0.5 *mass2_err/ren_mass2

    return ren_mass, mass_err


def get_ren_mass2_right_via_timeslices(S_t, lattice_size):

    spatial_vol = np.prod(lattice_size[1:])
    d = len(lattice_size)
    
    a = np.arange(0,lattice_size[0])
    a[a>int(lattice_size[0]/2)] = lattice_size[0]-a[a>int(lattice_size[0]/2)]


    connected_corr_t, connected_corr_t_err = get_connected_2pt_fct(S_t)
    chi2 = spatial_vol * np.sum( connected_corr_t)
    mu2 = d * spatial_vol * np.sum( connected_corr_t * a**2)
    
    ren_mass2 = 2*d * chi2/mu2     
    chi_err = spatial_vol * np.sqrt( np.sum( connected_corr_t_err**2 ))
    mu_err = 2*spatial_vol * np.sqrt( np.sum( (a**2 * connected_corr_t_err)**2  )   )
    mass2_err = ren_mass2 * np.sqrt( (chi_err/chi2)**2 + (mu_err/mu2)**2 )


    return ren_mass2, mass2_err
    
    
    
    

#Get the Correlator ---------------------------------------------------------------------------------

#out of configurations

#connected 2pt fct from timeslices
def get_connected_2pt_fct(S_t):
    S_t_mean = S_t.mean(axis=0)
    tmp = S_t - S_t_mean
    before_mean = []
    for t in range(len(S_t_mean)):
        before_mean.append( np.mean( tmp * np.roll(tmp,t,1), axis=1)  )

    before_mean = np.array(before_mean).T   #average over the configs
    
  
    return better_jackknife(before_mean)   #first entry are means, second are errors 



#connected 2pt fct from timeslices
#first version, better one for connected 2pt fct
def get_connected_2pt_fct_old_jackknife(S_t, get_error=False, number_processors=1):
    S_t_mean = S_t.mean(axis=0)
    tmp = S_t - S_t_mean
    before_mean = []
    for t in range(len(S_t_mean)):
        before_mean.append( np.mean( tmp * np.roll(tmp,t,1), axis=1)  )

    before_mean = np.array(before_mean).T   #average over the configs
    
    if get_error:
        if number_processors == 1:
            connected_corr_t = np.array(jackknife(before_mean))
        else:
            connected_corr_t = np.array(jackknife_para(before_mean, number_processors))
    else:
        connected_corr_t = np.array( [np.mean(before_mean, axis=0), np.zeros(len(S_t_mean))]    )

    return connected_corr_t    #first entry are means, second are errors 


#second way by calculating speratly disconnected and connected part
def get_connected_2pt_fct_version2(S_t):
    S_t_mean = S_t.mean(axis=0)

    discon = []
    for i in range(len(S_t_mean)):
        discon.append( np.mean(S_t_mean*np.roll(S_t_mean, i)) )
    discon = np.array(discon)#/dimensions[0]


    full = []
    for t in range(len(S_t_mean)):
        full.append( np.mean(S_t*np.roll(S_t,t,1))  )     #roll: howmany times, axis
    full = np.array(full)

    connected_corr_t =  full - discon

    return connected_corr_t



#overall correlator from configs
def get_overall_connected_correlator(configs):   #overall = time & spatial

    dimensions=np.array(configs.shape[1:])
    axis = tuple([i+1 for i in range(len(configs.shape)-1)])
    connected = []
    x_all = []
    tmp = configs-np.mean(configs,axis=0)
    for a in tqdm(range(0, dimensions[0], 1)): 
        for b in range(0, dimensions[1], 1):
            for c in range(0, dimensions[2], 1):
                for d in range(0, dimensions[3], 1):
                    x_2 = a**2 + b**2 + c**2 + d**2
                    x_all.append(x_2)
                    connected.append( np.mean(  tmp * np.roll(  np.roll(   np.roll( np.roll(tmp,a,1), b, 2) ,c,3), d,4), axis=axis) )  #mu+1 because first dimension are the configs

    connected_x_all = np.array(connected).mean(axis=1) #mean over different configurations


 
    x_unique = np.unique(x_all)
    connected_x = []
    for x in x_unique:
        connected_x.append(  np.mean( connected_x_all[np.argwhere(x == x_all)] ) )


    return connected_x, connected_x_all, np.array(x_all)
