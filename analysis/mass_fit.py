import h5py
import numpy as np
import matplotlib.pyplot as plt
import os, fnmatch
from numpy import linalg as LA
import copy 
import matplotlib


from correlations import *

##Note: don't fit cosh generally, fit exp! cosh is only for free case


def exp_fitfunc(t, a,m):
    return a*np.exp(-m*t)

def linear_fitfunc(t, a,m):
    return -m*t+a

def get_masses_from_small_t(S_t, dimensions, inputs,inc_in_fit=5, with_plot = False):
    mass2 = inputs[0]
    lam = inputs[1]
    yukawa = inputs[2]
    
    tmp = get_connected_2pt_fct(S_t)
    connected_corr_t = tmp[0]
    connected_corr_t_err = tmp[1]
    
    tmp = np.mean( np.array( [connected_corr_t[1:int(dimensions[0]/2)], np.flip(connected_corr_t[1+int(dimensions[0]/2):]) ]), axis=0)   #average half -> leave out site 0, array is uneven, so leave also out middle of uneven 1-59
    half_averaged_corr = np.abs( np.append( tmp , connected_corr_t[ int(dimensions[0]/2)]   ))
    

    ##how many values to include in fit:
    x = np.arange(1,inc_in_fit+1)


    ## Fit-procedure 1: Fit exp
    y = half_averaged_corr[:inc_in_fit]
    valid = ~(np.isnan(x) | np.isnan(y))  #include in fit bei linear
    popt_exp,pcov_exp = curve_fit(exp_fitfunc, x[valid], y[valid],p0=[10**-3,1], maxfev=10000)
    mass_exp = np.array([popt_exp[1], np.sqrt(pcov_exp[1][1])])
    if pcov_exp[1][1]>1 and ~(np.isinf(pcov_exp[1][1])):
        popt_exp = [np.nan,np.nan]
        pcov_exp[1][1] = np.nan
    mass_exp = np.array([popt_exp[1], np.sqrt(pcov_exp[1][1])])




    ## Fit-procedure 2: Take log and make linear Fit
    half_averaged_corr_log = np.log(half_averaged_corr)
    y = half_averaged_corr_log[:inc_in_fit]
    valid = ~(np.isnan(x) | np.isnan(y))  #include in fit bei linear
    try:
        popt_lin,pcov_lin = curve_fit(linear_fitfunc, x[valid], y[valid], p0=[1,1], maxfev=10000)
       
        if pcov_lin[1][1]>1 and ~(np.isinf(pcov_lin[1][1])):
            popt_lin = [np.nan,np.nan]
            pcov_lin[1][1] = np.nan
        mass_lin = np.array([popt_lin[1], np.sqrt(pcov_lin[1][1])])
    except TypeError:
        popt_lin = [np.nan,np.nan]
        mass_lin = np.array([np.nan, np.nan])
        print("TO LESS VALUES TO FIT")



    ## Procedure 3: Compute effective mass
    m_effs = np.abs( np.log( np.abs(connected_corr_t[1:-1]/connected_corr_t[2:])) ) #take out first weird point in time correlator
    n_t_half = np.arange(1.5,len(connected_corr_t)-0.5,1)
    valid = ~(np.isnan(m_effs))

    a = connected_corr_t[1:-1]
    b = connected_corr_t[2:]
    s_a = connected_corr_t_err[1:-1]
    s_b = connected_corr_t_err[2:]
    
    tmp = m_effs * np.sqrt( (s_a/a)**2 +(s_b/b)**2 )
    m_eff_err = 1/np.log(10)*tmp/m_effs
    m_eff = np.array([m_effs[valid][0], m_eff_err[valid][0]])
    
    """
except ValueError:
    inc_in_fit = 4
    ## Fit-procedure 1: Fit exp
    popt_exp,pcov_exp = curve_fit(exp_fitfunc,np.arange(1,inc_in_fit+1), half_averaged_corr[:inc_in_fit],p0=[10**-3,1], maxfev=10000)
    mass_exp = np.array([popt_exp[1], np.sqrt(pcov_exp[1][1])])
    print("Mass from exp Fit: ", popt_exp[1], np.sqrt(pcov_exp[1][1]))

    ## Fit-procedure 2: Take log and make linear Fit
    half_averaged_corr_log = np.log(half_averaged_corr)
    popt_lin,pcov_lin = curve_fit(linear_fitfunc,np.arange(1,inc_in_fit+1), half_averaged_corr_log[:inc_in_fit],p0=[-5,1], maxfev=10000)
    mass_lin = np.array([popt_lin[1], np.sqrt(pcov_lin[1][1])])
    print("Mass from Linear Fit: ", popt_lin[1], np.sqrt(pcov_lin[1][1]))

    ## Procedure 3: Compute effective mass
    m_effs = np.abs( np.log(connected_corr_t[1:-1]/connected_corr_t[2:]) ) #take out first weird point in time correlator
    n_t_half = np.arange(1.5,len(connected_corr_t)-0.5,1)
    m_eff = m_effs[0]
    print("Mass effective: ",m_eff)
    print("Mass effective averaged: ", np.nanmean(m_effs))
    """

    if with_plot:
        ##Plotting
        fig,ax = plt.subplots(1,3 , figsize=(20,4))
        ax[0].plot(np.arange(1,len(half_averaged_corr)+1), half_averaged_corr)
        ax[0].text(0.7, 0.8, '$m_{phys}=$'+str(np.round(mass_exp[0],3)), transform=ax[0].transAxes)
        ax[0].plot(np.arange(1,8), exp_fitfunc(np.arange(1,8),*popt_exp))
        ax[0].set_yscale("log")
        ax[0].set_xlabel("$N_t$", fontsize=13)
        ax[0].set_ylabel("$G_c(t)$", fontsize=13)
        ax[0].set_title(f"Exp. Fit: $m^2={mass2}, \; \lambda$={lam},\; g={yukawa}$") 
        
        ax[1].plot(np.arange(1,len(half_averaged_corr_log)+1), half_averaged_corr_log)
        ax[1].text(0.7, 0.8, '$m_{phys}=$'+str(np.round(mass_lin[0],3)), transform=ax[1].transAxes, fontsize=12)
        ax[1].plot(np.arange(1,8), linear_fitfunc(np.arange(1,8),*popt_lin))
        ax[1].set_xlabel("$N_t$", fontsize=13)
        ax[1].set_ylabel("$log( G_c(t) )$", fontsize=13)
        ax[1].set_title(f"Lin. Fit: $m^2={mass2}, \; \lambda$={lam},\; g={yukawa}$") 
        

        ax[2].hlines(m_eff[0],0,len(connected_corr_t), color="red", linestyle="--") 
        ax[2].text(0.7, 0.8, '$m_{phys}=$'+str(np.round(m_eff,3)), transform=ax[2].transAxes)
        ax[2].plot(n_t_half, m_effs)
        ax[2].set_xlabel("$N_t$", fontsize=13)
        ax[2].set_ylabel("$m_{eff}$", fontsize=13)
        ax[2].set_title(f"$m^2={mass2}, \; \lambda$={lam}, \; g={yukawa}$") 

        #plt.savefig("Figures/meff_new_m1.0_t60.pdf", bbox_inches="tight")
    
    
    return mass_exp, mass_lin, m_eff



def get_masses_from_large_t(S_t, dimensions, inputs,inc_in_fit=5, with_plot = True):
    mass2 = inputs[0]
    lam = inputs[1]
    yukawa = inputs[2]
    
    tmp = get_connected_2pt_fct(S_t)
    connected_corr_t = tmp[0]
    connected_corr_t_err = tmp[1]
    
    tmp = np.mean( np.array( [connected_corr_t[1:int(dimensions[0]/2)], np.flip(connected_corr_t[1+int(dimensions[0]/2):]) ]), axis=0)   #average half -> leave out site 0, array is uneven, so leave also out middle of uneven 1-59
    #k = np.arange(0,32)
    #test = np.mean( np.array( [k[1:int(dimensions[0]/2)], np.flip(k[1+int(dimensions[0]/2):]) ]), axis=0)  
    half_averaged_corr = np.abs( np.append( tmp , connected_corr_t[ int(dimensions[0]/2)]   ))
    

    ##how many values to include in fit:
    x = np.arange(1, len(half_averaged_corr)+1)[-inc_in_fit-1:-1] 
    x_plot = x

    ## Fit-procedure 1: Fit exp
    y = half_averaged_corr[-inc_in_fit-1:-1]
    valid = ~(np.isnan(x) | np.isnan(y))  #include in fit bei linear
    popt_exp,pcov_exp = curve_fit(exp_fitfunc, x[valid], y[valid],p0=[10**-3,1], maxfev=100000)
    mass_exp = np.array([popt_exp[1], np.sqrt(pcov_exp[1][1])])
    if pcov_exp[1][1]>1 and ~(np.isinf(pcov_exp[1][1])):
        popt_exp = [np.nan,np.nan]
        pcov_exp[1][1] = np.nan
    mass_exp = np.array([popt_exp[1], np.sqrt(pcov_exp[1][1])])


    ## Fit-procedure 2: Take log and make linear Fit
    half_averaged_corr_log = np.log(half_averaged_corr)
    y = half_averaged_corr_log[-inc_in_fit-1:-1]
    valid = ~(np.isnan(x) | np.isnan(y))  #include in fit bei linear
    try:
        popt_lin,pcov_lin = curve_fit(linear_fitfunc, x[valid], y[valid], p0=[1,1], maxfev=100000)
       
        if pcov_lin[1][1]>1 and ~(np.isinf(pcov_lin[1][1])):
            popt_lin = [np.nan,np.nan]
            pcov_lin[1][1] = np.nan
        mass_lin = np.array([popt_lin[1], np.sqrt(pcov_lin[1][1])])
    except TypeError:
        popt_lin = [np.nan,np.nan]
        mass_lin = np.array([np.nan, np.nan])
        print("TO LESS VALUES TO FIT")
        
    
    ## Procedure 3: Compute effective mass
    m_effs = np.abs( np.log(connected_corr_t[1:-1]/connected_corr_t[2:]) ) #take out first weird point in time correlator
    n_t_half = np.arange(1.5,len(connected_corr_t)-0.5,1)
    valid = ~(np.isnan(m_effs))

    a = connected_corr_t[1:-1]
    b = connected_corr_t[2:]
    s_a = connected_corr_t_err[1:-1]
    s_b = connected_corr_t_err[2:]
    
    tmp = m_effs * np.sqrt( (s_a/a)**2 +(s_b/b)**2 )
    m_eff_err = 1/np.log(10)*tmp/m_effs
    m_tmp = np.nanmean(m_effs[int(len(m_effs)/2)-5:int(len(m_effs/2))-1 ])
    m_eff = np.array([m_tmp, m_eff_err[valid][0]])
    
    if with_plot:
        ##Plotting
        fig,ax = plt.subplots(1,3 , figsize=(20,4))
        ax[0].plot(np.arange(1,len(half_averaged_corr)+1), half_averaged_corr)
        ax[0].text(0.7, 0.8, '$m_{phys}=$'+str(np.round(mass_exp[0],3)), transform=ax[0].transAxes)
        ax[0].plot(x_plot, exp_fitfunc(x_plot,*popt_exp))
        ax[0].set_yscale("log")
        ax[0].set_xlabel("$N_t$", fontsize=13)
        ax[0].set_ylabel("$G_c(t)$", fontsize=13)
        ax[0].set_title(f"Exp. Fit: $m^2={mass2}, \; \lambda$={lam},\; g={yukawa}$") 
        
        ax[1].plot(np.arange(1,len(half_averaged_corr_log)+1), half_averaged_corr_log)
        ax[1].text(0.7, 0.8, '$m_{phys}=$'+str(np.round(mass_lin[0],3)), transform=ax[1].transAxes, fontsize=12)
        ax[1].plot(x_plot, linear_fitfunc(x_plot,*popt_lin))
        ax[1].set_xlabel("$N_t$", fontsize=13)
        ax[1].set_ylabel("$log( G_c(t) )$", fontsize=13)
        ax[1].set_title(f"Lin. Fit: $m^2={mass2}, \; \lambda$={lam},\; g={yukawa}$") 
        

        ax[2].hlines(m_eff[0],0,len(connected_corr_t), color="red", linestyle="--") 
        ax[2].text(0.7, 0.8, '$m_{phys}=$'+str(np.round(m_eff,3)), transform=ax[2].transAxes)
        ax[2].plot(n_t_half, m_effs)
        ax[2].set_xlabel("$N_t$", fontsize=13)
        ax[2].set_ylabel("$m_{eff}$", fontsize=13)
        ax[2].set_title(f"$m^2={mass2}, \; \lambda$={lam}, \; g={yukawa}$") 

        #plt.savefig("Figures/meff_new_m1.0_t60.pdf", bbox_inches="tight")
    
    
    return mass_exp, mass_lin, m_eff