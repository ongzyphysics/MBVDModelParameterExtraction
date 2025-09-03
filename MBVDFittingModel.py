# -*- coding: utf-8 -*-
"""
Created on Fri Aug  8 09:43:49 2025

@author: ongzy
"""

import numpy as np
from scipy.interpolate import interp1d, PchipInterpolator
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit

fname = 'zout_28Jul2025.csv'
dtmp = np.loadtxt(fname,comments='#',skiprows=5,delimiter=',',dtype=complex);
f_expt = dtmp[:,0].T.real # frequency data
s_expt = -1j*2*np.pi*f_expt # complex angular frequency data
zdata = dtmp[:,1:].T # table of complex impedance data

# %%

def FitMBVD(f,Z,f_bnd):
    """ 
    This is a 5-parameter fitting model based on the MBVD model, which has 6 circuit elements.
    We have 5 (not 6) parameters because the tau parameter in the MBVD model is set to zero.
    """
    
    ngrid_fine = int(1E6) # number of grid intervals in fine grid
    f_fine = np.linspace(np.min(f),np.max(f),1+ngrid_fine) # very fine frequency grid
    s_fine = -1j*2*np.pi*f_fine # complex frequency
    fitfunc_real = PchipInterpolator(f,Z.real) # very fine real impedance grid from interpolation
    fitfunc_imag = PchipInterpolator(f,Z.imag) # very fine imag impedance grid from interpolation
    Z_fine = fitfunc_real(f_fine) + 1j*fitfunc_imag(f_fine) # very fine impedance grid  
    ZS_fine = Z_fine*s_fine 
     
    nfilter = (f_fine>=f_bnd[0]) * (f_fine<=f_bnd[1]) # frequency range filter for finding initial peak values
    ZSmax = np.max(np.log10(np.abs(ZS_fine[nfilter] )))
    fp = f_fine[np.log10(np.abs(ZS_fine))==ZSmax][0]; # parallel resonance frequency from fitting        
    ZSmin = np.min(np.log10(np.abs(ZS_fine[nfilter] )))
    fs = f_fine[np.log10(np.abs(ZS_fine))==ZSmin][0] # series resonance frequency from fitting 
    
    ws = 2*np.pi*fs # series resonance angular frequency 
    wp = 2*np.pi*fp # parallel resonance angular frequency    
    Cp = 0.5/np.abs(ZS_fine[0]) + 0.5*ws*ws/wp/wp/np.abs(ZS_fine[-1]) # parallel resonance capacitance       
    
    # === Estimate Qs from FWHM of ws antipeak/valley of expt data ===
    min_abs_ZS = np.abs(ZS_fine[f_fine==fs]) # antipeak value of absolute impedance
    
    ns_min = np.nonzero(f_fine==fs)[0][0] # index of antipeak    
    
    for ns in range(ngrid_fine):
        ZSs = ZS_fine[ns_min+ns] # current value of impedance
        fs_max = f_fine[ns_min+ns] # current value of frequency
        if np.abs(ZSs) > (min_abs_ZS)*10**(3/20): # found rightmost frequency for antipeak width
            break
    
    for ns in range(ngrid_fine):
        ZSs = ZS_fine[ns_min-ns] # current value of impedance
        fs_min = f_fine[ns_min-ns] # current value of frequency
        if np.abs(ZSs) > (min_abs_ZS)*10**(3/20): # found leftmost frequency for antipeak width
            break
        
    Qs = fs/(fs_max-fs_min) # quality factor of series resonance
    delta_ws = 2*np.pi*(fs_max-fs_min) # angular frequency width of series resonance     
            
    # === Estimate Qp from FWHM of wp peak of expt data ===
    max_abs_ZS = np.abs(ZS_fine[f_fine==fp]) # peak value of absolute impedance    
    
    nr_max = np.nonzero(f_fine==fp)[0][0] # index of peak    
    
    for nr in range(ngrid_fine):
        ZSp = ZS_fine[nr_max+nr] # current value of impedance
        fp_max = f_fine[nr_max+nr] # current value of frequency
        if np.abs(ZSp) < (max_abs_ZS)*10**(-3/20): # found rightmost frequency for peak width
            break
    
    for nr in range(ngrid_fine):
        ZSp = ZS_fine[nr_max-nr] # current value of impedance
        fp_min = f_fine[nr_max-nr] # current value of frequency
        if np.abs(ZSp) < (max_abs_ZS)*10**(-3/20): # found leftmost frequency for peak width
            break
    
    Qp = fp/(fp_max-fp_min) # quality factor of parallel resonance
    delta_wp = 2*np.pi*(fp_max-fp_min) # angular frequency width of parallel resonance         
    
    # === Set initial estimated values of mBVD model parameters ===
    Cp_init = Cp
    wp_init = wp
    Qp_init = Qp
    ws_init = ws
    Qs_init = Qs
    
    # === Print initial estimates of 5 fitting parameters ===
    print(f"INITIAL ESTIMATE : ", end="") # print initial estimates of MBVD model
    print(f"Cp = {Cp_init:.3e}, fs = {ws_init/2/np.pi: .3e} ,", end="") 
    print(f"fp = {wp_init/2/np.pi: .3e}, Qs = {Qs_init: .2f} , Qp = {Qp_init: .2f}")
    
    # === 2-step iterative fitting process ===
    for nstep in range(20):
        print(f"STEP {nstep:4d} : ",end="")
        # === Parallel resonance parameters fitting ===
        print("\tFitting parallel resonance parameters... :",end="")
        Hp_fine = np.log(np.abs( ZS_fine/(s_fine*s_fine + ws/Qs*s_fine + ws*ws) ))        
        
        def Hpfitfunc(f_in,x0,x1,x2):
            """
            This is just the log-abs part of the impedance function that looks
            like a Cauchy distribution and depends on Cp, wp, and Qp                    
            """
            x = [x0,x1,x2]
            s_in = -1j*2*np.pi*f_in
            hf = np.log(np.abs( 1/(x[0]*1E-15)/(s_in*s_in + (x[1]*1E9)/x[2]*s_in + (x[1]*1E9)*(x[1]*1E9)) ))
            return hf
        
        xh_init = np.array([Cp/1E-15, wp/1E9, Qp]) # initial fitting parameters for Hpfitfunc
        xh_lb = np.array([0.25*Cp/1E-15, (wp-2*delta_wp)/1E9, 20]) # lower bound for xh
        xh_ub = np.array([1.75*Cp/1E-15, (wp+2*delta_wp)/1E9, 1000]) # upper bound for xh        
        xh_fit, pcov_h = curve_fit(Hpfitfunc, f_fine, Hp_fine, p0=xh_init, bounds=(xh_lb,xh_ub))
        
        Cp_fit = xh_fit[0]*1E-15
        wp_fit = xh_fit[1]*1E9
        Qp_fit = xh_fit[2]
        Cp, wp, Qp = Cp_fit, wp_fit, Qp_fit
        
        # === Series resonance parameters fitting ===
        print("\tFitting series resonance parameters... ");
        Hs_fine = np.log(np.abs( ZS_fine*Cp*(s_fine*s_fine + (wp/Qp)*s_fine + wp*wp) ))
        
        def Hsfitfunc(f_in,x0,x1):
            """
            This is just the log-abs part of the impedance function that looks
            like an upside-down Cauchy distribution and depends on ws and Qs                    
            """
            x = [x0,x1]
            s_in = -1j*2*np.pi*f_in
            gf = np.log(np.abs( s_in*s_in + (x[0]*1E9/x[1])*s_in + x[0]*1E9*x[0]*1E9 ))
            return gf
        
        xg_init = np.array([ws/1E9, Qs]) # initial fitting parameters for Hsfitfunc
        xg_lb = np.array([(ws-2*delta_ws)/1E9, 20]) # lower bound for xg
        xg_ub = np.array([(ws+2*delta_ws)/1E9, 2000]) # upper bound for xg
        xg_fit, pcov_g = curve_fit(Hsfitfunc, f_fine, Hs_fine, p0=xg_init, bounds=(xg_lb,xg_ub))
                
        ws_fit = xg_fit[0]*1E9;
        Qs_fit = xg_fit[1]
        ws, Qs = ws_fit, Qs_fit     
        
        x_fit = np.array( [*xh_fit,*xg_fit] )
        x_init = np.array( [*xh_init,*xg_init] )
        convergence_delta = np.linalg.norm(x_fit-x_init)/np.linalg.norm(x_init) # convergence variable        
        
        # === Print intermediate estimates of 5 fitting parameters ===
        print(f"\tSTEP ESTIMATE : ",end="")
        print(f"Cp = {Cp_fit:.3e}, fs = {ws_fit/2/np.pi: .3e} ,",end="") 
        print(f"fp = {wp_fit/2/np.pi: .3e}, Qs = {Qs_fit: .2f} , Qp = {Qp_fit: .2f}")
        print(f"\tCONVERGENCE      : delta = {convergence_delta:.4e}")
        
        if convergence_delta<1E-6:
            print(f"FINAL STEP at step {nstep:4d} with delta = {convergence_delta:.4e} \n")
            
            print(f"INITIAL ESTIMATE : ",end="")
            print(f"Cp = {Cp_init:.3e}, fs = {ws_init/2/np.pi: .3e} ,",end="") 
            print(f"fp = {wp_init/2/np.pi: .3e}, Qs = {Qs_init: .2f} , Qp = {Qp_init: .2f}")
                    
            print(f"FINAL ESTIMATE : ",end="")
            print(f"Cp = {Cp_fit:.3e}, fs = {ws_fit/2/np.pi: .3e} ,",end="") 
            print(f"fp = {wp_fit/2/np.pi: .3e}, Qs = {Qs_fit: .2f} , Qp = {Qp_fit: .2f}")
            print("\n"*4)
            break # stop iterative fitting process
    
    
    # plt.plot(f_fine,Hp_fine)
    # plt.plot(f_fine,Hpfitfunc(f_fine,*xh_init))
    # plt.plot(f_fine,Hpfitfunc(f_fine,*xh_fit))
    # plt.show()
    
    # plt.plot(f_fine,Hs_fine)
    # plt.plot(f_fine,Hsfitfunc(f_fine,*xg_init))
    # plt.plot(f_fine,Hsfitfunc(f_fine,*xg_fit))
    # plt.show()
    
    s = -1j*2*np.pi*f # complex frequency    
    Z_fit = (s*s + (ws/Qs)*s + ws*ws)/(s*s + (wp/Qp)*s + wp*wp)/s/Cp
    return Z_fit

# === Fit experimental impedance data to 5-parameter MBVD model ===
for n_expt in range(50):
    Z_expt = zdata[n_expt,:] # Z spectrum     
    Z_fit = FitMBVD(f_expt,Z_expt,np.array([2.5E9, 7.5E9])) # Fit expt Z spectrum to MBVD model
    
    # == Plot of expt and fitted Z spectra ===
    plt.plot(f_expt/1E9,np.log(np.abs(Z_expt)),label='Expt')
    plt.plot(f_expt/1E9,np.log(np.abs(Z_fit)),label='MBVD')
    plt.xlim([2.5,7.5])
    plt.xlabel('Frequency (GHz)',fontsize=12)
    plt.ylabel('$\log|Z|$',fontsize=12)
    plt.legend(loc='lower right',fontsize=16)
    plt.show()



