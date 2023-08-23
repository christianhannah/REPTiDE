#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Jun 25 15:05:13 2022

@author: christian
"""

import numpy as np
from scipy import integrate
import matplotlib.pyplot as plt
from scipy.interpolate import interp1d
from astropy.io import fits

# CONSTANTS
pc_to_m = 3.08567758128e16 
M_sol =  1.989e30 # kg
R_sol = 696.34e6 # m
G = 6.6743e-11 # m^3 s^-2 kg^-1

# ============================ FUNCTIONS ======================================

# =============================================================================
# function to read in 3-D radius and 3-D densities from Stone & Metzger (2016)
def get_stone_data(gal_name): # use no space gal name 
    
    file_path = '../Data_Sets/Stone_Metzger_data/RenukaAnil.dat'
    f = open(file_path, 'r')
    all_lines = f.readlines()
    f.close()
    
    names = []
    radii = np.zeros(len(all_lines))
    densities = np.zeros(len(all_lines))

    for i in range(len(all_lines)):
        split = all_lines[i].split()
        names.append(split[0])
        radii[i] = float(split[1])
        densities[i] = float(split[2])
        
    names = np.array(names)
    names_f = names[np.where(names == gal_name)]
    radii_f = radii[np.where(names == gal_name)]
    densities_f = densities[np.where(names == gal_name)]
    
    if len(names_f) == 0:
        return 0
    else:
        return [names_f, radii_f, densities_f]
# =============================================================================


    
# =============================================================================
# a little function for finding maximum outliers
def get_n_max(arr, n):
    temp_arr = np.copy(arr)
    maxes = np.zeros((n))
    maxes_idx = np.zeros((n)).astype(int)
    #pdb.set_trace()
    for i in range(n):
        maxes[i] = np.max(temp_arr)
        maxes_idx[i] = np.argmax(temp_arr)
        #pdb.set_trace()
        temp_arr[maxes_idx[i]] = -999999
    return maxes, maxes_idx

# =============================================================================



# =============================================================================
# a little function for finding minimum outliers
def get_n_min(arr, n):
    temp_arr = np.copy(arr)
    mins = np.zeros((n))
    mins_idx = np.zeros((n)).astype(int)
    #pdb.set_trace()
    for i in range(n):
        mins[i] = np.max(temp_arr)
        mins_idx[i] = np.argmax(temp_arr)
        #pdb.set_trace()
        temp_arr[mins_idx[i]] = -999999
    return mins, mins_idx

# =============================================================================   

# =============================================================================
# 
sig_table = '../Data_Sets/Lauer2007_data/Lauer_07a_sigma_table.fit'
hdul = fits.open(sig_table)  # open a FITS file
sig_dat = hdul[1].data
sig_names_init = sig_dat['Name']
sig_names = np.zeros(len(sig_names_init)).astype(str)
for i in range(len(sig_names)):
    sig_names[i] = sig_names_init[i].replace(" ","")
sigs = sig_dat['sigma']
hdul.close()
def get_mbh(name):
    ind = np.where(sig_names == name)[0]
    sigma = sigs[ind]
    #pdb.set_trace()
    return 10**8.32*(sigma/200)**5.64 # M-sigma from McConnell & Ma (2013)
    
# =============================================================================

# =============================================================================
# function to return power-law inner density profile modified for broken powerlaw test
def get_rho_r_nuker(r,dens_rad,dens,toy_model_match):

    # Interpolate the y-values for the new x-data within the original range
    y_interp = 10**np.interp(np.log10(r),np.log10(dens_rad),np.log10(dens))

    # Extrapolate the y-values for new x-values greater than the original x-array
    x_max = np.max(dens_rad)
    x_extrapolate = r[r > x_max]
    y_extrapolate = y_interp[r > x_max]
    y_extrapolate = y_extrapolate*np.exp(-(x_extrapolate - x_max)/(0.4*x_max))
    y_interp[r > x_max] = y_extrapolate

    # Extrapolate the y-values for new x-values greater than the original x-array
    slope = (np.log10(dens[1])-np.log10(dens[0]))/(np.log10(dens_rad[1])-np.log10(dens_rad[0]))
    x_extrap = r[r < dens_rad[0]]
    y_extrap = y_interp[r < dens_rad[0]]
    y_extrap = 10**(slope * (np.log10(x_extrap) - np.log10(dens_rad[0])) + np.log10(dens[0]))
    y_interp[r < dens_rad[0]] = y_extrap
    
    #pdb.set_trace()
    return y_interp
    
# =============================================================================



# =============================================================================
def get_encm_pot_integrand_nuker(r,dens_rad,dens,toy_model_match):
    return r**2*get_rho_r_nuker(r,dens_rad,dens,toy_model_match)    
# function to compute the mass enclosed from density profile 
def get_enc_mass_nuker(r,dens_rad,dens,max_ind,toy_model_match):
    if max_ind == 0:
        return 0
    else:
        return 4*np.pi*integrate.trapz(get_encm_pot_integrand_nuker(r[0:max_ind+1],dens_rad,dens,toy_model_match), r[0:max_ind+1])
# =============================================================================


# =============================================================================
def get_ext_pot_integrand_nuker(r,dens_rad,dens,min_ind,toy_model_match):
    return r*get_rho_r_nuker(r,dens_rad,dens,toy_model_match)   
# function to compute the contribution to the potential of the galaxy at 
# larger radii
def get_ext_potential_nuker(r,dens_rad,dens,min_ind,toy_model_match):
    return 4*np.pi*G*integrate.trapz(get_ext_pot_integrand_nuker(r[min_ind:],dens_rad,dens,min_ind,toy_model_match),r[min_ind:])
# =============================================================================


# =============================================================================
# derive the total gravitational potential (psi(r)) as a function of r
def get_psi_r_nuker(r,M_BH,dens_rad,dens,toy_model_match):
    
    psi_1 = G*M_BH/r
    
    if toy_model_match:
        return psi_1,0,0,0
    else:
        M_enc = np.zeros_like(r)
        for i in range(len(M_enc)):
            M_enc[i] = get_enc_mass_nuker(r,dens_rad,dens,i,toy_model_match)
        psi_2 = G*M_enc/r
    
        psi_3 = np.zeros_like(r)
        for i in range(len(psi_3)):
            psi_3[i] = get_ext_potential_nuker(r,dens_rad,dens,i,toy_model_match)
        
        return psi_1+psi_2+psi_3,psi_1,psi_2,psi_3
# =============================================================================



# =============================================================================
# define function to find value in an array nearest to a supplied value
def find_nearest(array, value):
    array = np.asarray(array)
    idx = np.abs(array - value).argmin()
           
    return idx
# =============================================================================   



# =============================================================================  
def get_avg_M_sq(M_min,M_max):
    M_min = 0.08
    M_max = 1
    masses = np.linspace(M_min,M_max,1000)
    masses_kg = masses*M_sol
    
    # use Kroupa present-day stellar mass function from S&M+16
    PDMF = get_PDMF(masses)
        
    return integrate.trapz(PDMF*masses_kg**2, masses_kg)/M_sol
# =============================================================================  



# =============================================================================  
def get_PDMF(masses):
    # Kroupa present-day stellar mass function from S&M+16
    PDMF = np.zeros_like(masses)
    for i in range(len(masses)):
        if masses[i] < 0.5:
            PDMF[i] = (1/1.61)*(masses[i]/0.5)**(-1.3)
            #PDMF[i] = 0.98*masses[i]**(-1.3)
        else:
            PDMF[i] = (1/1.61)*(masses[i]/0.5)**(-2.3)
            #PDMF[i] = 2.4*masses[i]**(-2.3)
            
        # normalize the PDMF to have an area of 1
        area = integrate.trapz(PDMF,masses)
        norm_fac = 1/area
        
    return norm_fac*PDMF
# =============================================================================  



# =============================================================================
def get_psi_t(e,t):
    return (e/2)*(np.tanh(np.pi/2*np.sinh(t))+1)
# =============================================================================


# =============================================================================
def integrand_p(rs,r,psi_r,e):
    #pdb.set_trace()
    #psi_r_p = G*M_BH/rs
    if len(rs) == 1:
        new_r = np.arange(0,2+0.1,0.1)*rs
        psi_r_p = 10**np.interp(np.log10(new_r),np.log10(r),np.log10(psi_r))[10] 
    else:
        new_r = rs
    psi_r_p = 10**np.interp(np.log10(new_r),np.log10(r),np.log10(psi_r))
    #psi_r_p = get_psi_r(rs,G,M_BH,slope,rho_b,r_b,smooth)
    return(2*(psi_r_p-e)**(-1/2))
# =============================================================================



# =============================================================================
def integrand_I_12(es,psi_i,e_DF,DF):
    DF_interp = 10**np.interp(np.log10(es),np.log10(e_DF),np.log10(DF))
    return (2*(psi_i-es))**(1/2)*DF_interp
# =============================================================================



# =============================================================================
def integrand_I_32(es,psi_i,e_DF,DF):
    DF_interp = 10**np.interp(np.log10(es),np.log10(e_DF),np.log10(DF))
    return (2*(psi_i-es))**(3/2)*DF_interp
# =============================================================================


import pdb
# =============================================================================
def integrand_mu(rs_mu,r,psi_r,e_DF_i,e_DF,DF,I_0,M_BH,avg_M_sq):
    if len(rs_mu) == 1:
        new_r = np.arange(0,2+0.1,0.1)*rs_mu
        psi_rs_mu = np.array(10**np.interp(np.log10(new_r),np.log10(r),np.log10(psi_r))[10])
    else:
        new_r = rs_mu
        psi_rs_mu = 10**np.interp(np.log10(new_r),np.log10(r),np.log10(psi_r))

    I_12_r = np.zeros(len(rs_mu))
    I_32_r = np.zeros(len(rs_mu))
    
    for j in range(len(rs_mu)):
        if len(rs_mu) == 1:
            psi_i = psi_rs_mu
        else:
            psi_i = psi_rs_mu[j]
        es_i = np.linspace(e_DF_i,psi_i,10**3)
        DF_interp_i = 10**np.interp(np.log10(es_i),np.log10(e_DF),np.log10(DF))
        
        I_12_r[j] = (2*(psi_i-e_DF_i))**(-1/2)*integrate.trapz((2*(psi_i-es_i))**(1/2)*DF_interp_i,es_i)
        I_32_r[j] = (2*(psi_i-e_DF_i))**(-3/2)*integrate.trapz((2*(psi_i-es_i))**(3/2)*DF_interp_i,es_i)
    
    J_c_e = G*M_BH/(2*e_DF_i)**(1/2)    
    lim_thing_r = (32*np.pi**2*rs_mu**2*G**2*avg_M_sq*np.log(0.4*M_BH/M_sol))/(3*J_c_e**2)* \
                    (3*I_12_r - I_32_r + 2*I_0)
    #pdb.set_trace()
    return lim_thing_r/np.sqrt(2*(psi_rs_mu-e_DF_i))
# =============================================================================



# =========================== END OF FUNCTIONS ================================
    
    
    
    
    
    
    
    
    
    
    