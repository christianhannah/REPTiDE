#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Feb 21 12:40:32 2023

@author: christian
"""

import numpy as np
import matplotlib.pyplot as plt
from astropy.io import fits
from scipy.optimize import curve_fit
import pdb
from astropy.stats import median_absolute_deviation as mad
from astroquery.ned import Ned
import astropy.units as uni
from astropy.coordinates import SkyCoord
from astropy.coordinates import match_coordinates_sky
import csv
from numpy.linalg import eig, inv
from scipy import integrate
import time
from astropy.table import Table
from tqdm import tqdm

# =============================================================================
# define function to find value in an array nearest to a supplied value
def find_nearest(array, value):
    array = np.asarray(array)
    idx = np.abs(array - value).argmin()
           
    return idx
# =============================================================================   

# =============================================================================
# function to return power-law inner density profile
def get_rho_r_1(r,slope,rho_b,r_b,smooth,decay_params):
    cen = decay_params[0]
    wid = decay_params[1]
    brk = find_nearest(r,cen)
    r_low = r[0:brk]
    r_high = r[brk:]
    rho_low = np.zeros_like(r_low)
    rho_high = np.zeros_like(r_high)
    rho_low = rho_b*(r_low/r_b)**(-7/4)*(0.5*(1+(r_low/r_b)**(1/smooth)))**((7/4-slope)*smooth)
    rho_high = rho_b*(r_high/r_b)**(-7/4)*(0.5*(1+(r_high/r_b)**(1/smooth)))**((7/4-slope)*smooth)*np.exp(-(r_high-r[brk])/wid)
    return np.concatenate((rho_low,rho_high))
# =============================================================================


# =============================================================================
# function to return power-law inner density profile
def get_rho_r_new(r,slope,rho_b,r_b,smooth,decay_params):
    return np.piecewise(r, [r>=decay_params[0], r<decay_params[0]], 
                        [lambda r,slope,rho_b,r_b,smooth,decay_params : rho_b*(r/r_b)**(-7/4)*(0.5*(1+(r/r_b)**(1/smooth)))**((7/4-slope)*smooth)*np.exp(-(r-decay_params[0])/decay_params[1]), 
                         lambda r,slope,rho_b,r_b,smooth,decay_params : rho_b*(r/r_b)**(-7/4)*(0.5*(1+(r/r_b)**(1/smooth)))**((7/4-slope)*smooth)], slope,rho_b,r_b,smooth,decay_params)
# =============================================================================

# =============================================================================
# function to return power-law inner density profile
def get_rho_r(r,slope,rho_b,r_b,smooth):
    return rho_b*(r/r_b)**(-7/4)*(0.5*(1+(r/r_b)**(1/smooth)))**((7/4-slope)*smooth)
# =============================================================================

# =============================================================================
def get_y(r,slope,rho_b,r_b,smooth,decay_params):
    return r**2*get_rho_r_new(r,slope,rho_b,r_b,smooth,decay_params)    
# function to compute the mass enclosed from density profile 
def get_enc_mass(r,slope,rho_b,r_b,smooth,max_ind,decay_params):
    if max_ind == 0:
        return 0
    else:
        return 4*np.pi*integrate.trapz(get_y(r[0:max_ind+1],slope,rho_b,r_b,smooth,decay_params), r[0:max_ind+1])
        #return 4*np.pi*integrate.quadrature(get_y, r[0],r[max_ind],args=(slope,rho_b,r_b,smooth,decay_params),
        #                                    tol=10**-35,maxiter=500)[0]
# =============================================================================

# =============================================================================
def get_y_1(r,slope,rho_b,r_b,smooth,min_ind,decay_params):
    return r*get_rho_r_new(r,slope,rho_b,r_b,smooth,decay_params)   
# function to compute the contribution to the potential of the galaxy at 
# larger radii
def get_ext_potential(r,G,slope,rho_b,r_b,smooth,min_ind,decay_params):
    return 4*np.pi*G*integrate.trapz(get_y_1(r[min_ind:],slope,rho_b,r_b,smooth,min_ind,decay_params),r[min_ind:])
    #return 4*np.pi*G*integrate.quadrature(get_y_1,r[min_ind],r[-1],args=(slope,rho_b,r_b,smooth,min_ind,decay_params),
    #                                        tol=10**-35,maxiter=300)[0]
    #return 4*np.pi*G*integrate.quad(get_y_1,r[min_ind],r[-1],args=(slope,rho_b,r_b,smooth,min_ind,decay_params))[0]
# =============================================================================

# =============================================================================
# derive the total gravitational potential (psi(r)) as a function of r
def get_psi_r(r,G,M_BH,slope,rho_b,r_b,smooth,decay_params):
    psi_1 = G*M_BH/r
     
    M_enc = np.zeros_like(r)
    for i in range(len(M_enc)):
        M_enc[i] = get_enc_mass(r,slope,rho_b,r_b,smooth,i,decay_params)
    psi_2 = G*M_enc/r
    
    psi_3 = np.zeros_like(r)
    for i in range(len(psi_3)):
        psi_3[i] = get_ext_potential(r,G,slope,rho_b,r_b,smooth,i,decay_params)
        
    return psi_1+psi_2+psi_3,psi_1,psi_2,psi_3
# =============================================================================


# CONSTANTS
pc_to_m = 3.08567758128e16 
M_sol =  1.989e30 # kg
R_sol = 696.34e6 # m
G = 6.6743e-11 # m^3 s^-2 kg^-1

M_BH = 10**6*M_sol
slope,rho_b,r_b,smooth = (1.9, 5.101576486204166e-05, 308567758128.0, 0.1)
G = 6.6743e-11
#r = np.geomspace(3085676904942637.5, 3.085676904942637e+20,1000) # in m
r = np.geomspace(10**-6*pc_to_m, 10**15*pc_to_m,int(10000)) # in m

wids = np.array([10,20,30,50,100,1000,10000])*pc_to_m # m, radius to begin decay
wids_pc = np.array([10,20,30,50,100,1000,10000])
cens = np.ones(len(wids))*10*pc_to_m # m, width (halflife) of decay
decay_params = np.array([cens,wids])

wids_pc = np.array([10,100,1000,10000,100000,1000000,10000000])
wids = wids_pc*pc_to_m # m, radius to begin decay
cens = np.ones(len(wids))*10*pc_to_m # m, width (halflife) of decay
decay_params = np.array([cens,wids])

rho = get_rho_r(r, slope, rho_b, r_b, smooth)
rhos = np.zeros((len(cens),len(r)))
for i in range(len(cens)):
    rhos[i,:] = get_rho_r_new(r, slope, rho_b, r_b, smooth, decay_params[:,i])

#rho_1 = get_rho_r_1(r, slope, rho_b, r_b, smooth, decay_params)



#%%
# let's read in a real density profile
my_dat = Table.read('../Result_Tables/all_gal_data_2x_pixel_scale_or_10pc_extinction_corr_nsa_ml_w_vi_color.fits')
select = (my_dat['name'] == 'NGC 4472')
lograd_real = my_dat['lograd'][select].value[0]
logrho_real = my_dat['logdens'][select].value[0]
rad_real = 10**lograd_real*pc_to_m
rho_real = 10**logrho_real*M_sol/pc_to_m**3 #kg/m^3

#%%

pots = np.zeros((len(cens),4,len(r)))

for i in tqdm(range(len(cens)), position=0, leave=True):
    pots[i,:,:] = get_psi_r(r,G,M_BH,slope,rho_b,r_b,smooth,decay_params[:,i])

#a = get_psi_r(r,G,M_BH,slope,rho_b,r_b,smooth)

#%%
cmap=plt.get_cmap("turbo")

plt.figure(dpi=500)
plt.plot(np.log10(r/pc_to_m),np.log10(rho),color='k')

for i in range(len(cens)):
    plt.plot(np.log10(r/pc_to_m),np.log10(rhos[i,:]),label='10$^{:.0F}$ pc'.format(np.log10(wids_pc[i])),
             color=cmap((float(i)+1)/len(cens)))

#plt.plot(np.log10(r/pc_to_m),np.log10(rho_1))
#plt.plot(np.log10(rad_real/pc_to_m),np.log10(rho_real),'k')
#plt.ylim(-25,-10)
#plt.ylim(-15.8,-15.6)
plt.legend()
#plt.xlim(0.95,1.05)
plt.xlabel('log(Radius [pc])')
plt.ylabel('log($\\rho(r)$ [kg/m$^3$])')
plt.title('logM$_{BH}$=6, $\gamma$=-1.9, log($\\rho_{10pc}$)=3.5 M$_\odot$/pc$^3$')
plt.show()


#%%

plt.figure(dpi=500)
#plt.plot(np.log10(r*3.24078e-17),np.log10(psi_1+psi_2+psi_3),'k')
# =============================================================================
# for i in range(len(cens)):
#     plt.plot(np.log10(r/pc_to_m),np.log10(pots[i,0,:]),label='10$^{:.0F}$ pc'.format(np.log10(wids_pc[i])),
#              color=cmap((float(i)+1)/len(cens)),linewidth=3,alpha=0.5)
# =============================================================================
for i in range(len(cens)):
    plt.plot(np.log10(r/pc_to_m),np.log10(pots[i,2,:]),linestyle='--',
             color=cmap((float(i)+1)/len(cens)))
for i in range(len(cens)):
    plt.plot(np.log10(r/pc_to_m),np.log10(pots[i,3,:]),linestyle='-',
             color=cmap((float(i)+1)/len(cens)))
# =============================================================================
# plt.plot(np.log10(r*3.24078e-17),np.log10(psi_1),label='$\psi_{BH}$')
# plt.plot(np.log10(r*3.24078e-17),np.log10(psi_2),label='$\psi_{enc}$')
# plt.plot(np.log10(r*3.24078e-17),np.log10(psi_3),label='$\psi_{ext}$')
# plt.plot(np.log10(r*3.24078e-17),np.log10(psi_1+psi_2+psi_3),'k--', label='Total')
# =============================================================================
plt.plot(0,0,linestyle='--',label='$\psi_{enc}$',color='k')
plt.plot(0,0,linestyle='-',label='$\psi_{ext}$',color='k')
plt.plot(np.log10(r/pc_to_m),np.log10(pots[i,1,:]),label='$\psi_{BH}$',color='k',linestyle=':')
plt.legend()
#plt.ylim(np.min(np.log10(pots[i,1,:])),15)#np.max(np.log10(pots[i,1,:])))
plt.ylim(-5,15)
plt.xlim(-6,15)
plt.xlabel('log(Radius [pc])')
plt.ylabel('log($\psi(r)$ [m$^2$/s$^2$])')
plt.title('logM$_{BH}$=6, $\gamma$=-1.9, log($\\rho_{10pc}$)=3.5 M$_\odot$/pc$^3$')
plt.show()
