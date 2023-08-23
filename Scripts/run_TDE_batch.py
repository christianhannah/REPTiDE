#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Dec  5 23:34:47 2022

@author: christian
"""

import TDE_rate_modeling_batch_ as TDE
import numpy as np
import matplotlib.pyplot as plt
from astropy.io import fits
from scipy import integrate
from tqdm import tqdm

# CONSTANTS
pc_to_m = 3.08567758128e16 
M_sol =  1.989e30 # kg
R_sol = 696.34e6 # m
G = 6.6743e-11 # m^3 s^-2 kg^-1

# ================= DEFINE BATCH OF GALAXY PARAMETERS =========================




# =============================================================================
# functions using relations from Reines et al. 2015
def get_BH_mass_lt(m_gal):
    mean = 7.45 + 1.05*np.log10(m_gal/10**11) # gives log(M_BH)
    return mean
def get_BH_mass_et(m_gal):
    mean = 8.95 + 1.40*np.log10(m_gal/10**11) # gives log(M_BH)
    return mean
# =============================================================================


# ================= DEFINE BATCH OF GALAXY PARAMETERS =========================

model_gal_filename = '../Result_Tables/final_gal_data.fits'
hdul = fits.open(model_gal_filename)
gal_data = hdul[1].data 
hdul.close()

select = ((np.abs(gal_data['slope'])) < 2.25) & ((np.abs(gal_data['slope'])) > 0.5)

names = gal_data['name'][select]
num_runs = len(names)
slopes = np.abs(gal_data['slope'][select])
rho_5pc = 10**gal_data['cen_dens'][select]
types = gal_data['type'][select]
gal_masses = 10**gal_data['logmass'][select]

r_bs_pc = 1e-5
r_bs = r_bs_pc*pc_to_m # m
rho_bs = rho_5pc*(r_bs_pc/5)**(-slopes)*M_sol/pc_to_m**3 #kg/m^3
smooth = 0.1

# define BH masses for our galaxies using Reines+15 relations
bh_masses = np.zeros(len(names))
for i in range(len(bh_masses)):
    if types[i] == 0:
        bh_masses[i] = float(10**get_BH_mass_et(gal_masses[i])*M_sol)
    else:
        bh_masses[i] = float(10**get_BH_mass_lt(gal_masses[i])*M_sol)

decay_params = np.array([10,1000])*pc_to_m
decay_params_pc = np.array([10,1000])

# =============================================================================


#%%
# ========================= RUN TDE CODE FOR BATCH ============================
num_e = 98
num_r = 10**4

TDE_rates = np.zeros(num_runs)
orb_ens = np.zeros((num_runs, 98))
DFs = np.zeros((num_runs, 98))
qs = np.zeros((num_runs, 98))
LC_fluxes = np.zeros((num_runs, 98))
rs = np.zeros((num_runs, 10**4))
psi_tots = np.zeros((num_runs, 10**4))
psi_bhs = np.zeros((num_runs, 10**4))
psi_encs = np.zeros((num_runs, 10**4))
psi_exts = np.zeros((num_runs, 10**4))
for i in tqdm(range(len(names)), position=0, leave=True):
    TDE_rates[i],orb_ens[i,:],DFs[i,:],qs[i,:],LC_fluxes[i,:],rs[i,:],\
        psi_tots[i,:],psi_bhs[i,:],psi_encs[i,:],psi_exts[i,:]= \
        TDE.get_TDE_rate(float(slopes[i]), float(r_bs), float(rho_bs[i]), float(smooth), \
                         float(bh_masses[i]),decay_params)

# =============================================================================
#%%
# convert TDE rates from per second to per year 
sec_per_yr = 3.154e+7
TDE_rates = TDE_rates*sec_per_yr

# compute the TDE rates for a pure population of solar mass stars 
TDE_rates_single = np.zeros_like(TDE_rates)
for i in range(len(TDE_rates_single)):
    TDE_rates_single[i] = integrate.trapz(LC_fluxes[i,:], orb_ens[i,:])*sec_per_yr


#%%

# write the output to a fits file
c1 = fits.Column(name='TDE_rate_single', array=TDE_rates_single, format='D', unit='yr^-1')
c2 = fits.Column(name='TDE_rate', array=TDE_rates, format='D', unit='yr^-1')
c3 = fits.Column(name='orb_ens', array=orb_ens, format=str(num_e)+'D', unit='m^2/s^2')
c4 = fits.Column(name='DFs', array=DFs, format=str(num_e)+'D')
c5 = fits.Column(name='qs', array=qs, format=str(num_e)+'D')
c6 = fits.Column(name='LC_fluxes', array=LC_fluxes, format=str(num_e)+'D')
c7 = fits.Column(name='psi_rads', array=rs, format=str(num_r)+'D', unit='m^2/s^2')
c8 = fits.Column(name='psi_tots', array=psi_tots, format=str(num_r)+'D', unit='m^2/s^2')
c9 = fits.Column(name='psi_bhs', array=psi_bhs, format=str(num_r)+'D', unit='m^2/s^2')
c10 = fits.Column(name='psi_encs', array=psi_encs, format=str(num_r)+'D', unit='m^2/s^2')
c11 = fits.Column(name='psi_exts', array=psi_exts, format=str(num_r)+'D', unit='m^2/s^2')

c12 = fits.Column(name='names', array=names, format='A')
c13 = fits.Column(name='slopes', array=slopes, format='D')
c14 = fits.Column(name='rho_5pc', array=rho_5pc, format='D', unit='M_sol/pc^3')
c15 = fits.Column(name='types', array=types, format='I', unit='0=early,1=late')
c16 = fits.Column(name='gal_mass', array=gal_masses, format='D', unit='M_sol')
c17 = fits.Column(name='bh_mass', array=bh_masses/M_sol, format='D',unit='M_sol')

#extention used for the file name of results
sample_ext = 'our_gals_w_slope_0.5_to_2.25_reff_{}pc'.format(decay_params_pc[1])

t = fits.BinTableHDU.from_columns([c1,c2,c3,c4,c5,c6,c7,c8,c9,c10,c11,c12,c13,c14,c15,c16,c17])
t.writeto('../Result_Tables/TDE_output_'+sample_ext+'.fits',overwrite=True)

#%%

cmap=plt.get_cmap("turbo")

plt.figure(dpi=500)
for i in range(len(names)):
    plt.plot(np.log10(orb_ens[i,:]),np.log10(DFs[i,:]),
             color=cmap((float(i)+1)/len(names)))
#plt.ylim(-25,-10)
plt.ylim(-60,-57.5)
plt.xlim(7.5,14.5)
plt.legend()
#plt.xlim(0.95,1.05)
plt.ylabel('log(f($\epsilon$))')
plt.xlabel('log($\epsilon$)')
plt.title('logM$_{BH}$=6, $\gamma$=-1.9, log($\\rho_{10pc}$)=3.5 M$_\odot$/pc$^3$')
plt.show()

plt.figure(dpi=500)
for i in range(len(names)):
    plt.plot(np.log10(orb_ens[i,:]),np.log10(qs[i,:]),
             color=cmap((float(i)+1)/len(names)))
#plt.ylim(-25,-10)
plt.ylim(-9,6)
plt.legend()
plt.xlim(7.5,14.5)
plt.ylabel('log(q)')
plt.xlabel('log($\epsilon$)')
plt.title('logM$_{BH}$=6, $\gamma$=-1.9, log($\\rho_{10pc}$)=3.5 M$_\odot$/pc$^3$')
plt.show()


plt.figure(dpi=500)
for i in range(len(names)):
    plt.plot(np.log10(orb_ens[i,:]),np.log10(LC_fluxes[i,:]),
             color=cmap((float(i)+1)/len(names)))
#plt.ylim(-25,-10)
plt.ylim(-26,-23)
plt.xlim(7.5,14.5)
plt.legend()
#plt.xlim(0.95,1.05)
plt.ylabel('$log(\mathcal{F}(\epsilon)$)')
plt.xlabel('log($\epsilon$)')
plt.title('logM$_{BH}$=6, $\gamma$=-1.9, log($\\rho_{10pc}$)=3.5 M$_\odot$/pc$^3$')
plt.show()

#%%




plt.figure(dpi=500)
plt.scatter(slopes,TDE_rates_single,c=np.log10(rho_5pc),cmap='viridis')

# =============================================================================
# for i in range(len(names)):
#     plt.plot(slopes[i],TDE_rates_single[i],
#              color=cmap((float(i)+1)/len(names)),linestyle='',marker='o')
# =============================================================================

#plt.ylim(-25,-10)
#plt.ylim(5e-5,2e-4)
#plt.xlim(7.5,14.5)
#plt.legend()
plt.xlim(0.45,2.25)
plt.colorbar(label='log($\\rho_{5pc}$ [M$_\odot$/pc$^3$])')
plt.yscale('log')
plt.ylabel('$\dot N_{TDE}~[yr^{-1}])$')
plt.xlabel('$\gamma$')
#plt.title('logM$_{BH}$=6, $\gamma$=-1.9, log($\\rho_{10pc}$)=3.5 M$_\odot$/pc$^3$')
plt.show()

#%%
plt.figure(dpi=500)
#plt.plot(np.log10(r*3.24078e-17),np.log10(psi_1+psi_2+psi_3),'k')
# =============================================================================
# for i in range(len(names)):
#     plt.plot(np.log10(r/pc_to_m),np.log10(pots[i,0,:]),label='{} pc'.format(wids_pc[i]),
#              color=cmap((float(i)+1)/len(names)),linewidth=3,alpha=0.5)
# =============================================================================
for i in range(len(names)):
    plt.plot(np.log10(rs[i,:]/pc_to_m),np.log10(psi_encs[i,:]),linestyle='--',
             color=cmap((float(i)+1)/len(names)))
for i in range(len(names)):
    plt.plot(np.log10(rs[i,:]/pc_to_m),np.log10(psi_exts[i,:]),linestyle='-',
             color=cmap((float(i)+1)/len(names)))
for i in range(len(names)):
    plt.plot(np.log10(rs[i,:]/pc_to_m),np.log10(psi_bhs[i,:]),linestyle=':',
             color=cmap((float(i)+1)/len(names)))
plt.plot(0,0,linestyle='--',label='$\psi_{enc}$',color='k')
plt.plot(0,0,linestyle='-',label='$\psi_{ext}$',color='k')
plt.plot(np.log10(rs[0,:]/pc_to_m),np.log10(psi_bhs[0,:]),label='$\psi_{BH}$',color='k',linestyle=':')
plt.legend()
#plt.ylim(np.min(np.log10(psi_bhs[0,:])),np.max(np.log10(psi_bhs[0,:])))
plt.ylim(-5,15)
plt.xlim(-6,15)
plt.xlabel('log(Radius [pc])')
plt.ylabel('log($\psi(r)$ [m$^2$/s$^2$])')
plt.title('logM$_{BH}$=6, $\gamma$=-1.9, log($\\rho_{10pc}$)=3.5 M$_\odot$/pc$^3$')
plt.show()

#%%
plt.figure(dpi=500)
#plt.plot(np.log10(r*3.24078e-17),np.log10(psi_1+psi_2+psi_3),'k')
for i in range(len(names)):
    plt.plot(np.log10(rs[i,:]/pc_to_m),np.log10(psi_tots[i,:]),
             color=cmap((float(i)+1)/len(names)))
plt.legend()
#plt.ylim(np.min(np.log10(psi_bhs[0,:])),np.max(np.log10(psi_bhs[0,:])))
plt.ylim(-5,15)
plt.xlim(-6,15)
plt.xlabel('log(Radius [pc])')
plt.ylabel('log($\psi(r)$ [m$^2$/s$^2$])')
plt.title('logM$_{BH}$=6, $\gamma$=-1.9, log($\\rho_{10pc}$)=3.5 M$_\odot$/pc$^3$')
plt.show()
