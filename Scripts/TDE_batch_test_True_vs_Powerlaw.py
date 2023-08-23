#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jun 22 13:38:56 2023

This code executes TDE rate calculations for two density profiles (one true density profile and on parameterized version)

@author: christian
"""

import TDE_rate_modeling_batch_discrete as TDE_disc
import TDE_rate_modeling_batch as TDE
import numpy as np
import matplotlib.pyplot as plt
from astropy.io import fits
from scipy import integrate
from tqdm import tqdm
import TDE_util as tu
from astropy.table import Table, vstack
import multiprocessing as mp
import time
import istarmap
import warnings
warnings.filterwarnings("ignore")
import mge1d_util as u
from os import getpid
import psutil
import os
import subprocess
from scipy.optimize import curve_fit
from tqdm import tqdm
TDE_plot_dir = '../Plots/TDE_software_plots/Outer_Slope_Tests/'
TDE_results_dir = '../Result_Tables/TDE_tables/Outer_Slope_Test_Results/'
import pdb

# CONSTANTS
pc_to_m = 3.08567758128e16 
M_sol =  1.989e30 # kg
R_sol = 696.34e6 # m
G = 6.6743e-11 # m^3 s^-2 kg^-1


toy_model_match = False

#%%

slope_ext = '2x_pixel_scale'
phys_ext = '_or_10pc_extinction_corr_nsa_ml_w_vi_color'

gal_file = '../Result_Tables/all_gal_data_'+slope_ext+phys_ext+'.fits'
hdul = fits.open(gal_file)
head = hdul[0].data
dat = hdul[1].data
hdul.close()

names = dat['name']
vmags = dat['vmag']
dists = dat['dist'] # Mpc
MLs = dat['ml']
ML_types = dat['ml_type']
slopes = dat['slope']
cen_dens = dat['dens_at_5pc'] # M_sol/pc^3
lograds = dat['lograd'] # log(pc)
logdens = dat['logdens'] # log(M_sol/pc^3)
stone_slopes = dat['stone_slope']
stone_dens_at_5pc = dat['stone_dens_at_5pc']
NSC_comp = dat['NSC_comp']
all_MLs = dat['all_mls']
all_ML_types = dat['all_ml_types']
dist_flags = dat['dist_flag']
SBs = dat['SBs']
MGEs = dat['MGEs']
filts = dat['filt']
sample = dat['sample']

# isolate NGC 1023
ind = 3
name = names[ind]
slope = np.abs(slopes[ind])
rho_5pc = cen_dens[ind] # M_sol/pc^3
r_b_pc = 1.77 # pc, median resolution limit of sample

# convert profile to broken power-law w/ B-W cusp inward of r_b
r_b = r_b_pc*pc_to_m # m
rho_b = rho_5pc*(r_b_pc/5)**(-slope)*M_sol/pc_to_m**3 #kg/m^3
smooth = 0.1

decay_params = np.array([1.0611808940725666e+19,1.0611808940725666e+19])
decay_params_pc = decay_params/pc_to_m


M_BH = 10**6*M_sol

rads = 10**lograds[ind,:]*pc_to_m # m
dens = 10**logdens[ind,:]*(M_sol/pc_to_m**3) # kg/m^3
#%%

results = TDE.get_TDE_rate(name,slope,r_b,rho_b,smooth,M_BH,decay_params)


#%%

# =============================================================================
# radys = results['psi_rads'].value[0] 
# dens = tu.get_rho_r_new(radys,slope,rho_b,r_b,smooth,decay_params,toy_model_match)
# =============================================================================
results_disc = TDE_disc.get_TDE_rate(name,rads,dens,M_BH)

#%%


rad_disc = results_disc['radii'].value[0]
dens_disc = results_disc['dens'].value[0]

radys_disc = results_disc['psi_rads'].value[0] 
psi_bh_disc = results_disc['psi_bh'].value[0] 
psi_enc_disc = results_disc['psi_enc'].value[0] 
psi_ext_disc = results_disc['psi_ext'].value[0] 
psi_tot_disc = results_disc['psi_tot'].value[0]
epsilon_disc = results_disc['orb_ens'].value[0]
qs_disc = results_disc['q'].value[0]
DFs_disc = results_disc['DF'].value[0]
LC_flux_solar_disc = results_disc['LC_flux_solar'].value[0] 
TDE_rate_disc = results_disc['TDE_rate_full'].value[0]

radys = results['psi_rads'].value[0]  
psi_bh = results['psi_bh'].value[0]  
psi_enc = results['psi_enc'].value[0]  
psi_ext = results['psi_ext'].value[0]  
psi_tot = results['psi_tot'].value[0] 
densys = tu.get_rho_r_new(radys,slope,rho_b,r_b,smooth,decay_params,toy_model_match)
epsilon = results['orb_ens'].value[0] 
qs = results['q'].value[0] 
DFs = results['DF'].value[0] 
LC_flux_solar = results['LC_flux_solar'].value[0] 
TDE_rate = results['TDE_rate_full'].value[0]

print('Discrete TDE Rate: {:.2f}'.format(np.log10(TDE_rate_disc)))
print('Parameterized TDE Rate: {:.2f}'.format(np.log10(TDE_rate)))
#%%
cmap=plt.get_cmap("turbo")

# let's do some plotting here
# plot the density profiles
plt.figure(dpi=500)
#plt.title('Inner $\gamma$:-{:.1f}, Outer $\gamma$:-{:.1f}'.format(inner_slope,outer_slopes[i]))
plt.xlabel('log(Radius [pc])')
plt.ylabel('log($\\rho(r)~[kg/m^3]$)')
plt.plot(np.log10(radys/pc_to_m),np.log10(densys),color=cmap(0.1),marker='*',label='Parameterization')
plt.plot(np.log10(rad_disc/pc_to_m),np.log10(dens_disc),color=cmap(0.7),label='Discrete')
plt.legend()
plt.xlim(-0.8,4)
plt.ylim(-30,-10)

#plt.savefig(TDE_plot_dir+'{:.1f}s_{:.1f}d_{:.1f}mbh_{}_outer_slopes_densities.png'.format(inner_slope,log_rho5pc_msol_pc3,log_M_BH_msol,num_runs),
#             bbox_inches='tight', pad_inches=0.1, dpi=500)

#%%
# plot the potentials
plt.figure(dpi=500)
#plt.title('Inner $\gamma$:-{:.1f}, Outer $\gamma$:-{:.1f}'.format(inner_slope,outer_slopes[i]))
plt.xlabel('log(Radius [pc])')
plt.ylabel('log($\psi(r)$ [m$^2$/s$^2$])')

plt.plot(np.log10(radys/pc_to_m),np.log10(psi_tot),color=cmap(0.1),marker='*', label='Parameterization')
plt.plot(np.log10(rad_disc/pc_to_m),np.log10(psi_tot_disc),color=cmap(0.7),label='Discrete')
plt.legend()
#plt.savefig(TDE_plot_dir+'{:.1f}s_{:.1f}d_{:.1f}mbh_{}_outer_slopes_potentials.png'.format(inner_slope,log_rho5pc_msol_pc3,log_M_BH_msol,num_runs),
#             bbox_inches='tight', pad_inches=0.1, dpi=500)

plt.ylim(7.5,15)
plt.xlim(-5,10)


#%%

# plot the distribution functions
plt.figure(dpi=500)
#plt.title('Inner $\gamma$:-{:.1f}, Outer $\gamma$:-{:.1f}'.format(inner_slope,outer_slopes[i]))
plt.ylabel('log(f($\epsilon$))')
plt.xlabel('log($\epsilon$)')
plt.plot(np.log10(epsilon),np.log10(DFs),color=cmap(0.1), marker='*',label='Parameterization')
plt.plot(np.log10(epsilon),np.log10(DFs_disc),color=cmap(0.7),label='Discrete')
plt.legend()
#plt.savefig(TDE_plot_dir+'{:.1f}s_{:.1f}d_{:.1f}mbh_{}_outer_slopes_DFs.png'.format(inner_slope,log_rho5pc_msol_pc3,log_M_BH_msol,num_runs),
#             bbox_inches='tight', pad_inches=0.1, dpi=500)
plt.ylim(-67,-55)
plt.xlim(8,14.8)


#%%

# plot the diffusion coeffs
plt.figure(dpi=500)
plt.ylabel('log(q)')
plt.xlabel('log($\epsilon$)')
plt.plot(np.log10(epsilon),np.log10(qs),color=cmap(0.1), marker='*',label='Parameterization')
plt.plot(np.log10(epsilon),np.log10(qs_disc),color=cmap(0.7),label='Discrete')
plt.legend()
#plt.ylim(-80,-58)
#plt.savefig(TDE_plot_dir+'{:.1f}s_{:.1f}d_{:.1f}mbh_{}_outer_slopes_qs.png'.format(inner_slope,log_rho5pc_msol_pc3,log_M_BH_msol,num_runs),
#             bbox_inches='tight', pad_inches=0.1, dpi=500)


#%%

# plot the LC flux for solar mass stars
plt.figure(dpi=500)
plt.ylabel('$log(\mathcal{F}(\epsilon)$)')
plt.xlabel('log($\epsilon$)')
plt.plot(np.log10(epsilon),np.log10(LC_flux_solar),color=cmap(0.1),marker='*', label='Parameterization')
plt.plot(np.log10(epsilon),np.log10(LC_flux_solar_disc),color=cmap(0.7),label='Discrete')
plt.legend(fontsize='x-small')
plt.ylim(-50,-20)
#plt.savefig(TDE_plot_dir+'{:.1f}s_{:.1f}d_{:.1f}mbh_{}_outer_slopes_LC_fluxes.png'.format(inner_slope,log_rho5pc_msol_pc3,log_M_BH_msol,num_runs),
#             bbox_inches='tight', pad_inches=0.1, dpi=500)



