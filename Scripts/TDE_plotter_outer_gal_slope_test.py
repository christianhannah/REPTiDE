#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu May 18 23:34:05 2023

@author: christian
"""
import TDE_rate_batch_modeling_dbl_brkn_powerlaw as TDE
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

import glob

from os import getpid
import psutil

TDE_results_dir = '../Result_Tables/TDE_tables/Outer_Slope_Test_Results/'
TDE_plot_dir = '../Plots/TDE_software_plots/Outer_Slope_Tests/'


#%% 
# =============================================================================
# this part is for results from a single density profile 
# =============================================================================


filename = '1.6s_3.8d_6.0mbh_8_outer_slopes.fits'

log_M_BH_msol = float(filename[10:13])
log_rho5pc_msol_pc3 = float(filename[5:8])
num_runs = 8

output_table = Table.read(TDE_results_dir+filename)

# CONSTANTS
pc_to_m = 3.08567758128e16 
M_sol =  1.989e30 # kg
R_sol = 696.34e6 # m
G = 6.6743e-11 # m^3 s^-2 kg^-1

outer_slopes = output_table['outer_slope']
rho_break = output_table['rho_break'][0]
r_break = output_table['r_break'][0]
smooth = output_table['smooth'][0]
decay_params = output_table['decay_params'][0]
decay_params_pc = decay_params/pc_to_m
toy_model_match = False
inner_slope = output_table['inner_slope'][0]


# let's do some plotting here

# plot the density profiles
plt.figure(dpi=500)
#plt.title('Inner $\gamma$:-{:.1f}, Outer $\gamma$:-{:.1f}'.format(inner_slope,outer_slopes[i]))
plt.xlabel('log(Radius [pc])')
plt.ylabel('log($\\rho(r)~[kg/m^3]$)')
for i in range(len(output_table)):
    radys = np.geomspace(10**-3,10**16,10**4)*pc_to_m
    dens = tu.get_rho_r_nuker(radys,inner_slope,outer_slopes[i],rho_break,r_break,smooth,decay_params,toy_model_match)

    cmap=plt.get_cmap("turbo")
    indy_50 = tu.find_nearest(radys, r_break)
    indy_exp = tu.find_nearest(radys/pc_to_m, decay_params_pc[0])
    plt.plot(np.log10(radys[0:indy_50]/pc_to_m),np.log10(dens[0:indy_50]),color=cmap(0.1))
    plt.plot(np.log10(radys[indy_50:indy_exp]/pc_to_m),np.log10(dens[indy_50:indy_exp]),color=cmap(0.7))
    plt.plot(np.log10(radys[indy_exp:]/pc_to_m),np.log10(dens[indy_exp:]),color=cmap(1.5))
    #plt.xlim(-3,9)
    plt.ylim(-60,-7)

plt.savefig(TDE_plot_dir+'{:.1f}s_{:.1f}d_{:.1f}mbh_{}_outer_slopes_densities.png'.format(inner_slope,log_rho5pc_msol_pc3,log_M_BH_msol,num_runs),
             bbox_inches='tight', pad_inches=0.1, dpi=500)

#%%
# plot the potentials
plt.figure(dpi=500)
#plt.title('Inner $\gamma$:-{:.1f}, Outer $\gamma$:-{:.1f}'.format(inner_slope,outer_slopes[i]))
plt.xlabel('log(Radius [pc])')
plt.ylabel('log($\psi(r)$ [m$^2$/s$^2$])')
for i in range(len(output_table)):
    radys = output_table['psi_rads'][i] 
    psi_bh = output_table['psi_bh'][i] 
    psi_enc = output_table['psi_enc'][i] 
    psi_ext = output_table['psi_ext'][i] 
    psi_tot = output_table['psi_tot'][i]
    
# =============================================================================
#     plt.plot(np.log10(r_ls/pc_to_m),np.log10(psi_bh),label='$\psi_{BH}$')
#     plt.plot(np.log10(r_ls/pc_to_m),np.log10(psi_enc),label='$\psi_{enc}$')
#     plt.plot(np.log10(r_ls/pc_to_m),np.log10(psi_ext),label='$\psi_{ext}$')
# =============================================================================
    plt.plot(np.log10(radys/pc_to_m),np.log10(psi_tot),color=cmap((float(i)+1)/len(output_table)), label='Outer $\gamma$ = {:.1f}'.format(outer_slopes[i]))

plt.legend(fontsize='x-small')
plt.savefig(TDE_plot_dir+'{:.1f}s_{:.1f}d_{:.1f}mbh_{}_outer_slopes_potentials.png'.format(inner_slope,log_rho5pc_msol_pc3,log_M_BH_msol,num_runs),
             bbox_inches='tight', pad_inches=0.1, dpi=500)

# zoomed
plt.figure(dpi=500)
#plt.title('Inner $\gamma$:-{:.1f}, Outer $\gamma$:-{:.1f}'.format(inner_slope,outer_slopes[i]))
plt.xlabel('log(Radius [pc])')
plt.ylabel('log($\psi(r)$ [m$^2$/s$^2$])')
for i in range(len(output_table)):
    radys = output_table['psi_rads'][i] 
    psi_bh = output_table['psi_bh'][i] 
    psi_enc = output_table['psi_enc'][i] 
    psi_ext = output_table['psi_ext'][i] 
    psi_tot = output_table['psi_tot'][i] 
    
# =============================================================================
#     plt.plot(np.log10(r_ls/pc_to_m),np.log10(psi_bh),label='$\psi_{BH}$')
#     plt.plot(np.log10(r_ls/pc_to_m),np.log10(psi_enc),label='$\psi_{enc}$')
#     plt.plot(np.log10(r_ls/pc_to_m),np.log10(psi_ext),label='$\psi_{ext}$')
# =============================================================================
    plt.plot(np.log10(radys/pc_to_m),np.log10(psi_tot),color=cmap((float(i)+1)/len(output_table)), label='Outer $\gamma$ = {:.1f}'.format(outer_slopes[i]))

plt.ylim(7.5,15)
plt.xlim(-5,10)
plt.legend(fontsize='x-small')
plt.savefig(TDE_plot_dir+'{:.1f}s_{:.1f}d_{:.1f}mbh_{}_outer_slopes_potentials_zoom.png'.format(inner_slope,log_rho5pc_msol_pc3,log_M_BH_msol,num_runs),
             bbox_inches='tight', pad_inches=0.1, dpi=500)


#%%
# plot the solar TDE rate vs. the outer slope
plt.figure(dpi=500)
#plt.title('Inner $\gamma$:-{:.1f}, Outer $\gamma$:-{:.1f}'.format(inner_slope,outer_slopes[i]))
plt.ylabel('$\dot N_{TDE,\odot}~[yr^{-1}])$')
plt.xlabel('Outer $\gamma$')
# # for i in range(len(cens)):
# #     plt.plot(np.log10(wids_pc[i]),TDE_rates_single[i],label='10$^{:.0F}$ pc'.format(np.log10(wids_pc[i])),
# #              color=cmap((float(i)+1)/len(cens)),linestyle='',marker='o')
# # 
for i in range(len(output_table)):
    plt.plot(outer_slopes[i],output_table['TDE_rate_solar'][i],
             color=cmap((float(i)+1)/len(output_table)),linestyle='',marker='o')
plt.yscale('log')

plt.savefig(TDE_plot_dir+'{:.1f}s_{:.1f}d_{:.1f}mbh_{}_outer_slopes_TDE_rates_solar.png'.format(inner_slope,log_rho5pc_msol_pc3,log_M_BH_msol,num_runs),
             bbox_inches='tight', pad_inches=0.1, dpi=500)


#%%

# plot the distribution functions
plt.figure(dpi=500)
#plt.title('Inner $\gamma$:-{:.1f}, Outer $\gamma$:-{:.1f}'.format(inner_slope,outer_slopes[i]))
plt.ylabel('log(f($\epsilon$))')
plt.xlabel('log($\epsilon$)')
for i in range(len(output_table)):
    epsilon = output_table['orb_ens'][i] 
    DF = output_table['DF'][i] 
    plt.plot(np.log10(epsilon),np.log10(DF),color=cmap((float(i)+1)/len(output_table)), label='Outer $\gamma$ = {:.1f}'.format(outer_slopes[i]))

plt.legend(fontsize='x-small')
plt.ylim(-80,-55)
plt.savefig(TDE_plot_dir+'{:.1f}s_{:.1f}d_{:.1f}mbh_{}_outer_slopes_DFs.png'.format(inner_slope,log_rho5pc_msol_pc3,log_M_BH_msol,num_runs),
             bbox_inches='tight', pad_inches=0.1, dpi=500)

# zoomed
plt.figure(dpi=500)
#plt.title('Inner $\gamma$:-{:.1f}, Outer $\gamma$:-{:.1f}'.format(inner_slope,outer_slopes[i]))
plt.ylabel('log(f($\epsilon$))')
plt.xlabel('log($\epsilon$)')
for i in range(len(output_table)):
    epsilon = output_table['orb_ens'][i] 
    DF = output_table['DF'][i] 
    plt.plot(np.log10(epsilon),np.log10(DF),color=cmap((float(i)+1)/len(output_table)), label='Outer $\gamma$ = {:.1f}'.format(outer_slopes[i]))

plt.legend(fontsize='x-small')
plt.ylim(-67,-55)
plt.xlim(8,14.8)
plt.savefig(TDE_plot_dir+'{:.1f}s_{:.1f}d_{:.1f}mbh_{}_outer_slopes_DFs_zoom.png'.format(inner_slope,log_rho5pc_msol_pc3,log_M_BH_msol,num_runs),
             bbox_inches='tight', pad_inches=0.1, dpi=500)


#%%

# plot the diffusion coeffs
plt.figure(dpi=500)
plt.ylabel('log(q)')
plt.xlabel('log($\epsilon$)')
for i in range(len(output_table)):
    epsilon = output_table['orb_ens'][i] 
    qs = output_table['q'][i] 
    plt.plot(np.log10(epsilon),np.log10(qs),color=cmap((float(i)+1)/len(output_table)), label='Outer $\gamma$ = {:.1f}'.format(outer_slopes[i]))

plt.legend(fontsize='x-small')
#plt.ylim(-80,-58)
plt.savefig(TDE_plot_dir+'{:.1f}s_{:.1f}d_{:.1f}mbh_{}_outer_slopes_qs.png'.format(inner_slope,log_rho5pc_msol_pc3,log_M_BH_msol,num_runs),
             bbox_inches='tight', pad_inches=0.1, dpi=500)


#%%

# plot the LC flux for solar mass stars
plt.figure(dpi=500)
plt.ylabel('$log(\mathcal{F}(\epsilon)$)')
plt.xlabel('log($\epsilon$)')
for i in range(len(output_table)):
    epsilon = output_table['orb_ens'][i] 
    LC_flux_solar = output_table['LC_flux_solar'][i] 
    plt.plot(np.log10(epsilon),np.log10(LC_flux_solar),color=cmap((float(i)+1)/len(output_table)), label='Outer $\gamma$ = {:.1f}'.format(outer_slopes[i]))

plt.legend(fontsize='x-small')
plt.ylim(-50,-20)
plt.savefig(TDE_plot_dir+'{:.1f}s_{:.1f}d_{:.1f}mbh_{}_outer_slopes_LC_fluxes.png'.format(inner_slope,log_rho5pc_msol_pc3,log_M_BH_msol,num_runs),
             bbox_inches='tight', pad_inches=0.1, dpi=500)

#zoomed
plt.figure(dpi=500)
plt.ylabel('$log(\mathcal{F}(\epsilon)$)')
plt.xlabel('log($\epsilon$)')
for i in range(len(output_table)):
    epsilon = output_table['orb_ens'][i] 
    LC_flux_solar = output_table['LC_flux_solar'][i] 
    plt.plot(np.log10(epsilon),np.log10(LC_flux_solar),color=cmap((float(i)+1)/len(output_table)), label='Outer $\gamma$ = {:.1f}'.format(outer_slopes[i]))

plt.legend(fontsize='x-small')
plt.ylim(-30,-21)

plt.savefig(TDE_plot_dir+'{:.1f}s_{:.1f}d_{:.1f}mbh_{}_outer_slopes_LC_fluxes_zoom.png'.format(inner_slope,log_rho5pc_msol_pc3,log_M_BH_msol,num_runs),
             bbox_inches='tight', pad_inches=0.1, dpi=500)

# =============================================================================
# =============================================================================
# =============================================================================
#%%


# =============================================================================
# This part combines results from all different density profiles.
# =============================================================================
import pdb

filenames = glob.glob('../Result_Tables/TDE_tables/Outer_Slope_Test_Results/*.fits')


outer_slopes_i = np.array([2.0,2.4285714285714284,2.857142857142857,3.2857142857142856,
                         3.7142857142857144,4.142857142857142,4.571428571428571,5.0])

rho_breaks = np.zeros((len(outer_slopes),len(filenames)))
r_breaks = np.zeros((len(outer_slopes),len(filenames)))
smooths = np.zeros((len(outer_slopes),len(filenames)))
decay_params = np.zeros((len(outer_slopes),2,len(filenames)))
inner_slopes = np.zeros((len(outer_slopes),len(filenames)))
M_BHs = np.zeros((len(outer_slopes),len(filenames)))
TDE_rates_solar = np.zeros((len(outer_slopes),len(filenames)))
TDE_rates_full = np.zeros((len(outer_slopes),len(filenames)))
outer_slopes = np.zeros((len(outer_slopes),len(filenames)))

for i, name in enumerate(filenames):
    output_table = Table.read(name)
    outer_slopes[:,i] = output_table['outer_slope'].value
    
    if (outer_slopes_i != outer_slopes[:,i]).any():
        print("ERROR: Outer slopes don't match... ")
        pdb.set_trace()
        
    rho_breaks[:,i] = output_table['rho_break'].value
    r_breaks[:,i] = output_table['r_break'].value
    smooths[:,i] = output_table['smooth'].value
    decay_params[:,:,i] = output_table['decay_params'].value
    decay_params_pc = decay_params/pc_to_m
    inner_slopes[:,i] = output_table['inner_slope'].value
    M_BHs[:,i] = output_table['M_BH'].value
    TDE_rates_solar[:,i] = output_table['TDE_rate_solar'].value
    TDE_rates_full[:,i] = output_table['TDE_rate_full'].value




# let's do some plotting
for i in range(len(outer_slopes_i)):
    fig, ax = plt.subplots(nrows=1,ncols=1,dpi=600)
    fig.suptitle('-{:.2f} outer slope'.format(outer_slopes_i[i]))
    scat = ax.scatter(inner_slopes[i,:], np.log10(TDE_rates_solar[i,:]),
                      c=np.log10(rho_breaks[i,:]/(M_sol/pc_to_m**3)),cmap='viridis')
    ax.set_ylabel('log($\dot N_{TDE,\odot}~[yr^{-1}]))$')
    ax.set_xlabel('Inner Slope')
    fig.colorbar(scat)
    
#%%   
# let's put the data in a 3d scatter plot showing inner slope, rho, and tde rate colored by scatter

TDE_rates_solar[np.isnan(TDE_rates_solar)] = 0

scats = np.zeros((len(filenames)))
meds = np.zeros((len(filenames)))
points = np.zeros((len(filenames),3))
mbhs = np.zeros((len(filenames)))
for i in range(len(scats)):
    meds[i] = np.log10(np.median(TDE_rates_solar[:,i]))
    scats[i] = np.std(TDE_rates_solar[:,i])/np.median(TDE_rates_solar[:,i])
    points[i,0] = inner_slopes[0,i]
    points[i,1] = np.log10(rho_breaks[0,i]/(M_sol/pc_to_m**3))
    points[i,2] = meds[i]
    mbhs[i] = M_BHs[0,i]


bh6 = np.where(mbhs == 1.989e+36)[0]
bh9 = np.where(mbhs == 1.989e+39)[0]

plt.figure(dpi=500)
plt.title('M$_{BH}$ = 10$^6$ M$_\odot$')
plt.scatter(points[bh6,0], meds[bh6], c=scats[bh6])
plt.colorbar(label='$\sigma_{\dot N_{TDE}}/\dot N_{TDE}$')
plt.xlabel('Inner $\gamma$')
plt.ylabel('log($\dot N_{TDE,\odot}~[yr^{-1}]))$')

plt.figure(dpi=500)
plt.title('M$_{BH}$ = 10$^9$ M$_\odot$')
plt.scatter(points[bh9,0], meds[bh9], c=scats[bh9])
plt.colorbar(label='$\sigma_{\dot N_{TDE}}/\dot N_{TDE}$')
plt.xlabel('Inner $\gamma$')
plt.ylabel('log($\dot N_{TDE,\odot}~[yr^{-1}]))$')


plt.figure(dpi=500)
plt.title('M$_{BH}$ = 10$^6$ M$_\odot$')
plt.scatter(points[bh6,1], meds[bh6], c=scats[bh6])
plt.colorbar(label='$\sigma_{\dot N_{TDE}}/\dot N_{TDE}$')
plt.xlabel('log($\\rho_{50pc}$ [M$_\odot$/pc$^3$])')
plt.ylabel('log($\dot N_{TDE,\odot}~[yr^{-1}]))$')

plt.figure(dpi=500)
plt.title('M$_{BH}$ = 10$^9$ M$_\odot$')
plt.scatter(points[bh9,1], meds[bh9], c=scats[bh9])
plt.colorbar(label='$\sigma_{\dot N_{TDE}}/\dot N_{TDE}$')
plt.xlabel('log($\\rho_{50pc}$ [M$_\odot$/pc$^3$])')
plt.ylabel('log($\dot N_{TDE,\odot}~[yr^{-1}]))$')



#%%
points[np.isinf(points[:,2]),2] = 0
scats[np.isinf(scats)] = 0


plt.figure(dpi=600,figsize=(8,6))
ax = plt.axes(projection='3d')
scat = ax.scatter3D(points[:,0],points[:,1],points[:,2],s=40,c=scats,cmap='viridis')
fig.colorbar(scat,label='$\sigma_{\dot N_{TDE}}$',pad=0.1)#,orientation='horizontal')
ax.set_xlabel('Inner $\gamma$')
ax.set_ylabel('log($\\rho_{50pc}$ [M$_\odot$/pc$^3$])')
ax.set_zlabel('log($\dot N_{TDE,\odot}~[yr^{-1}]))$')
# uncomment below to alter viewing angle
ax.view_init(0,0)

# =============================================================================
# =============================================================================
# =============================================================================


