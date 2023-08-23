#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Feb 20 18:27:11 2023

This code executes TDE rate calculations for a batch of galaxy density/BH parameters.

@author: christian
"""

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

# CONSTANTS
pc_to_m = 3.08567758128e16 
M_sol =  1.989e30 # kg
R_sol = 696.34e6 # m
G = 6.6743e-11 # m^3 s^-2 kg^-1


toy_model_match = False

# =============================================================================
# # === DEFINE BATCH OF DENSITY PARAMETERS W/ VARYING EFFECTIVE RADII =========
# 
# num_runs = 12
# 
# names = np.zeros(num_runs)
# slopes = np.ones_like(names)*1.9
# rho_infls = np.ones_like(names)*3e3*M_sol/pc_to_m**3 #kg/m^3
# r_infls = np.ones_like(names)*10*pc_to_m # m
# bh_masses = np.ones_like(names)*10**6*M_sol
# 
# # convert profile to broken power-law w/ B-W cusp inward of r_b
# r_bs = 10e-6*pc_to_m # m
# rho_bs = rho_infls*(r_bs/r_infls)**(-slopes)
# smooth = 0.1
# 
# wids_pc = np.geomspace(10,10.0**8,num_runs)
# wids = wids_pc*pc_to_m # m, radius to begin decay
# cens = np.ones(len(wids))*10*pc_to_m # m, width (halflife) of decay
# decay_params = np.array([cens,wids])
# 
# =============================================================================


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


num_runs = len(names)-30

# =============================================================================



#%%
# ========================= RUN TDE CODE FOR BATCH ============================

if __name__ == '__main__':
    
    inputs = []
    for i in range(num_runs):
        inputs.append((names[i],slopes[i],r_bs,rho_bs[i],smooth,
                       bh_masses[i],decay_params))#[:,i]))

    start = time.time()
    print()
    print('Beginning TDE rate computation...')
    print('Batch Info: ')
    print('\t # of Processes: {}'.format(mp.cpu_count()))
    print('\t # of Runs: {}'.format(num_runs))
    print()
    with mp.Pool(mp.cpu_count()) as p:
        results = list(tqdm(p.istarmap(TDE.get_TDE_rate, inputs), 
                            total=len(inputs)))
        p.close()
        p.join()
    
    end = time.time()
    
    output_table = Table()
    for r in results:
        output_table = vstack([output_table,r])

    print()    
    print('Done.')
    print()
    print('Runtime: {:.2f} minutes / {:.2f} hours'.format(round(end-start,3)/60,
                                                          round(end-start,3)/3600))
    print()



#%%
# =============================================================================
# cmap=plt.get_cmap("turbo")
# plt.figure(dpi=500)
# #plt.plot(np.log10(r*3.24078e-17),np.log10(psi_1+psi_2+psi_3),'k')
# # =============================================================================
# # for i in range(len(output_table['names'])):
# #     plt.plot(np.log10(r/pc_to_m),np.log10(pots[i,0,:]),label='{} pc'.format(wids_pc[i]),
# #              color=cmap((float(i)+1)/len(output_table['names'])),linewidth=3,alpha=0.5)
# # =============================================================================
# # =============================================================================
# # for i in range(len(output_table['slope'])):
# #     plt.plot(np.log10(output_table['psi_rads'][i]/pc_to_m),np.log10(output_table['psi_enc'][i]*output_table['psi_rads'][i]/(G*M_sol)),linestyle='--',
# #              color=cmap((float(i)+1)/len(output_table['slope'])))
# # =============================================================================
# for i in range(len(output_table['slope'])):
#     plt.plot(np.log10(output_table['psi_rads'][i]/pc_to_m),np.log10(output_table['psi_enc'][i]),linestyle='--',
#              color=cmap((float(i)+1)/len(output_table['slope'])))
# for i in range(len(output_table['slope'])):
#     plt.plot(np.log10(output_table['psi_rads'][i]/pc_to_m),np.log10(output_table['psi_ext'][i]),linestyle='-',
#              color=cmap((float(i)+1)/len(output_table['slope'])))
# # =============================================================================
# # for i in range(len(output_table['slope'])):
# #     plt.plot(np.log10(output_table['psi_rads'][i]/pc_to_m),np.log10(output_table['psi_bh'][i]),linestyle=':',
# #              color=cmap((float(i)+1)/len(output_table['slope'])))
# # =============================================================================
# plt.plot(0,0,linestyle='--',label='$\psi_{enc}$',color='k')
# plt.plot(0,0,linestyle='-',label='$\psi_{ext}$',color='k')
# plt.plot(np.log10(output_table['psi_rads'][0]/pc_to_m),np.log10(output_table['psi_bh'][0]),label='$\psi_{BH}$',color='k',linestyle=':')
# plt.legend()
# #plt.ylim(np.min(np.log10(psi_bhs[0,:])),np.max(np.log10(psi_bhs[0,:])))
# plt.ylim(-5,15)
# plt.xlim(-6,15)
# plt.ylabel('log($\psi(r)$ [m$^2$/s$^2$])')
# plt.xlabel('log(Radius [pc])')
# plt.title('logM$_{BH}$=6, $\gamma$=-1.9, log($\\rho_{10pc}$)=3.5 M$_\odot$/pc$^3$')
# plt.show()
# 
# 
# 
# # =============================================================================
# # =============================================================================
# # #%%
# # # convert TDE rates from per second to per year 
# # sec_per_yr = 3.154e+7
# # TDE_rates = TDE_rates*sec_per_yr
# # 
# # # compute the TDE rates for a pure population of solar mass stars 
# # TDE_rates_single = np.zeros_like(TDE_rates)
# # for i in range(len(TDE_rates_single)):
# #     TDE_rates_single[i] = integrate.trapz(LC_fluxes[i,:], orb_ens[i,:])*sec_per_yr
# # 
# # #%%
# # 
# # labelfontsize=20
# # tickfontsize=16
# # plt.rcParams['xtick.direction']='in'
# # plt.rcParams['ytick.direction']='in'
# # plt.rcParams['xtick.labelsize']=tickfontsize
# # plt.rcParams['ytick.labelsize']=tickfontsize
# # plt.rcParams['figure.figsize']=(8,6)
# # plt.rcParams['axes.titlesize'] = 20
# # plt.rcParams['axes.labelsize']=labelfontsize
# # plt.rcParams['ytick.major.width'] = 1.5
# # plt.rcParams['xtick.major.width'] = 1.5
# # plt.rcParams['ytick.major.size'] = 5
# # plt.rcParams['xtick.major.size'] = 5
# # plt.rcParams['legend.fontsize'] = 15
# # 
# # 
# # cmap=plt.get_cmap("turbo")
# # 
# # plt.figure(dpi=500)
# # for i in range(len(cens)):
# #     plt.plot(np.log10(orb_ens[i,:]),np.log10(DFs[i,:]),label='10$^{:.0F}$ pc'.format(np.log10(wids_pc[i])),
# #              color=cmap((float(i)+1)/len(cens)))
# # #plt.ylim(-25,-10)
# # plt.ylim(-60,-57.5)
# # plt.xlim(7.5,14.5)
# # plt.legend()
# # #plt.xlim(0.95,1.05)
# # plt.ylabel('log(f($\epsilon$))')
# # plt.xlabel('log($\epsilon$)')
# # plt.title('logM$_{BH}$=6, $\gamma$=-1.9, log($\\rho_{10pc}$)=3.5 M$_\odot$/pc$^3$')
# # plt.show()
# # 
# # plt.figure(dpi=500)
# # for i in range(len(cens)):
# #     plt.plot(np.log10(orb_ens[i,:]),np.log10(qs[i,:]),label='10$^{:.0F}$ pc'.format(np.log10(wids_pc[i])),
# #              color=cmap((float(i)+1)/len(cens)))
# # #plt.ylim(-25,-10)
# # plt.ylim(-9,6)
# # plt.legend()
# # plt.xlim(7.5,14.5)
# # plt.ylabel('log(q)')
# # plt.xlabel('log($\epsilon$)')
# # plt.title('logM$_{BH}$=6, $\gamma$=-1.9, log($\\rho_{10pc}$)=3.5 M$_\odot$/pc$^3$')
# # plt.show()
# # 
# # 
# # plt.figure(dpi=500)
# # for i in range(len(cens)):
# #     plt.plot(np.log10(orb_ens[i,:]),np.log10(LC_fluxes[i,:]),label='10$^{:.0F}$ pc'.format(np.log10(wids_pc[i])),
# #              color=cmap((float(i)+1)/len(cens)))
# # #plt.ylim(-25,-10)
# # plt.ylim(-26,-23)
# # plt.xlim(7.5,14.5)
# # plt.legend()
# # #plt.xlim(0.95,1.05)
# # plt.ylabel('$log(\mathcal{F}(\epsilon)$)')
# # plt.xlabel('log($\epsilon$)')
# # plt.title('logM$_{BH}$=6, $\gamma$=-1.9, log($\\rho_{10pc}$)=3.5 M$_\odot$/pc$^3$')
# # plt.show()
# # 
# # #%%
# # 
# # plt.figure(dpi=500)
# # for i in range(len(cens)):
# #     plt.plot(np.log10(wids_pc[i]),TDE_rates_single[i],label='10$^{:.0F}$ pc'.format(np.log10(wids_pc[i])),
# #              color=cmap((float(i)+1)/len(cens)),linestyle='',marker='o')
# # 
# # #plt.ylim(-25,-10)
# # #plt.ylim(5e-5,2e-4)
# # #plt.xlim(7.5,14.5)
# # #plt.legend()
# # #plt.xlim(0.95,1.05)
# # plt.yscale('log')
# # plt.ylabel('$\dot N_{TDE}~[yr^{-1}])$')
# # plt.xlabel('log($R_{eff}~[pc]$)')
# # plt.title('logM$_{BH}$=6, $\gamma$=-1.9, log($\\rho_{10pc}$)=3.5 M$_\odot$/pc$^3$')
# # plt.show()
# # 
# # #%%
# # plt.figure(dpi=500)
# # #plt.plot(np.log10(r*3.24078e-17),np.log10(psi_1+psi_2+psi_3),'k')
# # # =============================================================================
# # # for i in range(len(cens)):
# # #     plt.plot(np.log10(r/pc_to_m),np.log10(pots[i,0,:]),label='10$^{:.0F}$ pc'.format(np.log10(wids_pc[i])),
# # #              color=cmap((float(i)+1)/len(cens)),linewidth=3,alpha=0.5)
# # # =============================================================================
# # for i in range(len(cens)):
# #     plt.plot(np.log10(rs[i,:]/pc_to_m),np.log10(psi_encs[i,:]),linestyle='--',
# #              color=cmap((float(i)+1)/len(cens)))
# # for i in range(len(cens)):
# #     plt.plot(np.log10(rs[i,:]/pc_to_m),np.log10(psi_exts[i,:]),linestyle='-',
# #              color=cmap((float(i)+1)/len(cens)))
# # for i in range(len(cens)):
# #     plt.plot(np.log10(rs[i,:]/pc_to_m),np.log10(psi_bhs[i,:]),linestyle=':',
# #              color=cmap((float(i)+1)/len(cens)))
# # plt.plot(0,0,linestyle='--',label='$\psi_{enc}$',color='k')
# # plt.plot(0,0,linestyle='-',label='$\psi_{ext}$',color='k')
# # plt.plot(np.log10(rs[0,:]/pc_to_m),np.log10(psi_bhs[0,:]),label='$\psi_{BH}$',color='k',linestyle=':')
# # plt.legend()
# # #plt.ylim(np.min(np.log10(psi_bhs[0,:])),np.max(np.log10(psi_bhs[0,:])))
# # plt.ylim(-5,15)
# # plt.xlim(-6,15)
# # plt.xlabel('log(Radius [pc])')
# # plt.ylabel('log($\psi(r)$ [m$^2$/s$^2$])')
# # plt.title('logM$_{BH}$=6, $\gamma$=-1.9, log($\\rho_{10pc}$)=3.5 M$_\odot$/pc$^3$')
# # plt.show()
# # 
# # #%%
# # plt.figure(dpi=500)
# # #plt.plot(np.log10(r*3.24078e-17),np.log10(psi_1+psi_2+psi_3),'k')
# # for i in range(len(cens)):
# #     plt.plot(np.log10(rs[i,:]/pc_to_m),np.log10(psi_tots[i,:]),label='10$^{:.0F}$ pc'.format(np.log10(wids_pc[i])),
# #              color=cmap((float(i)+1)/len(cens)))
# # plt.legend()
# # #plt.ylim(np.min(np.log10(psi_bhs[0,:])),np.max(np.log10(psi_bhs[0,:])))
# # plt.ylim(-5,15)
# # plt.xlim(-6,15)
# # plt.xlabel('log(Radius [pc])')
# # plt.ylabel('log($\psi(r)$ [m$^2$/s$^2$])')
# # plt.title('logM$_{BH}$=6, $\gamma$=-1.9, log($\\rho_{10pc}$)=3.5 M$_\odot$/pc$^3$')
# # plt.show()
# # 
# # #%%
# # 
# # rhos = np.zeros_like(rs)
# # for i in range(len(cens)):
# #     rhos[i,:] = tu.get_rho_r_new(rs[i,:],slopes[i],r_bs[i],rho_bs[i],smooth,decay_params[:,i])
# # 
# # plt.figure(dpi=500)
# # #plt.plot(np.log10(r*3.24078e-17),np.log10(psi_1+psi_2+psi_3),'k')
# # for i in range(len(cens)):
# #     plt.plot(np.log10(psi_tots[i,:]),np.log10(rhos[i,:]),label='10$^{:.0F}$ pc'.format(np.log10(wids_pc[i])),
# #              color=cmap((float(i)+1)/len(cens)))
# # plt.legend()
# # #plt.ylim(np.min(np.log10(psi_bhs[0,:])),np.max(np.log10(psi_bhs[0,:])))
# # plt.ylim(-100,-10)
# # plt.ylabel('log($\\rho(r)$ [kg/m$^3$])')
# # plt.xlabel('log($\psi(r)$ [m$^2$/s$^2$])')
# # plt.title('logM$_{BH}$=6, $\gamma$=-1.9, log($\\rho_{10pc}$)=3.5 M$_\odot$/pc$^3$')
# # plt.show()
# # 
# # 
# # 
# # 
# # 
# # 
# # 
# # 
# # 
# # 
# # 
# # =============================================================================
# 
# 
# 
# =============================================================================
