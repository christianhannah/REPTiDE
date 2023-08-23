#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri May 12 15:12:20 2023

This code executes TDE rate calculations for a batch of galaxy density/BH parameters
designed to test the sensitivity of rates to the outer galaxy slopes.

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

from os import getpid
import psutil

TDE_plot_dir = '../Plots/TDE_software_plots/Outer_Slope_Tests/'
TDE_results_dir = '../Result_Tables/TDE_tables/Outer_Slope_Test_Results/'

# CONSTANTS
pc_to_m = 3.08567758128e16 
M_sol =  1.989e30 # kg
R_sol = 696.34e6 # m
G = 6.6743e-11 # m^3 s^-2 kg^-1


toy_model_match = False

# === DEFINE BATCH OF DENSITY PARAMETERS W/ VARYING EFFECTIVE RADII =========

num_runs = 8

M_BH = 10**6*M_sol
log_M_BH_msol = np.log10(M_BH/M_sol)
r_break = 50*pc_to_m # m

smooth = 0.1
decay_params_pc = np.array([1e4,1e4])
decay_params = decay_params_pc*pc_to_m

outer_slopes = np.linspace(2,5,num_runs)

inner_slope = 0.51
logrho5pc = 5.0
rho_5pc = 10.0**(logrho5pc)*M_sol/pc_to_m**3 #kg/m^3
r_5pc = 5*pc_to_m # m
log_rho5pc_msol_pc3 = np.log10(rho_5pc/M_sol*pc_to_m**3)


radys = np.geomspace(10**-3,10**16,10**4)*pc_to_m
rho_inits = rho_5pc*(radys/r_5pc)**(-inner_slope)
ind_50 = tu.find_nearest(radys/pc_to_m,5)
rho_break = rho_inits[ind_50] # kg/m^3

name = 'Test Galaxy'

#%%

import multiprocessing
import time


if __name__ == '__main__':
    
    inputs = []
    for i in range(num_runs):
        inputs.append(((name,inner_slope,outer_slopes[i],r_break,rho_break,smooth,M_BH,
                       decay_params)))

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


output_table.write(TDE_results_dir+'{:.1f}s_{:.1f}d_{:.1f}mbh_{}_outer_slopes.fits'.format(inner_slope,log_rho5pc_msol_pc3,log_M_BH_msol,num_runs), 
                   format='fits', overwrite=True)

#%%

# =============================================================================
# # ========================= RUN TDE CODE FOR BATCH ============================
# 
# if __name__ == '__main__':
#     
#     inputs = []
#     keys = (np.arange(num_runs)+1).astype(str)
#     in_dat = {}
#     for i in range(num_runs):
#         inputs.append((name,inner_slope,outer_slopes[i],r_break,rho_break,smooth,M_BH,
#                        decay_params))
#         in_dat[keys[i]] = inputs[i]
#         
#     start = time.time()
#     print()
#     print('Beginning TDE rate computation...')
#     print('Batch Info: ')
#     print('\t # of Runs: {:.2f}'.format(num_runs))
#     
#     if mp.cpu_count() < num_runs:
#         num_processes = mp.cpu_count()
#     else:
#         num_processes = num_runs
#     
#     print('\t # of Processes: {}'.format(num_processes))
#     print()
#     
#     
# # =============================================================================
# #     # apply_async example
# #     pbar = tqdm(total=len(in_dat))
# #     def update(*a):
# #         pbar.update()
# # 
# #     results = []
# #     with mp.Pool(num_processes) as p:
# #         for key, value in in_dat.items():
# #             results.append(p.apply_async(TDE.get_TDE_rate, (value), callback=update))
# #             
# # =============================================================================
#     p = mp.Pool(num_processes)
#     results = {}
# 
#     for key, value in in_dat.items():
#         results[key] = p.apply_async(TDE.get_TDE_rate, (value,))
# 
#     children = mp.active_children()
#     all_done = False
#     while not all_done:
#         first_print = True
#         names = (np.arange(len(children))+1).astype(str)
#         pids = []
#         cpu = []
#         mem = []
#         for i in range(len(children)):
#             my_process = psutil.Process(children[i].pid)
#             pids.append(my_process.pid)
#             cpu.append(my_process.cpu_percent(interval=1))
#             mem.append(my_process.memory_percent())
#         
#             if first_print:
#                 print('{: >15} {: >15} {: >15} {: >15}'.format('Name','PID','CPU%','MEM%'))
#             print('{: >15} {: >15} {: >15} {: >15}'.format(names[i],pids[i],cpu[i],mem[i]))
#             first_print = False
# # =============================================================================
# #         # This code block can be (re)run whenever you want to check on the progress of the pool.
# #         running, successful, error = [], [], []
# #         current_time = time.time()
# #         for key, result in results.items():
# #             try:
# #                 if result.successful():
# #                     successful.append(key)
# #                 else:
# #                     error.append(key)
# #             except ValueError:
# #                 running.append(key)
# #         rate = (len(successful) + len(error)) / (current_time - start)
# #         if len(successful) == num_runs:
# #             all_done = True
# #             break
# #         #print('Running:', sorted(running))
# #         #print('Successful:', sorted(successful))
# #         #print('Error:', sorted(error))
# #         print('Time Elapsed: {} minutes'.format((current_time - start)/60))
# #         print('{}/{} Completed'.format(len(successful),num_runs))
# #         print()
# #         #print('Rate:', round(rate, 3))
# # =============================================================================
#         time.sleep(5)
#         #print('Estimated time to completion:', time.strftime('%H:%M:%S', time.gmtime(len(running) / rate)))
#                 
#     p.close()
#     p.join()
#     
#         
#     end = time.time()
# 
#     print()    
#     print('Done.')
#     print()
#     print('Runtime: {:.2f} minutes / {:.2f} hours'.format(round(end-start,3)/60,
#                                                           round(end-start,3)/3600))
#     print()
# =============================================================================

# commented below because the children like to play in here
#%%
output_table = Table()
#for key, r in results.items():
#    output_table = vstack([output_table,r])
for r in results:
    output_table = vstack([output_table,r])

#%%

print('CPU USAGE:')
per_cpu = psutil.cpu_percent(percpu=True)
# For individual core usage with blocking, psutil.cpu_percent(interval=1, percpu=True)
for idx, usage in enumerate(per_cpu):
    print(f"CORE_{idx+1}: {usage}%")
    
mem_usage = psutil.virtual_memory()
print()
print('MEMORY USAGE:')
print(f"Free: {mem_usage.percent}%")
print(f"Total: {mem_usage.total/(1024**3):.2f}G")
print(f"Used: {mem_usage.used/(1024**3):.2f}G")
print()

my_process = psutil.Process(getpid())
print("Name:", my_process.name())
print("PID:", my_process.pid)
print("Executable:", my_process.exe())
print("CPU%:", my_process.cpu_percent(interval=1))
print("MEM%:", my_process.memory_percent())


#%%

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
    radys = output_table['psi_rads'][i].data
    psi_bh = output_table['psi_bh'][i].data
    psi_enc = output_table['psi_enc'][i].data
    psi_ext = output_table['psi_ext'][i].data
    psi_tot = output_table['psi_tot'][i].data
    
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
    radys = output_table['psi_rads'][i].data
    psi_bh = output_table['psi_bh'][i].data
    psi_enc = output_table['psi_enc'][i].data
    psi_ext = output_table['psi_ext'][i].data
    psi_tot = output_table['psi_tot'][i].data
    
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

plt.savefig(TDE_plot_dir+'{:.1f}s_{:.1f}d_{:.1f}mbh_{}_outer_slopes_TDE_rates_solar.png'.format(inner_slope,log_rho5pc_msol_pc3,log_M_BH_msol,num_runs),
             bbox_inches='tight', pad_inches=0.1, dpi=500)


#%%

# plot the distribution functions
plt.figure(dpi=500)
#plt.title('Inner $\gamma$:-{:.1f}, Outer $\gamma$:-{:.1f}'.format(inner_slope,outer_slopes[i]))
plt.ylabel('log(f($\epsilon$))')
plt.xlabel('log($\epsilon$)')
for i in range(len(output_table)):
    epsilon = output_table['orb_ens'][i].data
    DF = output_table['DF'][i].data
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
    epsilon = output_table['orb_ens'][i].data
    DF = output_table['DF'][i].data
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
    epsilon = output_table['orb_ens'][i].data
    qs = output_table['q'][i].data
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
    epsilon = output_table['orb_ens'][i].data
    LC_flux_solar = output_table['LC_flux_solar'][i].data
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
    epsilon = output_table['orb_ens'][i].data
    LC_flux_solar = output_table['LC_flux_solar'][i].data
    plt.plot(np.log10(epsilon),np.log10(LC_flux_solar),color=cmap((float(i)+1)/len(output_table)), label='Outer $\gamma$ = {:.1f}'.format(outer_slopes[i]))

plt.legend(fontsize='x-small')
plt.ylim(-45,-34)

plt.savefig(TDE_plot_dir+'{:.1f}s_{:.1f}d_{:.1f}mbh_{}_outer_slopes_LC_fluxes_zoom.png'.format(inner_slope,log_rho5pc_msol_pc3,log_M_BH_msol,num_runs),
             bbox_inches='tight', pad_inches=0.1, dpi=500)

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
