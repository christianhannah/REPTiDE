#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jun 20 21:09:20 2023

This code executes TDE rate calculations for a batch of galaxy density/BH parameters
designed to test the sensitivity of rates to the outer galaxy slopes.

@author: christian
"""

import TDE_rate_modeling_batch_nuker as TDE
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

# CONSTANTS
pc_to_m = 3.08567758128e16 
M_sol =  1.989e30 # kg
R_sol = 696.34e6 # m
G = 6.6743e-11 # m^3 s^-2 kg^-1


toy_model_match = False

#%%

# Define the range of radii
R = np.linspace(0.01, 1e5, 1000)

# Define the alpha parameters to vary
alpha_params = np.linspace(0.2, 2.5, 5)

# Set the central surface brightness and break radius
I_b = 1
R_b = 14.2
gam = 0.37
beta = 5

cmap=plt.get_cmap("turbo")
plt.figure(dpi=600)
# Generate the plots for different alpha parameters
for i, alpha in enumerate(alpha_params):
    b = 2 * alpha - 0.324
    profile = 2**((beta-gam)/alpha)*I_b*(R_b/R)**gam*(1+(R_b/R)**alpha)**((beta-gam)/alpha)
    plt.plot(np.log10(R), np.log10(profile), label=f'α = {alpha:.2f}',color=cmap((float(i)+1)/len(alpha_params)))

# Set plot properties
plt.xlabel('log(Radius)')
plt.ylabel('log(Surface Brightness)')
plt.title('Nuker Surface Brightness Profile with Varying Alpha Parameter')
plt.legend()
plt.grid(True)

# Display the plot
plt.show()

#%%


def piecewise_linear(x, x0, y0, k1, k2):
    return np.piecewise(x, [x < x0], [lambda x:k1*x + y0-k1*x0, lambda x:k2*x + y0-k2*x0])

# =============================================================================
# =============================================================================
# =============================================================================


# === DEFINE BATCH OF DENSITY PARAMETERS FROM STONE & METZGER 2016 =========
        
file_path = '../Data_Sets/Stone_Metzger_data/RenukaAnil.dat'
f = open(file_path, 'r')
all_lines = f.readlines()
f.close()

names_init = []
for i in range(len(all_lines)):
    split = all_lines[i].split()
    names_init.append(split[0])    
names_init = np.array(names_init)

nuke_names = np.unique(names_init)

#%%
# find the density profile in Stone&Metzger 2016 and compute the slopes
#nuke_name = 'NGC1023' # f_pinhole for NGC 1023 is 0.161

inner_slopes = np.zeros((len(nuke_names)))
outer_slopes = np.zeros((len(nuke_names)))
r_bs = np.zeros((len(nuke_names)))
rho_bs = np.zeros((len(nuke_names)))
M_BHs = np.zeros((len(nuke_names)))
log_M_BH_msols = np.zeros((len(nuke_names)))
r_breaks = np.zeros((len(nuke_names)))
rho_breaks = np.zeros((len(nuke_names)))

for i in tqdm(range(len(nuke_names)),position=0,leave=True):
    stone_data =  tu.get_stone_data(nuke_names[i])   
    stone_rad = stone_data[1] 
    stone_dens = stone_data[2]
    indy_5 = tu.find_nearest(stone_rad,5)
    stone_dens_at_5pc = stone_dens[indy_5]
    stone_rad_interp = np.geomspace(np.min(stone_rad),np.max(stone_rad),10**4)
    stone_dens_interp = 10**np.interp(np.log10(stone_rad_interp),np.log10(stone_rad),np.log10(stone_dens))


    # fit the broken power law function to the data
    popt, pcov = curve_fit(piecewise_linear, np.log10(stone_rad_interp), np.log10(stone_dens_interp),p0=[3,10,1,3])
    inner_slopes[i] = np.abs(popt[2])
    outer_slopes[i] = np.abs(popt[3])
    break_rad = 10**popt[0]

    # plot the results
    plt.figure(dpi=600)
    plt.plot(np.log10(stone_rad), np.log10(stone_dens), '*', label='S&M16 Data')
    plt.plot(np.log10(stone_rad_interp), piecewise_linear(np.log10(stone_rad_interp), *popt), 'r-', label='broken power-law fit')
    plt.legend()
    plt.show()

    rho_5pc = stone_dens_at_5pc # M_sol/pc^3
    r_b_pc = break_rad # pc

    # convert profile to broken power-law w/ B-W cusp inward of r_b
    r_breaks[i] = r_b_pc*pc_to_m # m
    rho_breaks[i] = rho_5pc*(r_b_pc/5)**(-inner_slopes[i])*M_sol/pc_to_m**3 #kg/m^3
    smooth = 0.1

    # specify the nature of the exponential decay
    # (radius to begin decay, width of decay) in pc
    decay_params_pc = np.array([1e6,1e6])
    decay_params = decay_params_pc*pc_to_m

    M_BHs[i] = 0

#%%

num_runs = len(nuke_names)


# ================= RUN BATCH THROUGH REPTiDE ===============================

if __name__ == '__main__':
    
    inputs = []
    for i in range(num_runs):
        inputs.append(((nuke_names[i],inner_slopes[i],outer_slopes[i],r_breaks[i],
                        rho_breaks[i],smooth,M_BHs[i],
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
    print('Runtime: {:.2f} minutes / {:.2f} hours'.format(round(end-start,
                                                                3)/60,
                                                          round(end-start,
                                                                3)/3600))
    print()


    output_table.write(TDE_results_dir+
                       'stone_comparison_TDE.fits',
                       format='fits', overwrite=True)
    
    os.system('say "Reptide calculation complete"')

    applescript = """
    display dialog "REPTiDE Calculation Complete" ¬
    with title "REPTiDE" ¬
    buttons {"OK"}
    """
    subprocess.call("osascript -e '{}'".format(applescript), shell=True)
