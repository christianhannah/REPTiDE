#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jun 20 21:09:20 2023

This code executes TDE rate calculations for a batch of galaxy density/BH parameters
designed to test the sensitivity of rates to the outer galaxy slopes.

@author: christian
"""

import TDE_rate_modeling_batch_discrete as TDE
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

stone_rads = []
stone_denses = []
M_BHs = np.zeros((len(nuke_names)))
log_M_BH_msols = np.zeros((len(nuke_names)))

for i in tqdm(range(len(nuke_names)),position=0,leave=True):
    stone_data =  tu.get_stone_data(nuke_names[i])   
    stone_rad = stone_data[1] 
    stone_rads.append(stone_rad)
    stone_dens = stone_data[2]
    stone_denses.append(stone_dens)

    r = np.geomspace(0.01,1e8,2000)
    new_dens = tu.get_rho_r_discrete(r,stone_rad,stone_dens,False)
    plt.figure(dpi=600)
    plt.plot(np.log10(stone_rad),np.log10(stone_dens),'*',label='S&M16 Data')
    plt.plot(np.log10(r),np.log10(new_dens),label='Interpolation w/ decay')
    plt.legend()
    plt.xlabel('log(Radius [pc])')
    plt.ylabel('log(Density [M$_\odot$/pc$^3$])')
    plt.ylim(-50,10)
    plt.savefig('../Plots/Stone_Densities/{}.png'.format(nuke_names[i]),
                 bbox_inches='tight', pad_inches=0.1, dpi=500)
    plt.close()

    M_BHs[i] = tu.get_mbh(nuke_names[i])

#%%
bh_inds = np.where(M_BHs != 0.0)[0]
nuke_names = nuke_names[bh_inds]
stone_rads = np.array(stone_rads)[bh_inds] * pc_to_m
stone_denses = np.array(stone_denses)[bh_inds] * (M_sol/pc_to_m**3)
M_BHs = M_BHs[bh_inds] * M_sol

#%%

num_runs = 22#len(nuke_names)

indyboi = 22
# ================= RUN BATCH THROUGH REPTiDE ===============================

if __name__ == '__main__':
    
    inputs = []
    for i in range(num_runs):
        inputs.append(((nuke_names[indyboi+i],stone_rads[indyboi+i],stone_denses[indyboi+i],M_BHs[indyboi+i])))

    start = time.time()
    print()
    print('Beginning TDE rate computation...')
    print('Batch Info: ')
    print('\t # of Processes: {}'.format(min(num_runs, mp.cpu_count())))
    print('\t # of Runs: {}'.format(num_runs))
    print()
    with mp.Pool(min(num_runs, mp.cpu_count())) as p:
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
                       'stone_comparison_TDE_second_22.fits',
                       format='fits', overwrite=True)
    
    os.system('say "Reptide calculation complete"')

    applescript = """
    display dialog "REPTiDE Calculation Complete" ¬
    with title "REPTiDE" ¬
    buttons {"OK"}
    """
    subprocess.call("osascript -e '{}'".format(applescript), shell=True)
