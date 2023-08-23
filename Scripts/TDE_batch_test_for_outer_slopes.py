#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri May 12 15:12:20 2023

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

from os import getpid
import psutil
import os
import subprocess

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

inner_slope = 1.6
logrho5pc = 3.8 # Msol/pc^3
rho_5pc = 10.0**(logrho5pc)*M_sol/pc_to_m**3 #kg/m^3
r_5pc = 5*pc_to_m # m
log_rho5pc_msol_pc3 = np.log10(rho_5pc/M_sol*pc_to_m**3)


radys = np.geomspace(10**-3,10**16,10**4)*pc_to_m
rho_inits = rho_5pc*(radys/r_5pc)**(-inner_slope)
ind_50 = tu.find_nearest(radys/pc_to_m,5)
rho_break = rho_inits[ind_50] # kg/m^3

name = 'Test Galaxy'

# ================= RUN BATCH THROUGH REPTiDE ===============================

if __name__ == '__main__':
    
    inputs = []
    for i in range(num_runs):
        inputs.append(((name,inner_slope,outer_slopes[i],r_break,
                        rho_break,smooth,M_BH,
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
                       '{:.1f}s_{:.1f}d_{:.1f}mbh_{}_'.format(inner_slope,
                                                              log_rho5pc_msol_pc3,
                                                              log_M_BH_msol,
                                                              num_runs)+
                       'outer_slopes.fits',
                       format='fits', overwrite=True)
    
    os.system('say "Reptide calculation complete"')

    applescript = """
    display dialog "REPTiDE Calculation Complete" ¬
    with title "REPTiDE" ¬
    buttons {"OK"}
    """
    subprocess.call("osascript -e '{}'".format(applescript), shell=True)
