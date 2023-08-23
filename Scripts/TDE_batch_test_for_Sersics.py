#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jun 22 12:03:42 2023

This code executes TDE rate calculations for a batch of galaxy density/BH parameters
designed to test the sensitivity of rates to sersic parameters.

@author: christian
"""

import TDE_rate_modeling_batch_sersic as TDE
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

# =============================================================================
# =============================================================================
# =============================================================================


# === DEFINE BATCH OF DENSITY PARAMETERS FROM STONE & METZGER 2016 =========
        
# TODO
# fill in the parameters for the batch run of multiple sersic indices.


#%%

num_runs = 1#len(nuke_names)


# ================= RUN BATCH THROUGH REPTiDE ===============================

if __name__ == '__main__':
    
    inputs = []
    for i in range(num_runs):
        inputs.append(((nuke_names[i],stone_rads[i],stone_denses[i],M_BHs[i])))

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
                       'stone_comparison_TDE.fits',
                       format='fits', overwrite=True)
    
    os.system('say "Reptide calculation complete"')

    applescript = """
    display dialog "REPTiDE Calculation Complete" ¬
    with title "REPTiDE" ¬
    buttons {"OK"}
    """
    subprocess.call("osascript -e '{}'".format(applescript), shell=True)
