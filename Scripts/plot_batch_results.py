#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Mar  2 10:10:26 2023

@author: christian
"""

from astropy.table import Table
import matplotlib.pyplot as plt
import numpy as np

pc_to_m = 3.08567758128e16 

dat =Table.read('../Result_Tables/TDE_output_our_gals_w_slope_0.5_to_2.25_reff_1000pc.fits')

cmap=plt.get_cmap("turbo")

plt.figure(dpi=500)
for i in range(len(dat['names'])):
    plt.plot(np.log10(dat['orb_ens'][i]),np.log10(dat['DFs'][i]),
             color=cmap((float(i)+1)/len(dat['names'])))
#plt.ylim(-25,-10)
#plt.ylim(-60,-57.5)
plt.ylim(-100,-50)
#plt.xlim(7.5,14.5)
plt.legend()
#plt.xlim(0.95,1.05)
plt.ylabel('log(f($\epsilon$))')
plt.xlabel('log($\epsilon$)')
plt.title('logM$_{BH}$=6, $\gamma$=-1.9, log($\\rho_{10pc}$)=3.5 M$_\odot$/pc$^3$')
plt.show()

plt.figure(dpi=500)
for i in range(len(dat['names'])):
    plt.plot(np.log10(dat['orb_ens'][i]),np.log10(dat['qs'][i]),
             color=cmap((float(i)+1)/len(dat['names'])))
#plt.ylim(-25,-10)
plt.ylim(-20,6)
plt.legend()
plt.xlim(7.5,14.5)
plt.ylabel('log(q)')
plt.xlabel('log($\epsilon$)')
plt.title('logM$_{BH}$=6, $\gamma$=-1.9, log($\\rho_{10pc}$)=3.5 M$_\odot$/pc$^3$')
plt.show()


plt.figure(dpi=500)
for i in range(len(dat['names'])):
    plt.plot(np.log10(dat['orb_ens'][i]),np.log10(dat['LC_fluxes'][i]),
             color=cmap((float(i)+1)/len(dat['names'])))
#plt.ylim(-25,-10)
#plt.ylim(-26,-23)
plt.ylim(-50,-22)
#plt.xlim(7.5,14.5)
plt.legend()
#plt.xlim(0.95,1.05)
plt.ylabel('$log(\mathcal{F}(\epsilon)$)')
plt.xlabel('log($\epsilon$)')
plt.title('logM$_{BH}$=6, $\gamma$=-1.9, log($\\rho_{10pc}$)=3.5 M$_\odot$/pc$^3$')
plt.show()

#%%




plt.figure(dpi=500)
plt.scatter(-dat['slopes'],np.log10(dat['TDE_rate_single']),c=np.log10(dat['rho_5pc']),cmap='viridis')

# =============================================================================
# for i in range(len(dat['names'])):
#     plt.plot(dat['slopes'][i],dat['TDE_rate_single'][i],
#              color=cmap((float(i)+1)/len(dat['names'])),linestyle='',marker='o')
# =============================================================================

#plt.ylim(-25,-10)
#plt.ylim(5e-5,2e-4)
#plt.xlim(7.5,14.5)
#plt.legend()
plt.xlim(-2.25,-0.45)
plt.colorbar(label='log($\\rho_{5pc}$ [M$_\odot$/pc$^3$])')
#plt.yscale('log')
plt.ylabel('$\dot N_{TDE}~[yr^{-1}])$')
plt.xlabel('$\gamma$')
#plt.title('logM$_{BH}$=6, $\gamma$=-1.9, log($\\rho_{10pc}$)=3.5 M$_\odot$/pc$^3$')
plt.show()

#%%
plt.figure(dpi=500)
#plt.plot(np.log10(r*3.24078e-17),np.log10(psi_1+psi_2+psi_3),'k')
# =============================================================================
# for i in range(len(dat['names'])):
#     plt.plot(np.log10(r/pc_to_m),np.log10(pots[i,0,:]),label='{} pc'.format(wids_pc[i]),
#              color=cmap((float(i)+1)/len(dat['names'])),linewidth=3,alpha=0.5)
# =============================================================================
for i in range(len(dat['names'])):
    plt.plot(np.log10(dat['psi_rads'][i]/pc_to_m),np.log10(dat['psi_encs'][i]),linestyle='--',
             color=cmap((float(i)+1)/len(dat['names'])))
for i in range(len(dat['names'])):
    plt.plot(np.log10(dat['psi_rads'][i]/pc_to_m),np.log10(dat['psi_exts'][i]),linestyle='-',
             color=cmap((float(i)+1)/len(dat['names'])))
for i in range(len(dat['names'])):
    plt.plot(np.log10(dat['psi_rads'][i]/pc_to_m),np.log10(dat['psi_bhs'][i]),linestyle=':',
             color=cmap((float(i)+1)/len(dat['names'])))
plt.plot(0,0,linestyle='--',label='$\psi_{enc}$',color='k')
plt.plot(0,0,linestyle='-',label='$\psi_{ext}$',color='k')
plt.plot(np.log10(dat['psi_rads'][0]/pc_to_m),np.log10(dat['psi_bhs'][0]),label='$\psi_{BH}$',color='k',linestyle=':')
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
for i in range(len(dat['names'])):
    plt.plot(np.log10(dat['psi_rads'][i]/pc_to_m),np.log10(dat['psi_tots'][i]),
             color=cmap((float(i)+1)/len(dat['names'])))
plt.legend()
#plt.ylim(np.min(np.log10(psi_bhs[0,:])),np.max(np.log10(psi_bhs[0,:])))
plt.ylim(-5,15)
plt.xlim(-6,15)
plt.xlabel('log(Radius [pc])')
plt.ylabel('log($\psi(r)$ [m$^2$/s$^2$])')
plt.title('logM$_{BH}$=6, $\gamma$=-1.9, log($\\rho_{10pc}$)=3.5 M$_\odot$/pc$^3$')
plt.show()
