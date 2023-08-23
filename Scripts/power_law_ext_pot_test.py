#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jun 15 23:44:54 2023

@author: christian
"""

import numpy as np
from scipy import integrate
import matplotlib.pyplot as plt
from matplotlib.collections import LineCollection

# CONSTANTS
pc_to_m = 3.08567758128e16 
M_sol =  1.989e30 # kg
R_sol = 696.34e6 # m
G = 6.6743e-11 # m^3 s^-2 kg^-1


def density(r,rho_0,r_0,gam):
    decay_params = np.array([1e5,1e3])*pc_to_m
    return np.piecewise(r, [r>=decay_params[0], r<decay_params[0]], 
                    [lambda r,rho_0,r_0,gam,decay_params : rho_0*(r/r_0)**(-gam)*np.exp(-(r-decay_params[0])/decay_params[1]), 
                     lambda r,rho_0,r_0,gam,decay_params : rho_0*(r/r_0)**(-gam)], rho_0,r_0,gam,decay_params)
    #return rho_0*(r/r_0)**(-gam)
    
def get_ext_pot_integrand(r,rho_0,r_0,gam):
    return r*density(r,rho_0,r_0,gam)   
# function to compute the contribution to the potential of the galaxy at 
# larger radii
def get_ext_potential(r,rho_0,r_0,gam,min_ind):
    return 4*np.pi*G*integrate.trapz(get_ext_pot_integrand(r[min_ind:],rho_0,r_0,gam),r[min_ind:])

r_pc = np.geomspace(0.01,1e10,2000)
r = r_pc*pc_to_m

gams = np.linspace(0.5,10,20)
rho_0_pc = 10**3.8
rho_0 = rho_0_pc * (M_sol/pc_to_m**3) # kg/m^3
r_0_pc = 5
r_0 = r_0_pc*pc_to_m # m


ex_pots = np.zeros((len(r),len(gams)))
for i,g in enumerate(gams):
    for j in range(len(r)):
        ex_pots[j,i] = get_ext_potential(r,rho_0,r_0,g,j)


ls = []
for i in range(len(gams)-1):
    a = np.zeros((len(r),2))
    a[:,0] = np.log10(r_pc)
    a[:,1] = np.log10(ex_pots[:,i])
    #a[:,1] = (ex_pots[:,-1]-ex_pots[:,i+1])/ex_pots[:,-1]
    ls.append(a)


fig, ax = plt.subplots(dpi=500)
lines = LineCollection(ls, array=gams, cmap='turbo', alpha=0.6)
ax.add_collection(lines)
fig.colorbar(lines, label='$\gamma$')
ax.set_xlabel('log(Radius [pc])')
ax.set_ylabel('$\psi_{\gamma=10}$-$\psi_{\gamma}$/$\psi_{\gamma=10}$')
ax.autoscale()
plt.show()
plt.close()



#%%
cmap=plt.get_cmap("turbo")

max_inds = np.linspace(1800,1990,20).astype(int)
plt.figure(dpi=500)
plt.title('$\gamma$ = 0.5')

ex_pot_full = np.zeros((len(r)))
for j in range(len(r)):
    ex_pot_full[j] = get_ext_potential(r,rho_0,r_0,0.5,j)

for i in range(len(max_inds)):
    ex_pot = np.zeros((len(r[0:max_inds[i]+1])))
    for j in range(len(r[0:max_inds[i]+1])):
        ex_pot[j] = get_ext_potential(r[0:max_inds[i]+1],rho_0,r_0,0.5,j)
    #plt.plot(np.log10(r_pc[0:max_inds[i]+1]),np.log10(ex_pot),
    #         color = cmap((float(i)+1)/len(max_inds)),label='{:.2f}'.format(np.log10(r_pc[max_inds[i]])))
    plt.plot(np.log10(r_pc[0:max_inds[i]+1]),(ex_pot_full[0:max_inds[i]+1]-ex_pot)/ex_pot_full[0:max_inds[i]+1],
             color = cmap((float(i)+1)/len(max_inds)),label='{:.2f}'.format(np.log10(r_pc[max_inds[i]])))
plt.legend(loc='lower left',ncols=5)
plt.xlabel('log(Radius [pc])')
#plt.ylabel('log($\psi_{ext}(r)$ [m$^2$/s$^2$])')
plt.ylabel('$\psi_{ext,full}$-$\psi_{ext,i}$/$\psi_{ext,full}$')




