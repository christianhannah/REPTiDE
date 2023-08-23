#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Mar  2 13:48:31 2023

Code to compute the TDE rate of a single density profile.

@author: christian
"""

import matplotlib.pyplot as plt
import numpy as np
import pdb
from tqdm import tqdm
from scipy import integrate
import time
#import warnings
#warnings.filterwarnings("ignore")
import TDE_util as tu
from astropy.table import Table
import scipy.special as s
from scipy.optimize import curve_fit


# CONSTANTS
pc_to_m = 3.08567758128e16 
M_sol =  1.989e30 # kg
R_sol = 696.34e6 # m
G = 6.6743e-11 # m^3 s^-2 kg^-1


#%%

#========================== PROGRAM PARAMS ====================================

# specify the range of specific orbital energies to consider
e_min = 10**8
e_max = 3.5094627e+14
#e_max = G*M_BH/(17.78*pc_to_m)

# specify the minimum and maximum stellar mass to consider in the PDMF
M_min = 0.08
M_max = 1

# switch to match toy model for comparison
toy_model_match = False

# switch to use Nuker-law w/ exp decay 
nuker = True

#======================== END PROGRAM PARAMS ==================================

#%%
# ==================== SPECIFY DENSITY PARAMETERS =============================

# below was modified to do a broken power law at 10 pc with exponential density decay

# values below for NGC 1427
slope = 1.9 # positive value
rho_5pc = 10.0**3.5 # M_sol/pc^3
r_b_pc = 0.01 # pc

# convert profile to broken power-law w/ B-W cusp inward of r_b
r_b = r_b_pc*pc_to_m # m
rho_b = rho_5pc*(r_b_pc/5)**(-slope)*M_sol/pc_to_m**3 #kg/m^3
smooth = 0.1

# specify the nature of the exponential decay
# (radius to begin decay, width of decay) in pc
decay_params_pc = np.array([1e9,1e5])
decay_params = decay_params_pc*pc_to_m


if toy_model_match:
    # match toy model for comparison
    M_BH = 10**6*M_sol
    slope = 1.9
    rho_infl = 3e3*M_sol/pc_to_m**3 #kg/m^3
    r_infl = 10*pc_to_m # m
    r_b_pc = 10**-6
    r_b = r_b_pc*pc_to_m
    rho_b = rho_infl*(r_b/r_infl)**(-slope)#kg/m^3
    e_min = 10**6
    e_max = 3.5094627e+14
    t_bound = 3
    r_t = R_sol*(M_BH/M_sol)**(1/3)
    anal_df_term_1 = slope*(slope-0.5)/(np.sqrt(8)*np.pi**1.5)
    anal_df_term_2 = (rho_infl/M_sol)*((G*M_BH)/r_infl)**(-slope)
    anal_df_term_3 = s.gamma(slope)/s.gamma(slope+0.5)
    J_LC = np.sqrt(2*G*M_BH*r_t)
    q_prefac_1 = (16*np.pi**(1/2)*G**2*M_sol*rho_infl*np.log(0.4*M_BH/M_sol))/(3*J_LC**2)
    q_prefac_2 = (G*M_BH/r_infl)**(-slope)
    q_prefac_3 = (s.gamma(slope)*slope*(slope-1/2))/s.gamma(slope+1/2)


if nuker:
    # find the density profile in Stone&Metzger 2016 and compute the slopes
    nuke_name = 'NGC1023' # f_pinhole for NGC 1023 is 0.161
    stone_data =  tu.get_stone_data(nuke_name)   
    stone_rad = stone_data[1] 
    stone_dens = stone_data[2]
    indy_5 = tu.find_nearest(stone_rad,5)
    stone_dens_at_5pc = stone_dens[indy_5]
    stone_rad_interp = np.geomspace(np.min(stone_rad),np.max(stone_rad),10**4)
    stone_dens_interp = 10**np.interp(np.log10(stone_rad_interp),np.log10(stone_rad),np.log10(stone_dens))

    # specify BH mass
    M_BH = 10**8.37*M_sol

    def piecewise_linear(x, x0, y0, k1, k2):
        return np.piecewise(x, [x < x0], [lambda x:k1*x + y0-k1*x0, lambda x:k2*x + y0-k2*x0])

    # fit the broken power law function to the data
    popt, pcov = curve_fit(piecewise_linear, np.log10(stone_rad_interp), np.log10(stone_dens_interp))
    inner_slope = np.abs(popt[2])
    outer_slope = np.abs(popt[3])
    break_rad = 10**popt[0]
    
    # plot the results
    plt.plot(np.log10(stone_rad), np.log10(stone_dens), '*', label='S&M16 Data')
    plt.plot(np.log10(stone_rad_interp), piecewise_linear(np.log10(stone_rad_interp), *popt), 'r-', label='broken power-law fit')
    plt.legend()
    plt.show()
    
    
    #slope = inner_slope # positive value
    rho_5pc = stone_dens_at_5pc # M_sol/pc^3
    r_b_pc = break_rad # pc

    # convert profile to broken power-law w/ B-W cusp inward of r_b
    r_b = r_b_pc*pc_to_m # m
    rho_b = rho_5pc*(r_b_pc/5)**(-inner_slope)*M_sol/pc_to_m**3 #kg/m^3
    smooth = 0.1

    # specify the nature of the exponential decay
    # (radius to begin decay, width of decay) in pc
    decay_params_pc = np.array([1e9,1e9])
    decay_params = decay_params_pc*pc_to_m
    
    # specify BH mass
    M_BH = 10**6.28*M_sol # NGC 4474


# =============================================================================


radys = np.geomspace(10**-3,10**16,10**4)*pc_to_m
if not nuker:
    dens = tu.get_rho_r_new(radys,slope,rho_b,r_b,smooth,decay_params,toy_model_match)
    psis = tu.get_psi_r(radys,M_BH,slope,rho_b,r_b,smooth,decay_params,toy_model_match)
else:
    dens = tu.get_rho_r_nuker(radys,inner_slope,outer_slope,rho_b,r_b,smooth,decay_params,toy_model_match)
    psis = tu.get_psi_r_nuker(radys,M_BH,inner_slope,outer_slope,rho_b,r_b,smooth,decay_params,toy_model_match)

# let's try defining e_min and e_max from the range 
#%%
# plot the potentials and their components

#outer_slope = 4.0
if nuker:
    cmap=plt.get_cmap("turbo")
    plt.figure(dpi=500)
    #plt.plot(np.log10(radys/pc_to_m),np.log10(dens))
    plt.title('Nuker Test {}, Inner $\gamma$:-{:.1f}, Outer $\gamma$:-{:.1f}'.format(nuke_name,slope,outer_slope))
    indy_10 = tu.find_nearest(radys/pc_to_m, r_b_pc)
    indy_exp = tu.find_nearest(radys/pc_to_m, decay_params_pc[0])
    plt.plot(np.log10(radys[0:indy_10]/pc_to_m),np.log10(dens[0:indy_10]),color=cmap(0.1))
    plt.plot(np.log10(radys[indy_10:indy_exp]/pc_to_m),np.log10(dens[indy_10:indy_exp]),color=cmap(0.7))
    plt.plot(np.log10(radys[indy_exp:]/pc_to_m),np.log10(dens[indy_exp:]),color=cmap(1.5))
    plt.xlabel('log(Radius [pc])')
    plt.ylabel('log($\\rho(r)~[kg/m^3]$)')
    #plt.xlim(-3,9)
    #plt.plot([1.25,1.25],[-45,-9],color='r',label='r$_{infl}$ (S&M16)')
    plt.ylim(-40,-7)
    plt.plot(np.log10(stone_rad),np.log10(stone_dens*M_sol/pc_to_m**3),color='k',linestyle='--',label='S&M16')
    plt.show()


if toy_model_match:
    plt.figure(dpi=500)

    plt.title('Toy Model Test')
    plt.plot(np.log10(radys/pc_to_m),np.log10(psis[0]),color=cmap(0.1),linestyle=':')
    plt.xlabel('log(Radius [pc])')
    plt.ylabel('log($\psi(r)~[m^2/s^2]$)')
    plt.legend()
    #plt.xlim(-3,9)
    #plt.ylim(np.min(np.log10(psis[1])),np.max(np.log10(psis[1]))+0.1*np.max(np.log10(psis[1])))
    plt.show()
    
else:
    plt.figure(dpi=500)
    #plt.plot(np.log10(radys/pc_to_m),np.log10(dens))
    plt.title('Nuker Test {}, Inner $\gamma$:-{:.1f}, Outer $\gamma$:-{:.1f}'.format(nuke_name,slope,outer_slope))
    
    plt.plot(np.log10(radys/pc_to_m),np.log10(psis[0]),color='k')
    
    plt.plot(np.log10(radys[0:indy_10]/pc_to_m),np.log10(psis[1][0:indy_10]),color=cmap(0.1),linestyle=':')
    plt.plot(np.log10(radys[indy_10:indy_exp]/pc_to_m),np.log10(psis[1][indy_10:indy_exp]),color=cmap(0.7),linestyle=':')
    plt.plot(np.log10(radys[indy_exp:]/pc_to_m),np.log10(psis[1][indy_exp:]),color=cmap(1.5),linestyle=':')
    
    plt.plot(np.log10(radys[0:indy_10]/pc_to_m),np.log10(psis[2][0:indy_10]),color=cmap(0.1),linestyle='--')
    plt.plot(np.log10(radys[indy_10:indy_exp]/pc_to_m),np.log10(psis[2][indy_10:indy_exp]),color=cmap(0.7),linestyle='--')
    plt.plot(np.log10(radys[indy_exp:]/pc_to_m),np.log10(psis[2][indy_exp:]),color=cmap(1.5),linestyle='--')
    
    plt.plot(np.log10(radys[0:indy_10]/pc_to_m),np.log10(psis[3][0:indy_10]),color=cmap(0.1))
    plt.plot(np.log10(radys[indy_10:indy_exp]/pc_to_m),np.log10(psis[3][indy_10:indy_exp]),color=cmap(0.7))
    plt.plot(np.log10(radys[indy_exp:]/pc_to_m),np.log10(psis[3][indy_exp:]),color=cmap(1.5))
    
    plt.plot(0,0,label='$\psi_{BH}$',linestyle=':',color='k')
    plt.plot(0,0,label='$\psi_{enc}$',linestyle='--',color='k')
    plt.plot(0,0,label='$\psi_{ext}$',color='k')
   
    
    plt.plot([1.25,1.25],[0,14],color='r',label='r$_{infl}$ (S&M16)')
    
    plt.xlabel('log(Radius [pc])')
    plt.ylabel('log($\psi(r)~[m^2/s^2]$)')
    plt.legend()
    #plt.xlim(-3,9)
    #plt.ylim(np.min(np.log10(psis[1])),np.max(np.log10(psis[1]))+0.1*np.max(np.log10(psis[1])))
    plt.ylim(5,np.max(np.log10(psis[1]))+0.1*np.max(np.log10(psis[1])))
    plt.show()



#%%
#pdb.set_trace()

print()
print("=======================================")
print('STARTING SINGLE TDE RATE CALCULATION...')
print("=======================================")
print()


#%%
# =========================== DF Computation ==================================
# =============================================================================
# Compute the stellar distribution function (DF) as a function of specific 
# orbital energy (f(epsilon))
# =============================================================================
# =============================================================================

# specify the range of specific orbital energies to consider
e = np.geomspace(e_min,e_max,10**2)

# STEP 1: Define wide range of t
t_bound = 1.5
num_t = 10**3
t = np.linspace(-t_bound, t_bound, num_t)

integrals = np.zeros_like(e)

# let's make sure rho_r covers all possible values of psi_t
print('Ensuring psi_r covers psi_t...')
psi_t_min = np.min([np.min(tu.get_psi_t(e[0],t)),np.min(tu.get_psi_t(e[-1],t))])
psi_t_max = np.max([np.max(tu.get_psi_t(e[0],t)),np.max(tu.get_psi_t(e[-1],t))])
#pdb.set_trace()
def ensure_psi_r_covers_psi_t(psi_t_min,psi_t_max):
    r_min = 10**-6*pc_to_m
    r_max = 10**15*pc_to_m
    r_step_frac = 2
    coverage_met = False
    while not coverage_met:
        r_temp = np.geomspace(r_min,r_max,10**4)
        psi_r_temp,psi_bh_temp,psi_enc_temp,psi_ext_temp = \
            tu.get_psi_r(r_temp,M_BH,slope,rho_b,r_b,smooth,decay_params,toy_model_match)

        if psi_r_temp[-1] <= psi_t_min and psi_r_temp[0] >= psi_t_max:
            coverage_met = True
        else:
            if psi_r_temp[-1] > psi_t_min:
                r_max += r_step_frac*r_max
            if psi_r_temp[0] < psi_t_max:
                r_min -= r_step_frac*r_min
    return r_min, r_max

def ensure_psi_r_covers_psi_t_nuker(psi_t_min,psi_t_max):
    r_min = 10**-6*pc_to_m
    r_max = 10**15*pc_to_m
    r_step_frac = 2
    coverage_met = False
    while not coverage_met:
        r_temp = np.geomspace(r_min,r_max,10**4)
        psi_r_temp,psi_bh_temp,psi_enc_temp,psi_ext_temp = \
            tu.get_psi_r_nuker(r_temp,M_BH,inner_slope,outer_slope,rho_b,r_b,smooth,decay_params,toy_model_match)

        if psi_r_temp[-1] <= psi_t_min and psi_r_temp[0] >= psi_t_max:
            coverage_met = True
        else:
            if psi_r_temp[-1] > psi_t_min:
                r_max += r_step_frac*r_max
            if psi_r_temp[0] < psi_t_max:
                r_min -= r_step_frac*r_min
    return r_min, r_max

if nuker:
    r_min, r_max = ensure_psi_r_covers_psi_t_nuker(psi_t_min,psi_t_max)
else:
    r_min, r_max = ensure_psi_r_covers_psi_t(psi_t_min,psi_t_max)
num_rad = 10**4
r_ls = np.geomspace(r_min,r_max,num_rad)

if nuker:
    psi_r_init,psi_bh_init,psi_enc_init,psi_ext_init = tu.get_psi_r_nuker(r_ls,M_BH,inner_slope,outer_slope,rho_b,r_b,smooth,decay_params,toy_model_match)
else:
    psi_r_init,psi_bh_init,psi_enc_init,psi_ext_init = tu.get_psi_r(r_ls,M_BH,slope,rho_b,r_b,smooth,decay_params,toy_model_match)
print('Finished.')
print()

# =============================================================================
#     plt.figure(dpi=500)
#     plt.plot(np.log10(r_ls/pc_to_m),np.log10(psi_bh_init),label='$\psi_{BH}$')
#     plt.plot(np.log10(r_ls/pc_to_m),np.log10(psi_enc_init),label='$\psi_{enc}$')
#     plt.plot(np.log10(r_ls/pc_to_m),np.log10(psi_ext_init),label='$\psi_{ext}$')
#     plt.plot(np.log10(r_ls/pc_to_m),np.log10(psi_r_init),'k--', label='Total')
#     plt.legend()
#     plt.ylim(np.min(np.log10(psi_bh_init)),np.max(np.log10(psi_bh_init)))
#     plt.xlabel('log(Radius [pc])')
#     plt.ylabel('log($\psi(r)$ [m$^2$/s$^2$])')
#     plt.show()
# =============================================================================
#%%
print('Computing DF...')
print()
time.sleep(1)
def compute_DF(r_ls,psi_r_init,toy_model_match):
    for j in tqdm(range(len(e)), position=0, leave=True):
    #for j in range(len(e)):
        # STEP 2: Compute array of psi(t)
        psi_t = tu.get_psi_t(e[j],t)
    
        # since I am not using the true psi_r to step the radii, double check 
        # that the final radii do indeed give psi_r covering psi_t
        if np.min(psi_r_init) >= np.min(psi_t) or np.max(psi_r_init) <= np.max(psi_t):
            print('*** ERROR: psi(r) failed to cover psi(t) ***')
            pdb.set_trace()

        # STEP 4: Evaluate drho/dpsi at all values of psi_t
        d_rho_d_psi_t = np.zeros_like(psi_t)
        num_r = 10**4
        drhodpsi_ratios = np.zeros(len(psi_t))
        def get_drhodpsi(d_rho_d_psi_t,drhodpsi_ratios):
            for i in range(len(psi_t)):
                if psi_t[i] >= 1e-20:
                    r_ind = tu.find_nearest(psi_r_init,psi_t[i])
                    r_closest = r_ls[r_ind]
                    
                    spacing = ((r_closest+0.2*r_closest)-(r_closest-0.2*r_closest))/(num_r-1)
                    r = np.arange(0,num_r,1)*spacing+(r_closest-0.2*r_closest)
                    #psi_r = tu.get_psi_r(r,G,M_BH,slope,rho_b,r_b,smooth,decay_params,toy_model_match)
                    psi_r = 10**np.interp(np.log10(r),np.log10(r_ls),np.log10(psi_r_init))
                    if nuker:
                        rho_r = tu.get_rho_r_nuker(r,inner_slope,outer_slope,rho_b,r_b,smooth,decay_params,toy_model_match)
                    else:
                        rho_r = tu.get_rho_r_new(r,slope,rho_b,r_b,smooth,decay_params,toy_model_match)
                
                    rho_r[np.where(rho_r == 0.0)] = 1e-323
                
                    psi_t_ind = tu.find_nearest(psi_r,psi_t[i])

                    if psi_r[psi_t_ind]-psi_t[i] >= psi_t[i]*0.001:
                        print('ERROR: Finer grid needed for initial psi_r/psi_t match')
                        pdb.set_trace()

                    d_rho_d_psi_t[i] = (rho_r[psi_t_ind-1]-rho_r[psi_t_ind+1])/(psi_r[psi_t_ind-1]-psi_r[psi_t_ind+1])
                   
            return d_rho_d_psi_t
    
        d_rho_d_psi_t = get_drhodpsi(d_rho_d_psi_t,drhodpsi_ratios)


        # STEP 5: Use t and drho/dpsi to tabulate drho/dt
        d_rho_d_t = (e[j]/2)*(np.pi/2*np.cosh(t))/(np.cosh(np.pi/2*np.sinh(t)))**2*d_rho_d_psi_t


        # STEP 6: Tabulate the other factor from the double exponential transformed 
        # version of the DF integral
        frac_fac_t = 1/np.sqrt(e[j] - (e[j]/2*(np.tanh(np.pi/2*np.sinh(t)) + 1)))    


        integrands = d_rho_d_t * frac_fac_t

        # STEP 7: Evaluate the integral for all values of epsilon (e)
        integrals[j] = integrate.trapz(integrands,t)
        
    return integrals
  
integrals = compute_DF(r_ls,psi_r_init,toy_model_match)


# STEP 8: Compute the derivative of the integral values vs. epsilon for all 
# all values of epsilon
d_int_d_e = np.zeros(len(e)-2)
for i in range(len(e)-2):
    d_int_d_e[i] = (integrals[(i+1)-1]-integrals[(i+1)+1])/(e[(i+1)-1]-e[(i+1)+1])


DF = 1/(np.sqrt(8)*np.pi**2*M_sol)*d_int_d_e
#DF = 1/(np.sqrt(8)*np.pi**2)*d_int_d_e
orb_ens = e[1:-1]

# analytic toy model solution for comparison
if toy_model_match:
    DF_analytic = anal_df_term_1*anal_df_term_2*anal_df_term_3*orb_ens**(slope-1.5)


print('DF computation complete.')
print()


#%%
# plot the distribution function
plt.figure(dpi=500)
plt.plot(np.log10(orb_ens),np.log10(DF),color='c',marker='.')
if toy_model_match:
    plt.plot(np.log10(orb_ens),np.log10(DF_analytic),color='k',marker='',linestyle='--', label='Analytic Soln.')
    plt.legend()
#plt.ylim(-70,-60)
plt.ylabel('log(f($\epsilon$))')
plt.xlabel('log($\epsilon$)')
#%%

# =============================================================================
# =============================================================================
# =============================================================================
# =============================================================================



#%%
# ==================== q, mu, and P Computation ===============================
# =============================================================================
# Compute the orbit averaged angular momentum diffusion coefficient for 
# highly eccentric orbits, mu(epsilon), periods, and q (unitless change in 
# squared angular momentum per orbit)
# =============================================================================
# =============================================================================

mu_e = np.zeros(len(orb_ens))
periods_e = np.zeros(len(orb_ens))
r_t = R_sol*(M_BH/M_sol)**(1/3)
R_LC_e = (4*r_t*orb_ens)/(G*M_BH)

int_fac_rs = []

#avg_M_sq = tu.get_avg_M_sq(M_min,M_max)
avg_M_sq = M_sol**2

print('Computing q...')
print()
time.sleep(1)

#def compute_q(r_ref,psi_r_ref):
def compute_q(toy_model_match):
    for i in tqdm(range(len(orb_ens)), position=0, leave=True):
    #for i in range(len(orb_ens)):
        r_apo = G*M_BH/orb_ens[i]
        #rs_p = np.linspace(0,r_apo,10**3)
        #psi_r_p = G*M_BH/rs_p
        r_ref = np.geomspace(10**-6,r_apo,10**3)
        if nuker:
            psi_r_ref,psi_bh_ref,psi_enc_ref,psi_ext_ref = tu.get_psi_r_nuker(r_ref,M_BH,inner_slope,outer_slope,rho_b,r_b,smooth,decay_params,toy_model_match)
        else:
            psi_r_ref,psi_bh_ref,psi_enc_ref,psi_ext_ref = tu.get_psi_r(r_ref,M_BH,slope,rho_b,r_b,smooth,decay_params,toy_model_match)
        
        #rads = np.geomspace(r_t,r_apo,10**4)
        #integrds_p = tu.integrand_p(rads,r_ref,psi_r_ref,orb_ens[i],G,M_BH,slope,rho_b,r_b,smooth)
        periods_e[i] = 2*integrate.quadrature(tu.integrand_p,0,r_apo,args=(r_ref,psi_r_ref,orb_ens[i]),
                                              maxiter=200)[0]
        #periods_e[i] = 2*integrate.trapz(integrds_p, rads)

        es = np.geomspace(orb_ens[0],orb_ens[i],10**3)
        DF_interp = 10**np.interp(np.log10(es),np.log10(orb_ens),np.log10(DF))
        I_0 = integrate.trapz(DF_interp,es)

        #pdb.set_trace()
        #radys = np.geomspace(r_t,r_apo,10**4)
        #integrds = tu.integrand_mu(radys,r_ref,psi_r_ref,orb_ens[i],orb_ens,DF,I_0,G,M_BH,slope,rho_b,r_b,smooth,avg_M_sq)
        int_fac_r = integrate.quadrature(tu.integrand_mu,r_t,r_apo,args=(r_ref,psi_r_ref,orb_ens[i],orb_ens,DF,I_0,M_BH,avg_M_sq),
                                         tol=10**-35,maxiter=300)[0]
        #int_fac_r = integrate.trapz(integrds,radys)

        int_fac_rs.append(int_fac_r)

        mu_e[i] = 2*int_fac_r/periods_e[i]

    q_discrete = mu_e*periods_e/R_LC_e

    print('q computation complete.')
    print()
    return q_discrete, mu_e, periods_e

q, mu_e, periods_e = compute_q(toy_model_match)

# analytic solution for q under the toy model assumptions for comparison
if toy_model_match:
    Q_0_analytic = 5*np.pi/(8*(2*slope-1))*G**3*M_BH**3*orb_ens**(slope-4)
    Q_12_analytic = (np.pi)**(1/2)*((1811-798*slope+16*slope**2)/120*s.gamma(4-slope)/s.gamma(15/2-slope)+
                    (-1+2*slope)/(4*(slope-5)*(slope-4))*s.gamma(1/2+slope)/s.gamma(1+slope))*G**3*M_BH**3*orb_ens**(slope-4)
    Q_32_analytic = (np.pi/(40*s.gamma(slope-3)))*((np.pi**(1/2)*(-325+118*slope+8*slope**2)*(1/np.sin(np.pi*slope)))/s.gamma(15/2-slope)-
                    15*(2**(5-2*slope)*(1-2*slope)**2*(2*slope-7)*(2*slope-5)*(2*slope-3)*s.gamma(2*slope-8))/s.gamma(2+slope))*G**3*M_BH**3*orb_ens**(slope-4)
    q_analytic = q_prefac_1*q_prefac_2*q_prefac_3*(3*Q_12_analytic-Q_32_analytic+2*Q_0_analytic)

#%%
# plot q
plt.figure(dpi=500)
plt.plot(np.log10(orb_ens),np.log10(q),color='c',linewidth=3,marker='.')
if toy_model_match:
    plt.plot(np.log10(orb_ens),np.log10(q_analytic),color='k',marker='',linestyle='--', label='Analytic Soln.')
    plt.legend()
plt.xlabel('log($\epsilon$)')
plt.ylabel('log(q)')

#%%
# =============================================================================
# =============================================================================
# =============================================================================
# =============================================================================



#%%
# ==================== LC FLUX Computation ===============================
# =============================================================================
# Compute the flux of stars that scatter into the loss cone per unit time 
# and energy
# =============================================================================
# =============================================================================

def get_LC_flux(orb_ens,G,M_BH,DF,mu_e,q,R_LC_e,periods_e):
    J_c = G*M_BH/np.sqrt(2*orb_ens)
    ln_R_0 = np.zeros_like(orb_ens)

    for i in range(len(orb_ens)):
        if q[i] > 1:
            ln_R_0[i] = (q[i] - np.log(R_LC_e[i]))
        else:
            ln_R_0[i] = ((0.186*q[i]+0.824*np.sqrt(q[i])) - np.log(R_LC_e[i]))
        
    return (4*np.pi**2)*periods_e*J_c**2*mu_e*(DF/ln_R_0)


# =============================================================================
# =============================================================================
# =============================================================================
# =============================================================================



#%%
# ======================== TDE RATE Computation ===============================
# =============================================================================
# Compute the TDE rate by integrating the total flux into the loss cone for 
# stars of a given mass, and then by integrating over the stellar mass function.
# =============================================================================
# =============================================================================
sec_per_yr = 3.154e+7

# compute the TDE rates for a pure population of solar mass stars 
LC_flux_solar = get_LC_flux(orb_ens,G,M_BH,DF,mu_e,q,R_LC_e,periods_e)
TDE_rate_solar = integrate.trapz(LC_flux_solar, orb_ens)*sec_per_yr

if toy_model_match:
    k=0
    LC_flux_analytic = (32*np.pi)/(3*np.sqrt(2))*G**5*M_BH**3*rho_infl**2*np.log(0.4*M_BH)* \
                (G*M_BH/r_infl)**(-2*slope)*((slope*(slope-1/2)*s.gamma(slope))/s.gamma(slope+1/2))**2* \
                orb_ens[k:]**(2*slope-11/2)*(3*Q_12_analytic[k:]-Q_32_analytic[k:]+2*Q_0_analytic[k:])/(np.log(G*M_BH/(4*r_t*orb_ens[k:]))*(G**3*M_BH**3*orb_ens[k:]**(slope-4)))

#%%
# plot the solar LC flux
plt.figure(dpi=500)
plt.plot(np.log10(orb_ens),np.log10(LC_flux_solar), linewidth=3)
if toy_model_match:
    plt.plot(np.log10(orb_ens),np.log10(LC_flux_analytic), linestyle='--')
plt.ylim(-30,-20)
plt.ylabel('$log(\mathcal{F}(\epsilon)$)')
plt.xlabel('log($\epsilon$)')
#%%
masses = np.linspace(M_min,M_max,100)*M_sol
R_stars = (masses/M_sol)**0.8*R_sol # m

# get the total number of stars contributed to the LC for each mass
LC_contributions = np.zeros(len(masses))
LC_flux_per_mass = []
for i in range(len(masses)):
    r_t_adj = R_stars[i]*(M_BH/masses[i])**(1/3)
    R_LC_e_adj = (4*r_t_adj*orb_ens)/(G*M_BH)
    q_adj = mu_e*periods_e/R_LC_e_adj
    LC_flux_e = get_LC_flux(orb_ens,G,M_BH,DF,mu_e,q_adj,R_LC_e_adj,periods_e)
    LC_flux_per_mass.append(LC_flux_e)
    LC_contributions[i] = integrate.trapz(LC_flux_e, orb_ens)

LC_flux_per_mass = np.array(LC_flux_per_mass)

PDMF = tu.get_PDMF(masses)
TDE_rate_full = integrate.trapz(LC_contributions*PDMF, masses/M_sol)*sec_per_yr


#pdb.set_trace()
if toy_model_match:
    all_output = {'slope': [slope],
                  'r_BW': [r_b],
                  'rho_BW': [rho_b],
                  'smooth': [smooth],
                  'M_BH': [M_BH],
                  'decay_params': [decay_params],
                  'TDE_rate_solar': [TDE_rate_solar],
                  'TDE_rate_full': [TDE_rate_full],
                  'orb_ens': [orb_ens],
                  'DF': [DF],
                  'q': [q],
                  'LC_flux_solar': [LC_flux_solar],
                  'masses': [masses],
                  'LC_contributions_per_mass': [LC_contributions],
                  'LC_flux_per_mass': [LC_flux_per_mass],
                  'psi_rads': [r_ls],
                  'psi_bh': [psi_r_init],
                  'DF_integrals': [integrals],
                  'mu_integrals': [int_fac_rs],
                  'periods': [periods_e],
                  'mu': [mu_e],
                  'R_LC': [R_LC_e],
                  'DF_analytic': [DF_analytic],
                  'q_analytic': [q_analytic],
                  'LC_flux_solar_analytic': [LC_flux_analytic]
                  }
elif nuker:
    all_output = {'inner_slope': [inner_slope],
                  'outer_slope': [outer_slope],
                  'r_b': [r_b],
                  'rho_b': [rho_b],
                  'smooth': [smooth],
                  'M_BH': [M_BH],
                  'decay_params': [decay_params],
                  'TDE_rate_solar': [TDE_rate_solar],
                  'TDE_rate_full': [TDE_rate_full],
                  'orb_ens': [orb_ens],
                  'DF': [DF],
                  'q': [q],
                  'LC_flux_solar': [LC_flux_solar],
                  'masses': [masses],
                  'LC_contributions_per_mass': [LC_contributions],
                  'LC_flux_per_mass': [LC_flux_per_mass],
                  'psi_rads': [r_ls],
                  'psi_tot': [psi_r_init],
                  'psi_bh': [psi_bh_init],
                  'psi_enc': [psi_enc_init],
                  'psi_ext': [psi_ext_init],
                  'DF_integrals': [integrals],
                  'mu_integrals': [int_fac_rs],
                  'periods': [periods_e],
                  'mu': [mu_e],
                  'R_LC': [R_LC_e]}
else:    
    all_output = {'slope': [slope],
                  'r_BW': [r_b],
                  'rho_BW': [rho_b],
                  'smooth': [smooth],
                  'M_BH': [M_BH],
                  'decay_params': [decay_params],
                  'TDE_rate_solar': [TDE_rate_solar],
                  'TDE_rate_full': [TDE_rate_full],
                  'orb_ens': [orb_ens],
                  'DF': [DF],
                  'q': [q],
                  'LC_flux_solar': [LC_flux_solar],
                  'masses': [masses],
                  'LC_contributions_per_mass': [LC_contributions],
                  'LC_flux_per_mass': [LC_flux_per_mass],
                  'psi_rads': [r_ls],
                  'psi_tot': [psi_r_init],
                  'psi_bh': [psi_bh_init],
                  'psi_enc': [psi_enc_init],
                  'psi_ext': [psi_ext_init],
                  'DF_integrals': [integrals],
                  'mu_integrals': [int_fac_rs],
                  'periods': [periods_e],
                  'mu': [mu_e],
                  'R_LC': [R_LC_e]}

output_table = Table(all_output)
if toy_model_match:
    output_table.write('../Result_Tables/TDE_single_output_toy_model.fits', overwrite=True)
elif nuker:
    output_table.write('../Result_Tables/TDE_single_output_in_{:.2f}_out_{:.2e}_br{:.2e}_rhob{:.2f}.fits'.format(inner_slope,outer_slope,r_b,rho_b), overwrite=True)
else:
    output_table.write('../Result_Tables/TDE_single_output_{:.2f}_{:.2e}_{:.2e}.fits'.format(slope,r_b,rho_b), overwrite=True)


print("=======================================")
print('============= RUN COMPLETE ============')
print("=======================================")
print()


if nuker:
    print('Galxy Density Information: (NUKER)')
    print('Inner Slope: -{:.2f}'.format(inner_slope))
    print('Outer Slope: -{:.2f}'.format(outer_slope))
    print('log(Dens5pc [M_sol/pc^3]): {:.2f}'.format(np.log10(rho_5pc)))
    print('log(M_BH [M_sol]): {:.2f}'.format(np.log10(M_BH/M_sol)))
    print()
    print('Results:')
    print('Total TDE Rate: 10^{:.2f} yr^-1'.format(np.log10(TDE_rate_full)))
else:
    print('Galxy Density Information:')
    print('Slope: -{:.2f}'.format(slope))
    print('log(Dens5pc [M_sol/pc^3]): {:.2f}'.format(np.log10(rho_5pc)))
    print('log(M_BH [M_sol]): {:.2f}'.format(np.log10(M_BH/M_sol)))
    print()
    print('Results:')
    print('Total TDE Rate: 10^{:.2f} yr^-1'.format(np.log10(TDE_rate_full)))
    print('Single Mass TDE Rate: 10^{:.2f} yr^-1'.format(np.log10(TDE_rate_solar)))
#print('TDE rate computation complete.')
#print('Runtime: {:.2f} minutes'.format((TDE_end_time-TDE_start_time)/60))
print()



# =============================================================================
#     # write the output to a fits file
#     c1 = fits.Column(name='TDE_rate_single', array=TDE_rates_single, format='D', unit='yr^-1')
#     c2 = fits.Column(name='TDE_rate', array=TDE_rates, format='D', unit='yr^-1')
#     c3 = fits.Column(name='orb_ens', array=orb_ens, format=str(num_e)+'D', unit='m^2/s^2')
#     c4 = fits.Column(name='DFs', array=DFs, format=str(num_e)+'D')
#     c5 = fits.Column(name='qs', array=qs, format=str(num_e)+'D')
#     c6 = fits.Column(name='LC_fluxes', array=LC_fluxes, format=str(num_e)+'D')
#     c7 = fits.Column(name='psi_rads', array=rs, format=str(num_r)+'D', unit='m^2/s^2')
#     c8 = fits.Column(name='psi_tots', array=psi_tots, format=str(num_r)+'D', unit='m^2/s^2')
#     c9 = fits.Column(name='psi_bhs', array=psi_bhs, format=str(num_r)+'D', unit='m^2/s^2')
#     c10 = fits.Column(name='psi_encs', array=psi_encs, format=str(num_r)+'D', unit='m^2/s^2')
#     c11 = fits.Column(name='psi_exts', array=psi_exts, format=str(num_r)+'D', unit='m^2/s^2')
# 
#     c12 = fits.Column(name='names', array=names, format='A')
#     c13 = fits.Column(name='slopes', array=slopes, format='D')
#     c14 = fits.Column(name='rho_5pc', array=rho_5pc, format='D', unit='M_sol/pc^3')
#     c15 = fits.Column(name='types', array=types, format='I', unit='0=early,1=late')
#     c16 = fits.Column(name='gal_mass', array=gal_masses, format='D', unit='M_sol')
#     c17 = fits.Column(name='bh_mass', array=bh_masses/M_sol, format='D',unit='M_sol')
# 
#     #extention used for the file name of results
#     sample_ext = 'our_gals_w_slope_0.5_to_2.25_reff_{}pc'.format(decay_params_pc[1])
# 
#     t = fits.BinTableHDU.from_columns([c1,c2,c3,c4,c5,c6,c7,c8,c9,c10,c11,c12,c13,c14,c15,c16,c17])
#     t.writeto('../Result_Tables/TDE_output_'+sample_ext+'.fits',overwrite=True)
# 
# =============================================================================





#return TDE_rate_full, orb_ens, DF, q, LC_flux_e, r_ls, psi_r_init, psi_bh_init, psi_enc_init, psi_ext_init


