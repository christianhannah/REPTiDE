#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jun 22 13:31:23 2023

@author: christian
"""

import matplotlib.pyplot as plt
import numpy as np
import pdb
from tqdm import tqdm
import warnings
from scipy import integrate
import time
warnings.filterwarnings("ignore")
import TDE_util as tu
from astropy.table import Table
import sys

# CONSTANTS
pc_to_m = 3.08567758128e16 
M_sol =  1.989e30 # kg
R_sol = 696.34e6 # m
G = 6.6743e-11 # m^3 s^-2 kg^-1

#========================== PROGRAM PARAMS ====================================

# specify the range of specific orbital energies to consider
e_min = 10**6
e_max = 3.5094627e+14


# specify the minimum and maximum stellar mass to consider in the PDMF
M_min = 0.08
M_max = 1


no_print = True

#======================== END PROGRAM PARAMS ==================================

#%%
###############################################################################
###############################################################################
###############################################################################
# =============================================================================
# ============================= MAIN FUNCTION =================================
# =============================================================================
###############################################################################
###############################################################################
###############################################################################

def get_TDE_rate(name,i_eff,r_eff,n,M_BH):

    toy_model_match = False # enforces correct potential functions are used 
                            # toy model functionality uses only the BH potential

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
    if not no_print:
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
                tu.get_psi_r_sersic(r_temp,M_BH,i_eff,r_eff,n,toy_model_match)

            if psi_r_temp[-1] <= psi_t_min and psi_r_temp[0] >= psi_t_max:
                coverage_met = True
            else:
                if psi_r_temp[-1] > psi_t_min:
                    r_max += r_step_frac*r_max
                if psi_r_temp[0] < psi_t_max:
                    r_min -= r_step_frac*r_min
        return r_min, r_max

    r_min, r_max = ensure_psi_r_covers_psi_t(psi_t_min,psi_t_max)
    num_rad = 10**4
    r_ls = np.geomspace(r_min,r_max,num_rad)

    psi_r_init,psi_bh_init,psi_enc_init,psi_ext_init = tu.get_psi_r_sersic(r_ls,M_BH,i_eff,r_eff,n,toy_model_match)
    rho_r_init = tu.get_rho_r_sersic(r_ls,i_eff,r_eff,n,toy_model_match)
    
    if not no_print:
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

    if not no_print:
        print('Computing DF...')
        print()
    time.sleep(1)
    def compute_DF(r_ls,psi_r_init):
        #for j in tqdm(range(len(e)), position=0, leave=True):
        for j in range(len(e)):
            # STEP 2: Compute array of psi(t)
            psi_t = tu.get_psi_t(e[j],t)
        
            # since I am not using the true psi_r to step the radii, double check 
            # that the final radii do indeed give psi_r covering psi_t
            if np.min(psi_r_init) >= np.min(psi_t) or np.max(psi_r_init) <= np.max(psi_t):
                print('*** ERROR: psi(r) failed to cover psi(t) ***')
                sys.exit()
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
                        #psi_r = tu.get_psi_r(r,G,M_BH,slope,rho_b,r_b,smooth,decay_params)
                        psi_r = 10**np.interp(np.log10(r),np.log10(r_ls),np.log10(psi_r_init))
                        rho_r = tu.get_rho_r_sersic(r,i_eff,r_eff,n,toy_model_match)
                    
                        psi_t_ind = tu.find_nearest(psi_r,psi_t[i])
    
                        if psi_r[psi_t_ind]-psi_t[i] >= psi_t[i]*0.001:
                            print('ERROR: Finer grid needed for initial psi_r/psi_t match')
                            sys.exit()
                            #pdb.set_trace()
    
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
  
    integrals = compute_DF(r_ls,psi_r_init)


    # STEP 8: Compute the derivative of the integral values vs. epsilon for all 
    # all values of epsilon
    d_int_d_e = np.zeros(len(e)-2)
    for i in range(len(e)-2):
        d_int_d_e[i] = (integrals[(i+1)-1]-integrals[(i+1)+1])/(e[(i+1)-1]-e[(i+1)+1])


    DF = 1/(np.sqrt(8)*np.pi**2*M_sol)*d_int_d_e
    #DF = 1/(np.sqrt(8)*np.pi**2)*d_int_d_e
    orb_ens = e[1:-1]


    if not no_print:
        print('DF computation complete.')
        print()


# =============================================================================
#     # plot the distribution function
#     plt.figure(dpi=500)
#     plt.plot(np.log10(orb_ens),np.log10(DF),color='c',marker='.')
#     plt.ylabel('log(f($\epsilon$))')
#     plt.xlabel('log($\epsilon$)')
# =============================================================================

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

    #avg_M_sq = tu.get_avg_M_sq()
    avg_M_sq = M_sol**2

    if not no_print:
        print('Computing q...')
        print()
    time.sleep(1)
    
    #def compute_q(r_ref,psi_r_ref):
    def compute_q():
        #for i in tqdm(range(len(orb_ens)), position=0, leave=True):
        for i in range(len(orb_ens)):
            r_apo = G*M_BH/orb_ens[i]
            #rs_p = np.linspace(0,r_apo,10**3)
            #psi_r_p = G*M_BH/rs_p
            r_ref = np.geomspace(10**-6,r_apo,10**3)
            psi_r_ref,psi_bh_ref,psi_enc_ref,psi_ext_ref = tu.get_psi_r_sersic(r_ref,M_BH,i_eff,r_eff,n,toy_model_match)
            
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

        if not no_print:
            print('q computation complete.')
            print()
        return q_discrete, mu_e, periods_e

    q, mu_e, periods_e = compute_q()

    # plot q
# =============================================================================
#     plt.figure(dpi=500)
#     plt.plot(np.log10(orb_ens),np.log10(q),color='c',linewidth=3,marker='.')
#     plt.xlabel('log($\epsilon$)')
#     plt.ylabel('log(q)')
# =============================================================================

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

    # plot the solar LC flux
# =============================================================================
#     plt.figure(dpi=500)
#     plt.plot(np.log10(orb_ens),np.log10(LC_flux_e), linewidth=3)
#     plt.ylabel('$log(\mathcal{F}(\epsilon)$)')
#     plt.xlabel('log($\epsilon$)')
# =============================================================================

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

    PDMF = tu.get_PDMF(masses/M_sol)
    TDE_rate_full = integrate.trapz(LC_contributions*PDMF, masses/M_sol)*sec_per_yr


    #pdb.set_trace()
    all_output = {'name': [name],
                  'radii': [r_ls],
                  'dens': [rho_r_init],
                  'M_BH': [M_BH],
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
                  'R_LC': [R_LC_e]
                  }

    output_table = Table(all_output)

    return output_table


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

###############################################################################
###############################################################################
###############################################################################
# =============================================================================
# ======================== END OF MAIN FUNCTION ===============================
# =============================================================================
###############################################################################
###############################################################################
###############################################################################


