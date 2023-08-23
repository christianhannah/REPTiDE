#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Jul  9 18:00:25 2022

@author: christian
"""
from numba import jit
from numba import njit
import matplotlib.pyplot as plt
import numpy as np
import pdb
from tqdm import tqdm
import warnings
from scipy import integrate
import scipy.special as s
import time
import mge1d_util as u
warnings.filterwarnings("ignore")

#%%
# ============================ FUNCTIONS ======================================

# =============================================================================
# # =============================================================================
# # function to compute the mass enclosed from density profile 
# ##@njit
# def get_enc_mass(r,slope,rho_infl,r_infl,max_ind):
#     y = rho_infl*(r[0:max_ind+1]/r_infl)**(slope+2)
#     return 4*np.pi*integrate.trapezoid(y, r[0:max_ind+1])
# # =============================================================================
# =============================================================================

# =============================================================================
# function to return power-law inner density profile
##@njit
def get_rho_r(r,slope,rho_infl,r_infl):
    #r_eff = 1.54283879064e+18 # 50 pc
    return rho_infl*(r/r_infl)**(-slope)#*np.exp(-r/r_eff)
# =============================================================================

# =============================================================================
# # =============================================================================
# # function to return power-law inner density profile
# #@njit
# def get_log_rho(r,slope,rho_infl,r_infl,delta):
#     return np.log10(rho_infl*(r/r_infl)**(slope)+delta) 
# # =============================================================================
# =============================================================================

# =============================================================================
# # =============================================================================
# # function to compute the contribution to the potential of the galaxy at 
# # larger radii
# #@njit
# def get_ext_potential(r,slope,rho_infl,r_infl,min_ind):
#     G = 4.301e-3 # pc (km/s)^2 M_sol^-1
#     y = rho_infl*(r[min_ind:]/r_infl)**(slope+1)  
#     return 4*np.pi*G*integrate.trapezoid(y,r[min_ind:])
# # =============================================================================
# =============================================================================

# =============================================================================
# # =============================================================================
# # derive the total gravitational potential (psi(r)) as a function of r
# #@njit(parallel=True)
# def get_psi_r(r,M_BH,slope,rho_infl,r_infl):
#     G = 4.301e-3 # pc (km/s)^2 M_sol^-1
#     psi_1 = G*M_BH/r
#      
#     M_enc = np.zeros_like(r)
#     for i in range(len(M_enc)):
#         M_enc[i] = get_enc_mass(r,slope,rho_infl,r_infl,i)
#     psi_2 = G*M_enc/r
#     
#     psi_3 = np.zeros_like(r)
#     for i in range(len(psi_3)):
#         psi_3[i] = get_ext_potential(r,slope,rho_infl,r_infl,i)
#             
#     return psi_1+psi_2+psi_3
# # =============================================================================
# =============================================================================

# =============================================================================
# define function to find value in an array nearest to a supplied value
#@jit
def find_nearest(array, value):
    array = np.asarray(array)
    idx = np.abs(array - value).argmin()       
    return idx
# =============================================================================   

# =============================================================================
#@njit
def get_psi_t(e,t):
    return (e/2)*(np.tanh(np.pi/2*np.sinh(t))+1)
# =============================================================================

# =========================== END OF FUNCTIONS ================================




# CONSTANTS
pc_to_m = 3.08567758128e16 
M_sol_to_kg = 1.989e30
M_sol =  1.989e30 # kg
R_sol = 696.34e6 # m

G = 6.6743e-11 # m^3 s^-2 kg^-1
#G = 4.301e-3 # pc (km/s)^2 M_sol^-1
#G = 4.301e-3 * 10**6 # pc (m/s)^2 M_sol^-1
#%%

# specify density power-law slope, influence radius, and density at the 
# influence radius
slope = 1.9
#rho_infl = 3e3 # M_sol/pc^3
rho_infl = 3e3*M_sol_to_kg/pc_to_m**3 #kg/m^3
#r_infl = 10 # pc
r_infl = 10*pc_to_m # m

#%%

# We need the mass of our BH which is defined as the mass enclosed in stars 
# within r_infl
#M_BH = integrate.quadrature(get_rho_r,0,r_infl,args=(slope,rho_infl,r_infl),
#                            maxiter=1000)[0] # in M_sol

M_BH = 10**6*M_sol_to_kg

#%%
# =========================== DF Computation ==================================
# =============================================================================
# Compute the stellar distribution function (DF) as a function of specific 
# orbital energy (f(epsilon))
# =============================================================================
# =============================================================================

#e_max = G*M_BH/(10*r_infl)

#e_min = G*M_BH/M_BH**(1/3)

#e_min = 69218960
e_min = 10**8
e_max = 3.5094627e+14
e = np.geomspace(e_min,e_max,10**2)

# STEP 1: Define wide range of t
t_bound = 3
num_t = 10**3
t = np.linspace(-t_bound, t_bound, num_t)
#pdb.set_trace()
integrals = np.zeros_like(e)

print('Computing DF...')
print()
time.sleep(1)
##@njit(parallel=True)
def compute_DF():
    for j in tqdm(range(len(e)), position=0, leave=True):
    #for j in range(len(e)):
        # STEP 2: Compute array of psi(t)
        psi_t = get_psi_t(e[j],t)
        
        # STEP 3: Tabulate rho_r and psi_r over a range of r that ensures the range of
        # values in psi_t is covered by psi_r
        ##@njit(parallel=True)
        def ensure_psi_r_covers_psi_r():
            psi_t_min = np.min(psi_t)#[np.where(psi_t > 1e-6)])
            psi_t_max = np.max(psi_t)
            r_min = 10**-6 *pc_to_m
            r_max = 10**30 *pc_to_m
            r_step_frac = 0.1
            coverage_met = False
            while not coverage_met:
                psi_r_min = G*M_BH/r_max
                psi_r_max = G*M_BH/r_min
    
                if psi_r_min <= psi_t_min and psi_r_max >= psi_t_max:
                    coverage_met = True
                else:
                    if psi_r_min > psi_t_min:
                        r_max += r_step_frac*r_max
                    if psi_r_max < psi_t_max:
                        r_min -= r_step_frac*r_min
            return r_min, r_max
        r_min, r_max = ensure_psi_r_covers_psi_r()
        num_rad = 10**3
        r_ls = np.geomspace(r_min,r_max,num_rad)
        psi_r_init = G*M_BH/r_ls
        rho_r_init = get_rho_r(r_ls,slope,rho_infl,r_infl)

# =============================================================================
#     plt.figure(dpi=500)
#     plt.plot(np.log10(psi_r_init),np.log10(rho_r_init),color='0.7',linewidth=10)
#     plt.xlabel('log($\psi$(r))')
#     plt.ylabel('log($\\rho$(r))')
#     plt.xlim(-6,8)
#     plt.ylim(-37,0)
# =============================================================================
    # STEP 4: Evaluate drho/dpsi at all values of psi_t
        d_rho_d_psi_t = np.zeros_like(psi_t)
        num_r = 10**3
        #pdb.set_trace()
        drhodpsi_ratios = np.zeros(len(psi_t))
        #@jit
        def get_drhodpsi(d_rho_d_psi_t,drhodpsi_ratios):
            for i in range(len(psi_t)):
                if psi_t[i] >= 1e-20:
                    r_ind = find_nearest(psi_r_init,psi_t[i])
                    r_closest = r_ls[r_ind]
                    #print('psi(t)-psi(r) = ',psi_t[i]-psi_r_init[r_ind])
                    #pdb.set_trace()
                    spacing = ((r_closest+0.1*r_closest)-(r_closest-0.1*r_closest))/(num_r-1)
                    #r = np.linspace(r_closest-0.1*r_closest,r_closest+0.1*r_closest,num_r)
                    r = np.arange(0,num_r,1)*spacing+(r_closest-0.1*r_closest)
                    #r = 1
                    psi_r = G*M_BH/r
                    rho_r = get_rho_r(r,slope,rho_infl,r_infl)
                    
                    #plt.plot(np.log10(psi_r),np.log10(rho_r),marker='.')
                    #plt.show()
                    #pdb.set_trace()
                    
                    psi_t_ind = find_nearest(psi_r,psi_t[i])
                    #print('psi(t)-psi(r) = ',psi_t[i]-psi_r[psi_t_ind])
                    d_rho_d_psi_t[i] = (rho_r[psi_t_ind-1]-rho_r[psi_t_ind+1])/(psi_r[psi_t_ind-1]-psi_r[psi_t_ind+1])
                    d_rho_d_psi_analytic = rho_infl*slope*(r_infl/(G*M_BH))**slope*psi_t[i]**(slope-1)
                    drhodpsi_ratios[i] = d_rho_d_psi_t[i]/d_rho_d_psi_analytic
            return d_rho_d_psi_t,d_rho_d_psi_analytic,drhodpsi_ratios
        #a = get_drhodpsi(d_rho_d_psi_t,drhodpsi_ratios)
        d_rho_d_psi_t,d_rho_d_psi_analytic,drhodpsi_ratios = get_drhodpsi(d_rho_d_psi_t,drhodpsi_ratios)
# =============================================================================
#     plt.figure(dpi=500)
#     plt.plot(np.log10(psi_t),drhodpsi_ratios)
#     plt.show()
# =============================================================================
    #pdb.set_trace()
    
        # STEP 5: Use t and drho/dpsi to tabulate drho/dt
        #d_rho_d_t = (e[j]*np.sqrt(4*(np.arctanh(np.tanh(np.pi/2*np.sinh(t))))**2+np.pi**2))/ \
        #        (4*(np.cosh(np.pi/2*np.sinh(t))**2))*d_rho_d_psi_t

        d_rho_d_t = (e[j]/2)*(np.pi/2*np.cosh(t))/(np.cosh(np.pi/2*np.sinh(t)))**2*d_rho_d_psi_t


        #pdb.set_trace()
        # STEP 6: Tabulate the other factor from the double exponential transformed 
        # version of the DF integral
        frac_fac_t = 1/np.sqrt(e[j] - (e[j]/2*(np.tanh(np.pi/2*np.sinh(t)) + 1)))    

        frac_fac_t_analytic = 1/np.sqrt(e[j]-psi_t)


# =============================================================================
#     plt.figure(dpi=500)
#     plt.plot(t,frac_fac_t/frac_fac_t_analytic)
#     plt.show()
# =============================================================================

        integrands = d_rho_d_t * frac_fac_t

        # STEP 7: Evaluate the integral for all values of epsilon (e) using midpoint 
        # reimann sum
        fine_t = np.linspace(-t_bound,t_bound,10**6)
        t_space = fine_t[1]-fine_t[0]
        integrands_interp = np.interp(fine_t,t,integrands)
        integrals[j] = np.sum(integrands_interp*t_space)
        integral = integrate.trapz(integrands_interp,fine_t)
  
# =============================================================================
#     plt.figure(dpi=500)
#     plt.plot(t,integrands,marker='.')
#     plt.plot(fine_t,integrands_interp)
# =============================================================================
    #pdb.set_trace()
  
    return integrals
  
integrals = compute_DF()


    
# STEP 8: Compute the derivative of the integral values vs. epsilon for all 
# all values of epsilon
#%%
integrals_analytic = rho_infl*slope*(r_infl/(G*M_BH))**slope*((np.pi)**(1/2)*e**(slope-1/2)*s.gamma(slope))/s.gamma(slope+1/2)
#pdb.set_trace()
plt.figure(dpi=500)
plt.plot(np.log10(e),np.log10(integrals),linewidth=3)
plt.plot(np.log10(e),np.log10(integrals_analytic))
plt.xlabel('log($\epsilon$)')
#plt.ylabel('integral')
plt.ylabel('log($\int_{0}^{e}\\frac{d\\rho}{d\psi}\\frac{1}{\sqrt{\epsilon-\psi}}d\psi$)')

plt.figure(dpi=500)
plt.plot(np.log10(e),integrals_analytic/integrals)
#plt.plot(e,integrals_analytic)
plt.xlabel('log($\epsilon$)')
plt.ylabel('DF integral ratios')


#%%
d_int_d_e = np.zeros(len(e)-2)
for i in range(len(e)-2):
    d_int_d_e[i] = (integrals[(i+1)-1]-integrals[(i+1)+1])/(e[(i+1)-1]-e[(i+1)+1])


DF = 1/(np.sqrt(8)*np.pi**2*M_sol)*d_int_d_e
e_DF = e[1:-1]


term_1 = slope*(slope-0.5)/(np.sqrt(8)*np.pi**1.5)
term_2 = (rho_infl/M_sol)*((G*M_BH)/r_infl)**(-slope)
term_3 = s.gamma(slope)/s.gamma(slope+0.5)
DF_analytic = term_1*term_2*term_3*e_DF**(slope-1.5)

print('DF computation complete.')
print()
#%%
plt.figure(dpi=500)
plt.plot(np.log10(e_DF),np.log10(DF),color='c',label='Discrete',marker='.')
plt.plot(np.log10(e_DF),np.log10(DF_analytic),'--k',label='Analytic')
plt.ylabel('log(f($\epsilon$))')
plt.xlabel('log($\epsilon$)')
plt.legend()

plt.figure(dpi=500)
plt.plot(np.log10(e_DF),DF/DF_analytic,color='b')
#plt.plot(e_DF,DF_analytic,'--k')
plt.ylabel('f$_{discrete}$/f$_{analytic}$')
plt.xlabel('log($\epsilon$)')

#%%

# =============================================================================
# ============================== n test =======================================
# =============================================================================
# =============================================================================
r_fixed = np.linspace(.1,10,100)*pc_to_m
ns = np.zeros_like(r_fixed)
ns_analytic = np.zeros_like(r_fixed)
ns_exact = np.zeros_like(r_fixed)

for i in range(len(r_fixed)):
    v = np.linspace(0,np.sqrt(2*G*M_BH/r_fixed[i]),10**6)
    e_v = G*M_BH/r_fixed[i] - 0.5*v**2

    DF_v = 10**np.interp(np.log10(e_v),np.log10(e_DF),np.log10(DF))
    DF_v[-1]=0
    
    v_space = v[1]-v[0]

    #adjust the lack of coverage for interpolation using the slope
    #sl = np.mean((np.log10(DF[1:])-np.log10(DF[0:-1]))/(np.log10(e_DF[1:])-np.log10(e_DF[0:-1])))
    #ind_max = find_nearest(e_v,np.min(e_DF))

    #DF_v[ind_max:] = 10**(-sl*(np.log10(e_DF[0])-np.log10(e_v[ind_max:]))+np.log10(DF[0]))

    #DF_v[np.where(np.isnan(DF_v))] = 0

# =============================================================================
#     plt.figure(dpi=500)
#     plt.plot(e_DF,DF,color='b',linestyle='',marker='.', label='original')
#     plt.plot(e_v,DF_v,color='r', label='interpolated')
#     plt.xlabel('$\epsilon$')
#     plt.ylabel('f($\epsilon$)')
#     plt.legend()
# =============================================================================

    DF_v_analytic = term_1*term_2*term_3*e_v**(slope-1.5)
    DF_v_analytic[np.where(np.isnan(DF_v_analytic))] = 0
# =============================================================================
#     plt.figure(dpi=500)
#     plt.plot(v,DF_v,color='b')
#     plt.plot(v,DF_v_analytic,color='r')
#     plt.xlabel('v')
#     plt.ylabel('f(r=2pc,v)')
# =============================================================================


    n = np.sum(DF_v*v_space*v**2)
    #if np.isnan(n): n=0
    n_1 = 4*np.pi*integrate.trapz(DF_v*v**2,v)

    n_2 = 4*np.pi*integrate.trapz(DF_v_analytic*v**2,v)

    n_exact = rho_infl/M_sol*(r_fixed[i]/r_infl)**-slope
    #pdb.set_trace()
    ns[i] = n_1
    ns_analytic[i] = n_2
    ns_exact[i] = n_exact
    
    #pdb.set_trace()
    #print('Difference = ',n_exact-n)

plt.figure(dpi=500)
plt.plot(r_fixed,np.log10(ns_exact),linewidth=3,label='Expression for n')
plt.plot(r_fixed,np.log10(ns),label='Discrete DF')
plt.plot(r_fixed,np.log10(ns_analytic),label='Analytic DF')
#plt.plot(r_fixed,np.log10(ns_exact),linewidth=3,label='Expression for n')
plt.xlabel('r [pc]')
plt.ylabel('log(n [$pc^{-3}$])')
plt.legend()

#%%
plt.figure(dpi=500)
#plt.plot(r_fixed,ns_exact/ns,label='Dicrete DF')
plt.plot(r_fixed,ns_exact/ns_analytic,label='Analytic DF')
plt.xlabel('r [pc]')
plt.ylabel('n$_{expression}$/n$_{DF}$ [$pc^{-3}$]')
plt.legend()
plt.ylim(0.9,1.1)

plt.figure(dpi=500)
plt.plot(r_fixed,ns_exact/ns,label='Dicrete DF')
#plt.plot(r_fixed,ns_exact/ns_analytic,label='Analytic DF')
plt.xlabel('r [pc]')
plt.ylabel('n$_{expression}$/n$_{DF}$ [$pc^{-3}$]')
plt.legend()

#%%

# =============================================================================
# plt.figure(dpi=500)
# plt.plot(np.log10(e_v),np.log10(DF_v),linewidth=4)
# plt.plot(np.log10(e_DF),np.log10(DF))
# #plt.plot(np.log10(e_v[ind_max:]),np.log10(DF_v[ind_max:]))
# =============================================================================
# =============================================================================
# =============================================================================
# =============================================================================
# =============================================================================

#%%
# =============================================================================
# Let's compute the orbit averaged angular momentum diffusion coefficient for 
# highly eccentric orbits, mu(epsilon)
#@njit(parallel=True)
def integrand_p(rs,e):
    psi_r_p = G*M_BH/rs
    return(2*(psi_r_p-e)**(-1/2))

#@njit(parallel=True)
def integrand_I_12(es,psi_i,e_DF,DF):
    DF_interp = 10**np.interp(np.log10(es),np.log10(e_DF),np.log10(DF))
    return (2*(psi_i-es))**(1/2)*DF_interp

#@njit(parallel=True)
def integrand_I_32(es,psi_i,e_DF,DF):
    DF_interp = 10**np.interp(np.log10(es),np.log10(e_DF),np.log10(DF))
    return (2*(psi_i-es))**(3/2)*DF_interp

#@njit(parallel=True)
def integrand_mu(rs_mu,e_DF_i,e_DF,DF,I_0,G,M_BH,M_sol):
    psi_rs_mu = G*M_BH/rs_mu
    I_12_r = np.zeros(len(rs_mu))
    I_32_r = np.zeros(len(rs_mu))
    for j in range(len(rs_mu)):
        psi_i = psi_rs_mu[j]
        es_i = np.linspace(e_DF_i,psi_i,10**3)
        DF_interp_i = 10**np.interp(np.log10(es_i),np.log10(e_DF),np.log10(DF))
        
# =============================================================================
#         plt.figure(dpi=500)
#         plt.plot(np.log10(es_i),np.log10(DF_interp_i),linewidth=4)
#         plt.plot(np.log10(e_DF),np.log10(DF))
#         plt.xlim(np.log10(np.min(es_i)),np.log10(psi_i))
#         plt.show()
#         pdb.set_trace()
# =============================================================================
        
        #I_12_r[j] = (2*(psi_i-e_DF_i))**(-1/2)*integrate.quadrature(integrand_I_12,e_DF_i,psi_i,args=(psi_i,e_DF,DF),maxiter=100)[0]
        #I_32_r[j] = (2*(psi_i-e_DF_i))**(-3/2)*integrate.quadrature(integrand_I_12,e_DF_i,psi_i,args=(psi_i,e_DF,DF),maxiter=100)[0]
        
        I_12_r[j] = (2*(psi_i-e_DF_i))**(-1/2)*integrate.trapz((2*(psi_i-es_i))**(1/2)*DF_interp_i,es_i)
        I_32_r[j] = (2*(psi_i-e_DF_i))**(-3/2)*integrate.trapz((2*(psi_i-es_i))**(3/2)*DF_interp_i,es_i)
    #I_12_r[-1] = 0
    #I_32_r[-1] = 0
    
    J_c_e = G*M_BH/(2*e_DF_i)**(1/2)    
    lim_thing_r = (32*np.pi**2*rs_mu**2*G**2*M_sol**2*np.log(0.4*M_BH/M_sol))/(3*J_c_e**2)* \
                    (3*I_12_r - I_32_r + 2*I_0)

    #pdb.set_trace()
    return lim_thing_r/np.sqrt(2*(psi_rs_mu-e_DF_i))

mu_e = np.zeros(len(e_DF))
periods_e = np.zeros(len(e_DF))
#r_t = 2.2566843e-8*M_BH**(1/3) # in pc
r_t = R_sol*(M_BH/M_sol)**(1/3)
R_LC_e = (4*r_t*e_DF)/(G*M_BH)

int_fac_rs = []

print('Computing q...')
print()
time.sleep(1)
###@njit(parallel=True)
def compute_q():
    for i in tqdm(range(len(e_DF)), position=0, leave=True):
        r_apo = G*M_BH/e_DF[i]
        #rs_p = np.linspace(0,r_apo,10**3)
        #psi_r_p = G*M_BH/rs_p
        periods_e[i] = 2*integrate.quadrature(integrand_p,0,r_apo,args=(e_DF[i]),
                                              maxiter=300)[0]
    
        es = np.geomspace(e_DF[0],e_DF[i],10**3)
        DF_interp = 10**np.interp(np.log10(es),np.log10(e_DF),np.log10(DF))
        I_0 = integrate.trapz(DF_interp,es)
    
        #DF_analytic = term_1*term_2*term_3*es**(slope-1.5)
        #I_0_analytic = integrate.trapz(DF_analytic,es)
    
        #print('I_0 diff = ',I_0-I_0_analytic)
        
        #rs_mu = np.linspace(r_t,r_apo,10**2)
        int_fac_r = integrate.quadrature(integrand_mu,r_t,r_apo,args=(e_DF[i],e_DF,DF,I_0,G,M_BH,M_sol),
                                         tol=10**-35,maxiter=300)[0]
    
        int_fac_rs.append(int_fac_r)
        #pdb.set_trace()
    
        mu_e[i] = 2*int_fac_r/periods_e[i]

    q_discrete = mu_e*periods_e/R_LC_e#*1.02

    print('q computation complete.')
    print()
    return q_discrete, mu_e, periods_e

q_discrete, mu_e, periods_e = compute_q()
# =============================================================================

#%%

### TO-DO ###
# use the DF to calculate the orbit-averaged angular momentum diffusion 
# coefficient for highly eccentric orbits
# Note: I believe that here we want the local diffusion coefficient expressed
# in terms of the DF moments

# =============================================================================
# # let's define the prefactors in the equation for q(epsilon)
# #r_t = 2.2566843e-8*M_BH**(1/3) # in pc
# J_LC = np.sqrt(2*G*M_BH*r_t)
# prefac_1 = (16*np.pi**(1/2)*G**2*M_sol*rho_infl*np.log(0.4*M_BH/M_sol))/(3*J_LC**2)
# prefac_2 = (G*M_BH/r_infl)**(-slope)
# prefac_3 = (s.gamma(slope)*slope*(slope-1/2))/s.gamma(slope+1/2)
# 
# =============================================================================
# =============================================================================
# Q_0 = np.zeros_like(e_DF)
# Q_12 = np.zeros_like(e_DF)
# Q_32 = np.zeros_like(e_DF)
# 
# # =============================================================================
# # G_1 = G*(pc_to_m/M_sol_to_kg)
# # M_BH_1 = M_BH*M_sol_to_kg
# # rho_infl_1 = rho_infl*(M_sol_to_kg/pc_to_m**3)
# # r_infl_1 = r_infl*pc_to_m
# # =============================================================================
# 
# def integrand_r_0(rs,G,M_BH,int_e,e_DF):
#     psi_r_q = G*M_BH/rs
#     return int_e*rs**2/(psi_r_q-e_DF)**(1/2)
# 
# def integrand_r_12(rs,G,M_BH,e_DF):
#     psi_r_q = G*M_BH/rs
#     int_es = np.zeros(len(rs))
#     for i in range(len(int_es)):
#         es = np.linspace(e_DF, psi_r_q[i],10**3)
#         integrand_e = (2*psi_r_q[i]-2*e_DF)**(1/2)*es**(slope-3/2)
#         int_es[i] = integrate.trapz(integrand_e, es)
# 
#     return int_es*rs**2/(2**(1/2)*(psi_r_q-e_DF))
# 
# def integrand_r_32(rs,G,M_BH,e_DF):
#     psi_r_q = G*M_BH/rs
#     int_es = np.zeros(len(rs))
#     for i in range(len(int_es)):
#         es = np.linspace(e_DF, psi_r_q[i],10**3)
#         integrand_e = (2*psi_r_q[i]-2*e_DF)**(3/2)*es**(slope-3/2)
#         int_es[i] = integrate.trapz(integrand_e, es)
# 
#     return int_es*rs**2/(2**(3/2)*(psi_r_q-e_DF)**2)
# 
# 
# # compute the Q values needed
# q = np.zeros(len(e_DF))
# for i in range(len(e_DF)):
#     
#     # define our radial grid for the Q integrals
#     r_apo = G*M_BH/e_DF[i]
#     #r_apo = G_1*M_BH_1/e_DF[i]
#     rs = np.linspace(1e-6,r_apo,10**3)
#     
#     #define psi for these values of r
#     #psi_r_q = get_psi_r(rs,M_BH,slope,rho_infl,r_infl)
#     #psi_r_q = get_psi_r(rs,M_BH_1,slope,rho_infl_1,r_infl_1)
#     psi_r_q = G*M_BH/rs
#     
#     ################## Q_0 ##########################
#     
#     # define our epsilon grid for the Q_0 integral
#     es = np.linspace(0,e_DF[i],10**2)
#     
#     # define the integrand for the epsilon integral
#     integrand_e = es**(slope-3/2)
#     
#     # store the integral over epsilon
#     int_e = integrate.trapz(integrand_e, es)
# 
#     # perform the integral for Q_0(epsilon)
#     Q_0[i] = integrate.quadrature(integrand_r_0,0,r_apo,args=(G,M_BH,int_e,e_DF[i]),
#                                   maxiter=100)[0]
# 
#     #################################################
#     
#     ################## Q_1/2 ########################
#     
#     # perform the integral for Q_1/2(epsilon)
#     Q_12[i] = integrate.quadrature(integrand_r_12,0,r_apo,args=(G,M_BH,e_DF[i]),
#                                   maxiter=100)[0]
#     
#     #################################################
#     
#     ################## Q_3/2 ########################
# 
#     # perform the integral for Q_3/2(epsilon)
#     Q_32[i] = integrate.quadrature(integrand_r_32,0,r_apo,args=(G,M_BH,e_DF[i]),
#                                   maxiter=100)[0]
# 
#     #################################################
#     
#     
#     q[i] = (16*np.pi**(1/2)*G**2*rho_infl*np.log(0.4*M_BH))/(6*G*M_BH*r_t)*(G*M_BH/r_infl)**(-slope)*(3*Q_12[i]-Q_32[i]+2*Q_0[i])
# 
# =============================================================================
#%%

J_LC = np.sqrt(2*G*M_BH*r_t)
prefac_1 = (16*np.pi**(1/2)*G**2*M_sol*rho_infl*np.log(0.4*M_BH/M_sol))/(3*J_LC**2)
prefac_2 = (G*M_BH/r_infl)**(-slope)
prefac_3 = (s.gamma(slope)*slope*(slope-1/2))/s.gamma(slope+1/2)

Q_0_analytic = 5*np.pi/(8*(2*slope-1))*G**3*M_BH**3*e_DF**(slope-4)

# from 2016 paper
#Q_12_analytic_16 = (np.pi)**(1/2)*((1811-798*slope+16*slope**2)/120*s.gamma(4-slope)/s.gamma(15/2-slope)+
#                (-1+2*slope)/(4*(slope-5)*(slope-4))*s.gamma(1/2+slope)/s.gamma(1+slope))*G**3*M_BH**3*e_DF**(slope-4)

# from 2018 paper
Q_12_analytic = (np.pi)**(1/2)*((1811-798*slope+16*slope**2)/120*s.gamma(4-slope)/s.gamma(15/2-slope)+
                (-1+2*slope)/(4*(-slope+5)*(-slope+4))*s.gamma(1/2+slope)/s.gamma(1+slope))*G**3*M_BH**3*e_DF**(slope-4)

Q_32_analytic = (np.pi/(40*s.gamma(slope-3)))*((np.pi**(1/2)*(-325+118*slope+8*slope**2)*(1/np.sin(np.pi*slope)))/s.gamma(15/2-slope)-
                15*(2**(5-2*slope)*(1-2*slope)**2*(2*slope-7)*(2*slope-5)*(2*slope-3)*s.gamma(2*slope-8))/s.gamma(2+slope))*G**3*M_BH**3*e_DF**(slope-4)

#q = (16*np.pi**(1/2)*G**2*rho_infl*np.log(0.4*M_BH))/(6*G*M_BH*r_t)*(G*M_BH/r_infl)**(-slope)*(3*Q_12_analytic-Q_32_analytic+2*Q_0_analytic)
q_analytic = prefac_1*prefac_2*prefac_3*(3*Q_12_analytic-Q_32_analytic+2*Q_0_analytic)
#%%
# =============================================================================
# #%%
# plt.figure(dpi=500)
# plt.plot(np.log10(e_DF),np.log10(Q_0),linewidth=3,color='c',label='Discrete')
# plt.plot(np.log10(e_DF),np.log10(Q_0_analytic),linestyle='--',color='k',label='Analytic')
# plt.xlabel('log($\epsilon$)')
# plt.ylabel('log(Q$_0$)')
# plt.legend()
# #plt.plot(e_DF,Q_0)
# #plt.plot(e_DF,Q_0_analytic)
# 
# plt.figure(dpi=500)
# plt.plot(np.log10(e_DF),np.log10(Q_12),linewidth=3,color='c',label='Discrete')
# plt.plot(np.log10(e_DF),np.log10(Q_12_analytic),linestyle='--',color='k',label='Analytic')
# plt.xlabel('log($\epsilon$)')
# plt.ylabel('log(Q$_{1/2}$)')
# plt.legend()
# 
# plt.figure(dpi=500)
# plt.plot(np.log10(e_DF),np.log10(Q_32),linewidth=3,color='c',label='Discrete')
# plt.plot(np.log10(e_DF),np.log10(Q_32_analytic),linestyle='--',color='k',label='Analytic')
# plt.xlabel('log($\epsilon$)')
# plt.ylabel('log(Q$_{3/2}$)')
# plt.legend()
# 
# 
# 
# plt.figure(dpi=500)
# plt.plot(np.log10(e_DF),Q_0/Q_0_analytic,linewidth=3,color='c',label='Discrete')
# #plt.plot(e_DF,Q_0_analytic,linestyle='--',color='k',label='Analytic')
# plt.xlabel('log($\epsilon$)')
# plt.ylabel('Q$_0$/Q$_{0,analytic}$')
# plt.legend()
# #plt.plot(e_DF,Q_0)
# #plt.plot(e_DF,Q_0_analytic)
# #plt.ylim(0,2)
# 
# plt.figure(dpi=500)
# plt.plot(np.log10(e_DF),Q_12/Q_12_analytic,linewidth=3,color='c',label='Discrete')
# #plt.plot(e_DF,Q_12_analytic,linestyle='--',color='k',label='Analytic')
# plt.xlabel('log($\epsilon$)')
# plt.ylabel('Q$_{1/2}$/Q$_{1/2,analytic}$')
# plt.legend()
# #plt.ylim(0,2)
# 
# plt.figure(dpi=500)
# plt.plot(np.log10(e_DF),Q_32/Q_32_analytic,linewidth=3,color='c',label='Discrete')
# #plt.plot(e_DF,Q_32_analytic,linestyle='--',color='k',label='Analytic')
# plt.xlabel('log($\epsilon$)')
# plt.ylabel('Q$_{3/2}$/Q$_{3/2,analytic}$')
# plt.legend()
# #plt.ylim(0,2)
# =============================================================================

#%%
plt.figure(dpi=500)
plt.plot(np.log10(e_DF),np.log10(q_discrete),color='c',linewidth=3,label='Discrete',marker='.')
plt.plot(np.log10(e_DF),np.log10(q_analytic),color='k',linestyle='--',label='Analytic')
plt.xlabel('log($\epsilon$)')
plt.ylabel('log(q)')
plt.legend()

plt.figure(dpi=500)
plt.plot(np.log10(e_DF),q_discrete/q_analytic,marker='.')#,color='c',linewidth=3,label='Discrete')
#plt.plot(np.log10(e_DF),np.log10(q),color='k',linestyle='--',label='Analytic')
plt.xlabel('log($\epsilon$)')
plt.ylabel('q$_{discrete}$/q$_{analytic}$')
plt.ylim(0.8,1.)

#%%
### TO-DO ###
# Compute the flux of stars that scatter into the loss cone per unit time 
# and energy
import sys
sys.modules['_decimal'] = None
import decimal
decimal.getcontext().prec = 20
decimal.getcontext().Emin = -999999999999999999
decimal.getcontext().Emax = 999999999999999999

# =============================================================================
# def get_F_of_epsilon(e_DF,G,M_BH,r_t,DF,mu_e,q):
#     J_c = G*M_BH/np.sqrt(2*e_DF)
#     R_LC = 4*r_t*e_DF/(G*M_BH)
#     R_0 = np.zeros_like(e_DF)
#     #pdb.set_trace()
#     
#     J_c_d = [] 
#     R_LC_d = []
#     mu_e_d = []
#     DF_d = []
#     R_0_d = []
#     q_d = []
#     for i in range(len(e_DF)):
#         J_c_d.append(decimal.Decimal(J_c[i]))
#         R_LC_d.append(decimal.Decimal(R_LC[i]))
#         mu_e_d.append(decimal.Decimal(mu_e[i]))
#         DF_d.append(decimal.Decimal(DF[i]))
#         #R_0_d.append(decimal.Decimal(R_0[i]))
#         q_d.append(decimal.Decimal(q[i]))
#         exp = decimal.Decimal(math.exp(1))
#         if q[i] > 1:
#             R_0_d.append(R_LC_d[i]*exp**(-q_d[i]))
#         else:
#             R_0_d.append(R_LC_d[i]*exp**(-decimal.Decimal(0.186)*q_d[i]-decimal.Decimal(0.824)*q_d[i]**decimal.Decimal(1/2)))
#     pdb.set_trace()
#     return np.array(decimal.Decimal(4*np.pi**2)*J_c_d**2*mu_e_d*(DF_d/math.log(1/R_0_d)))
# 
# =============================================================================


k=0
def get_F_of_epsilon(e_DF,G,M_BH,r_t,DF,mu_e,q,R_LC,periods_e):
    J_c = G*M_BH/np.sqrt(2*e_DF)
    #R_LC = 4*r_t*e_DF/(G*M_BH)
    ln_R_0 = np.zeros_like(e_DF)
    ln_R_0_1 = np.zeros_like(e_DF)
    #pdb.set_trace()
    
    for i in range(len(e_DF)):
        if q[i] > 1:
            ln_R_0[i] = (q[i] - np.log(R_LC[i]))
        else:
            ln_R_0[i] = ((0.186*q[i]+0.824*np.sqrt(q[i])) - np.log(R_LC[i]))
            
    #print(ln_R_0 - ln_R_0_1)
    #pdb.set_trace()
    return (4*np.pi**2)*periods_e*J_c**2*mu_e*(DF/ln_R_0)
    #return (4*np.pi**2)*J_c**2*q*R_LC*(DF/ln_R_0)

#F = get_F_of_epsilon(e_DF[k:],G,M_BH,r_t,DF[k:],mu_e[k:],q_discrete[k:],R_LC_e[k:])

F = get_F_of_epsilon(e_DF[k:],G,M_BH,r_t,DF[k:],mu_e[k:],q_discrete[k:],R_LC_e[k:],periods_e[k:])

F_analytic = (32*np.pi)/(3*np.sqrt(2))*G**5*M_BH**3*rho_infl**2*np.log(0.4*M_BH)* \
            (G*M_BH/r_infl)**(-2*slope)*((slope*(slope-1/2)*s.gamma(slope))/s.gamma(slope+1/2))**2* \
            e_DF[k:]**(2*slope-11/2)*(3*Q_12_analytic[k:]-Q_32_analytic[k:]+2*Q_0_analytic[k:])/(np.log(G*M_BH/(4*r_t*e_DF[k:]))*(G**3*M_BH**3*e_DF[k:]**(slope-4)))    


#%%

p = u.find_nearest(q_analytic, 1)


#(1/(2*np.pi))*

#F_mod = (6.531288836450116)*F
F_mod = F


k=0
plt.figure(dpi=500)
plt.plot(np.log10(e_DF[k:]),np.log10(F_mod[k:]), label='Discrete',linewidth=3)
plt.plot(np.log10(e_DF[k:]),np.log10(F_analytic[k:]), label='Analytic')
plt.plot([np.log10(e_DF[p]),np.log10(e_DF[p])],[np.min(np.log10(F_mod[k:])),np.max(np.log10(F_analytic[k:]))],
         color='k',linestyle='--')
plt.plot([np.log10(e_DF[-1]),np.log10(e_DF[-1])],[np.min(np.log10(F_mod[k:])),np.max(np.log10(F_analytic[k:]))],
         color='k',linestyle='--', label='q$\leq$1')
plt.ylabel('$log(\mathcal{F}(\epsilon)$)')
plt.xlabel('log($\epsilon$)')
plt.legend()


k=p
plt.figure(dpi=500)
plt.plot(np.log10(e_DF[k:]),np.log10((F_mod[k:])/(F_analytic[k:])))
plt.plot([np.log10(e_DF[p]),np.log10(e_DF[p])],[np.min(np.log10((F_mod[k:])/(F_analytic[k:]))),np.max(np.log10((F_mod[k:])/(F_analytic[k:])))],
         color='k',linestyle='--')
plt.plot([np.log10(e_DF[-1]),np.log10(e_DF[-1])],[np.min(np.log10((F_mod[k:])/(F_analytic[k:]))),np.max(np.log10((F_mod[k:])/(F_analytic[k:])))],
         color='k',linestyle='--', label='q$\leq$1')
plt.ylabel('$log(\mathcal{F}(\epsilon)_{discrete})$/$\mathcal{F}(\epsilon)_{analytic}$)')
plt.xlabel('log($\epsilon$)')
#plt.xlim(np.log10(e_DF[p])-0.2,np.log10(e_DF[-1])+0.2)
plt.ylim()
plt.legend()

#%%

# =============================================================================
# plt.figure(dpi=500)
# plt.plot(np.log10(e_DF),np.log10(e_DF*F), label='Discrete')
# plt.plot(np.log10(e_DF),np.log10(e_DF*F_analytic), label='Analytic')
# plt.ylabel('log($\epsilon\mathcal{F}(\epsilon)$)')
# plt.xlabel('log($\epsilon$)')
# 
# 
# plt.figure(dpi=500)
# plt.plot(e_DF,e_DF*F, label='Discrete')
# #plt.plot(e_DF,e_DF*F_analytic, label='Analytic')
# plt.ylabel('$\epsilon\mathcal{F}(\epsilon)$')
# plt.xlabel('$\epsilon$')
# plt.legend()
# 
# 
# 
# plt.figure(dpi=500)
# plt.plot(np.log10(e_DF),F, label='Discrete')
# plt.plot(np.log10(e_DF),F_analytic, label='Analytic')
# plt.ylabel('$\mathcal{F}(\epsilon)$')
# plt.xlabel('log($\epsilon$)')
# plt.legend()
# ### TO-DO ###
# # Compute the TDE rate by integrating the total flux into the loss cone for 
# # for stars of a given mass, and then by integrating over the stellar mass function.
# 
# 
# 
# =============================================================================






