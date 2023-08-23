#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun May 29 16:36:01 2022

@author: christian
"""

import matplotlib.pyplot as plt
import numpy as np
from scipy import integrate
import mge1d_util as u
from scipy.spatial import ConvexHull
import sys
import pdb
from astropy.io import fits

#%%

# =============================================================================
# ===================== SPECIFY SETUP PARAMETERS ==============================
# =============================================================================

# switch for which GSMF we use (0 = Driver+22, 1 = Baldry+12)
gsmf_select = 0
if gsmf_select == 0:
    gsmf_ext = 'driver22'
elif gsmf_select == 1:
    gsmf_ext = 'baldry12'

# specify the total volume to be simulated
R = 100*10**6 # pc
vol_ext = '_R_{:.2f}Mpc'.format(R/10**6)

print('##############################')
print('###### STARTING BUILD ########')
print('##############################')
print()
# =============================================================================
# =============================================================================
# =============================================================================




# =============================================================================
# =================== Get the desired GSMF and compute CDF ====================
# =============================================================================


# function to compute the GSMF given the Schecter parameters
def get_gsmf(M, M_break, phi, alpha):
    # compute GSMF (number density of gals in [dex^-1 Mpc^-3])
    if len(phi) == 1:
        return np.log(10)*np.exp(-10**(np.log10(M)-np.log10(M_break)))*\
        (phi[0]*(10**(np.log10(M)-np.log10(M_break)))**(alpha[0]+1))
    else:
        return np.log(10)*np.exp(-10**(np.log10(M)-np.log10(M_break)))*\
            (phi[0]*(10**(np.log10(M)-np.log10(M_break)))**(alpha[0]+1) +\
             phi[1]*(10**(np.log10(M)-np.log10(M_break)))**(alpha[1]+1))



# define the range of masses over which to evaluate the GSMF
min_mass = 7.5
max_mass = 11.0
logM = np.arange(min_mass, max_mass, 0.00001)
M = 10**logM


if gsmf_select == 1: # use Baldry+12 GSMFs

    # get the GSMF for blue galaxies
    M_brk_b = 10**10.72
    phi_b = [0.71*10**(-3)]
    alpha_b = [-1.45]
    gsmf_b = get_gsmf(M, M_brk_b, phi_b, alpha_b)

    # get the GSMF for red galaxies
    M_brk_r = 10**10.72
    phi_r = [3.25*10**(-3), 0.08*10**(-3)]
    alpha_r = [-0.45, -1.45]
    gsmf_r = get_gsmf(M, M_brk_r, phi_r, alpha_r)

elif gsmf_select == 0: # use Driver+22 GSMFs

    # let's look at the morphological results of Driver et al. (2022)
    M_brk_e = 10**10.954
    phi_e = [10**(-2.994+0.0866)]
    alpha_e = [-0.524]
    gsmf_e = get_gsmf(M, M_brk_e, phi_e, alpha_e)

    # let's try adding in the C group from Driver+22 to early-type GSMF
    M_brk_ce = 10**11.170
    phi_ce = [10**(-6.419+0.0866)]
    alpha_ce = [-1.978]
    gsmf_ce = get_gsmf(M, M_brk_ce, phi_ce, alpha_ce)

    gsmf_r = gsmf_e+gsmf_ce


    M_brk_l = 10**10.436
    phi_l = [10**(-3.332+0.0866)]
    alpha_l = [-1.569]
    gsmf_l = get_gsmf(M, M_brk_l, phi_l, alpha_l)
    
    # let's try adding in the cBD and dBD groups from Driver+22 to late-type GSMF
    M_brk_c = 10**10.499
    phi_c = [10**(-2.469+0.0866)]
    alpha_c = [0.003]
    gsmf_c = get_gsmf(M, M_brk_c, phi_c, alpha_c)
    M_brk_d = 10**10.513
    phi_d = [10**(-3.065+0.0866)]
    alpha_d = [-1.264]
    gsmf_d = get_gsmf(M, M_brk_d, phi_d, alpha_d)

    gsmf_b = gsmf_l+gsmf_d+gsmf_c

else:
    sys.exit("ERROR: Must select proper GSMF selection value.")
#%%

#================= FANCY PLOT OF BOTH GSMFs w/ MASS BOUNDS ===================#

# define Baldry GSMFs just for plotting
# get the GSMF for blue galaxies
Bald_M_brk_b = 10**10.72
Bald_phi_b = [0.71*10**(-3)]
Bald_alpha_b = [-1.45]
Bald_gsmf_b = get_gsmf(M, Bald_M_brk_b, Bald_phi_b, Bald_alpha_b)

# get the GSMF for red galaxies
Bald_M_brk_r = 10**10.72
Bald_phi_r = [3.25*10**(-3), 0.08*10**(-3)]
Bald_alpha_r = [-0.45, -1.45]
Bald_gsmf_r = get_gsmf(M, Bald_M_brk_r, Bald_phi_r, Bald_alpha_r)

ylims = (2e-6, 0.4)

# plot the GSMFs we will be using
plt.figure(dpi=500)
plt.plot(logM,Bald_gsmf_b, color='b', linestyle=':', alpha=0.3)
plt.plot(logM,Bald_gsmf_r, color='r', linestyle=':', alpha=0.3)
Bald_min_ind_r = u.find_nearest(logM, 8.4867)
Bald_max_ind_r = u.find_nearest(logM, 11.5033)
Bald_min_ind_b = u.find_nearest(logM, 8.0980)
Bald_max_ind_b = u.find_nearest(logM, 11.3007)
plt.plot(logM[Bald_min_ind_b:Bald_max_ind_b+1],Bald_gsmf_b[Bald_min_ind_b:Bald_max_ind_b+1], color='b',
         linewidth=2.5, alpha=0.3)
plt.plot(logM[Bald_min_ind_r:Bald_max_ind_r+1],Bald_gsmf_r[Bald_min_ind_r:Bald_max_ind_r+1], color='r',
         linewidth=2.5, alpha=0.3)

# add in mass limits from relation fits
min_ind_r = u.find_nearest(logM, 6.875)
max_ind_r = u.find_nearest(logM, 11.625)
min_ind_b = u.find_nearest(logM, 6.625)
max_ind_b = u.find_nearest(logM, 11.375)
plt.plot(logM[min_ind_b:max_ind_b+1],gsmf_b[min_ind_b:max_ind_b+1], color='b',
         linewidth=5.0, alpha=0.4)
plt.plot(logM[min_ind_r:max_ind_r+1],gsmf_r[min_ind_r:max_ind_r+1], color='r',
         linewidth=5.0, alpha=0.4)
plt.plot(logM,gsmf_b, color='b',label='Late')
plt.plot(logM,gsmf_r, color='r',label='Early')

plt.xlabel('log(M/M$_\odot$)')
plt.ylabel('number density [dex$^{-1}$ Mpc$^{-3}$]')
plt.yscale('log')
plt.xlim(6.3,11.9)
plt.ylim(ylims)
plt.legend()

#=============================================================================#

#%%

# first off we need the PDF (integrate the GSMF and normalize)
tot_r = integrate.trapezoid(gsmf_r, logM)
tot_b = integrate.trapezoid(gsmf_b, logM)
pdf_r = gsmf_r/tot_r
pdf_b = gsmf_b/tot_b

# =============================================================================
# # plot the PDFs of each GSMF
# plt.figure(dpi=500)
# plt.title('PDF')
# plt.plot(logM, pdf_r, color='r', label='Early ')
# plt.plot(logM, pdf_b, color='b', label='Late')
# #plt.ylabel('Probability Density')
# plt.xlabel('log(M/M$_\odot$)')
# plt.legend()
# =============================================================================

# now let's use the PDF's to compute the CDF's
cdf_r = np.cumsum(pdf_r)
cdf_r = cdf_r/cdf_r[-1]
cdf_b = np.cumsum(pdf_b)
cdf_b = cdf_b/cdf_b[-1]

# =============================================================================
# # plot the CDFs of each GSMF
# plt.figure(dpi=500)
# plt.title('CDF')
# plt.plot(logM, cdf_r, color='r', label='Early (Baldry+12)')
# plt.plot(logM, cdf_b, color='b', label='Late (Baldry+12)')
# #plt.ylabel('Probability Density')
# plt.xlabel('log(M/M$_\odot$)')
# plt.legend()
# =============================================================================


# =============================================================================
# =============================================================================
# =============================================================================




# =============================================================================
# =================== Build Model Galaxy Sample ===============================
# =============================================================================

def spherical_vol(r):
    return (4/3)*np.pi*r**3

# get the total volume of simulation for file names
tot_vol = spherical_vol(R/10**6)
print('Total Volume: {:.2f} Mpc^3'.format(tot_vol))
print()

# specify the total number of galaxies in our sample using volume and GSMFs
minM = u.find_nearest(logM, min_mass)
maxM = u.find_nearest(logM, max_mass)
galdens_r = integrate.trapezoid(gsmf_r[minM:maxM+1], logM[minM:maxM+1]) # gals/Mpc^3
galdens_b = integrate.trapezoid(gsmf_b[minM:maxM+1], logM[minM:maxM+1]) # gals/Mpc^3
ngal_r = int(galdens_r*spherical_vol(R/10**6))
ngal_b = int(galdens_b*spherical_vol(R/10**6))

print('# of Early-types: {}'.format(ngal_r))
print('# of Late-types: {}'.format(ngal_b))
print()

##### MASS ASSIGNMENT #####

# specify the random draws from uniform distribution to be used with CDF to 
# assign masses
draws_r = np.random.uniform(size=ngal_r)
draws_b = np.random.uniform(size=ngal_b)
masses_r = np.zeros(len(draws_r))
masses_b = np.zeros(len(draws_b))

print('Assigning masses...')


# assign the masses using the CDF and the random draws
interp_width = 0.01
for i in range(len(draws_r)):
    x = np.arange(-1,1+0.01,0.01)*0.01+draws_r[i]
    y = np.interp(x, cdf_r, logM)
    masses_r[i] = 10**(y[100])
for i in range(len(draws_b)):
    x = np.arange(-1,1+0.01,0.01)*0.01+draws_b[i]
    y = np.interp(x, cdf_b, logM)
    masses_b[i] = 10**(y[100])

print('MASSES DONE')
print()

# store the masses in log form as well
logmasses_r = np.log10(masses_r)
logmasses_b = np.log10(masses_b)

# =============================================================================
# # plot the distributions of galaxy masses by type
# plt.figure(dpi=500)
# blah, bins_r, blah = plt.hist(np.log10(masses_r), bins=50, color='m', alpha=0.7)
# plt.hist(np.log10(masses_b), bins=bins_r, color='c', alpha=0.7)
# plt.ylabel('# of galaxies')
# plt.xlabel('log(M/M$_\odot$)')
# =============================================================================


##### DENSITY AND SLOPE ASSIGNMENT #####

# use the masses with our relations to assign the central densities and 
# power-law slopes
# function to get a power-slope value from our relations for a galaxy of log(mass), logm
def get_ets(logm):
    # get the mean of the normal distribution from the mass and our relation
    mean = 0.1863*(logm - 9.) - 1.6776
    scat = 0.6331 # our scatter from linmix fit
    return np.random.normal(loc=mean,scale=scat)
# function to get a central density value from our relations for a galaxy of log(mass), logm
def get_etd(logm):
    # get the mean of the normal distribution from the mass and our relation
    mean = 0.7151*(logm - 9.) + 3.3998
    scat = 0.4000 # our scatter from linmix fit
    return np.random.normal(loc=mean,scale=scat)
# function to get a power-slope value from our relations for a galaxy of log(mass), logm
def get_lts(logm):
    # get the mean of the normal distribution from the mass and our relation
    mean = 0.1872*(logm - 9.) -2.1876
    scat = 0.5931 # our scatter from linmix fit
    return np.random.normal(loc=mean,scale=scat)
# function to get a central density value from our relations for a galaxy of log(mass), logm
def get_ltd(logm):
    # get the mean of the normal distribution from the mass and our relation
    mean = 0.6186*(logm - 9.) + 2.8454
    scat = 0.4572 # our scatter from linmix fit
    return np.random.normal(loc=mean,scale=scat)

print('Assigning slopes and densities...')

slopes_r = np.zeros_like(masses_r)
densities_r = np.zeros_like(masses_r)
slopes_b = np.zeros_like(masses_b)
densities_b = np.zeros_like(masses_b)
for i in range(len(masses_r)):
    slopes_r[i] = get_ets(logmasses_r[i])
    densities_r[i] = get_etd(logmasses_r[i])
for i in range(len(masses_b)):
    slopes_b[i] = get_lts(logmasses_b[i])
    densities_b[i] = get_ltd(logmasses_b[i])

print('SLOPES/DENSITIES DONE')
print()
#%%
##### DISTANCE ASSIGNMENT #####

dr = R/10**3
r = np.arange(0, R+dr, dr)
rad_cdf = spherical_vol(r)/spherical_vol(R)
rad_pdf = 3*r**2/R**3

print('Assigning distances...')

# use the CDF of the radii to assign distances
draws_r_rad = np.random.uniform(size=ngal_r)
draws_b_rad = np.random.uniform(size=ngal_b)
radii_r = np.zeros_like(draws_r_rad)
radii_b = np.zeros_like(draws_b_rad)
for i in range(len(draws_r_rad)):
    x = np.arange(-1,1+0.01,0.01)*0.01+draws_r_rad[i]
    y = np.interp(x, rad_cdf, r)
    radii_r[i] = y[100]
for i in range(len(draws_b_rad)):
    x = np.arange(-1,1+0.01,0.01)*0.01+draws_b_rad[i]
    y = np.interp(x, rad_cdf, r)
    radii_b[i] = y[100]

print('DISTANCES DONE')
print()    

# =============================================================================
# # plot the distributions of radii by type
# plt.figure(dpi=500)
# blah, bins_r_rad, blah = plt.hist(radii_r/10**6, bins=100, color='m', alpha=0.7)
# plt.hist(radii_b/10**6, bins=bins_r_rad, color='c', alpha=0.7)
# plt.ylabel('# of galaxies')
# plt.xlabel('Distance [Mpc]')
# =============================================================================

# =============================================================================
# # plot the PDF of the radii
# plt.figure(dpi=500)
# plt.title('PDF')
# plt.plot(r/10**6, rad_pdf, label='PDF')
# plt.xlabel('Radius [Mpc]')
# params_pdf = np.polyfit(r, rad_pdf, deg=2)
# plt.plot(r/10**6, params_pdf[0]*r**2+params_pdf[1]*r+params_pdf[2], 
#          linestyle='--', label='2nd order poly')
# plt.text(0.7, 0.25, '$\propto r^2$',color='k', transform=plt.gcf().transFigure, size=20)
# plt.legend(loc='upper left')
# 
# # plot the CDF of the radii
# plt.figure(dpi=500)
# plt.title('CDF')
# plt.plot(r/10**6, rad_cdf, label='CDF')
# plt.xlabel('Radius [Mpc]')
# params_cdf = np.polyfit(r, rad_cdf, deg=3)
# plt.plot(r/10**6, params_cdf[0]*r**3+params_cdf[1]*r**2+params_cdf[2]*r+params_cdf[3], 
#          linestyle='--',label='3rd order poly')
# plt.text(0.7, 0.25, '$\propto r^3$',color='k', transform=plt.gcf().transFigure, size=20)
# plt.legend(loc='upper left')
# =============================================================================


#%%

##### BH Mass Assignment #####
BH_mass_ext = '_reines15'

# functions using relations from Reines et al. 2015
def get_BH_mass_lt(m_gal):
    mean = 7.45 + 1.05*np.log10(m_gal/10**11) # gives log(M_BH)
    scat = 0.55 #dex
    return np.random.normal(loc=mean,scale=scat)
def get_BH_mass_et(m_gal):
    mean = 8.95 + 1.40*np.log10(m_gal/10**11) # gives log(M_BH)
    scat = 0.47 #dex
    return np.random.normal(loc=mean,scale=scat)

print('Assigning BH masses...')

BH_masses_r = np.zeros_like(masses_r)
BH_masses_b = np.zeros_like(masses_b)
for i in range(len(masses_r)):
    BH_masses_r[i] = get_BH_mass_et(masses_r[i])
for i in range(len(masses_b)):
    BH_masses_b[i] = get_BH_mass_lt(masses_b[i])

print('BH MASSES DONE')
print()
print('##############################')
print('###### BUILD COMPLETE ########')
print('##############################')

# =============================================================================
# # plot the BH masses vs galaxy logmass
# plt.figure(dpi=500)
# plt.title('Reines et al. 2015 Relations')
# plt.plot(logmasses_r, BH_masses_r, linestyle='', marker='.', color='m', alpha=0.5)
# plt.plot(logmasses_b, BH_masses_b, linestyle='', marker='.', color='c', alpha=0.5)
# plt.plot(logM, 1.40*np.log10(M/10**11)+8.95, linestyle='--', color='r')
# plt.plot(logM, 1.05*np.log10(M/10**11)+7.45, linestyle='--', color='b')
# plt.xlabel('log(M$_*$/M$_\odot$)')
# plt.ylabel('log(M$_{BH}$/M$_\odot$)')
# =============================================================================

#%%

# =============================================================================
# # plot the slopes vs galaxy logmass
# plt.figure(dpi=500)
# plt.plot(logmasses_r, slopes_r, linestyle='', marker='.', color='m', alpha=0.5)
# plt.plot(logmasses_b, slopes_b, linestyle='', marker='.', color='c', alpha=0.5)
# plt.plot(logM, 0.1863*(logM - 9.) - 1.6776,color='r',linestyle='--')
# plt.plot(logM, 0.1872*(logM - 9.) -2.1876,color='b',linestyle='--')
# plt.ylabel('$\gamma$')
# plt.xlabel('log(M/M$_\odot$)')
# 
# # plot the densities vs galaxy logmass
# plt.figure(dpi=500)
# plt.plot(logmasses_r, densities_r, linestyle='', marker='.', color='m', alpha=0.5)
# plt.plot(logmasses_b, densities_b, linestyle='', marker='.', color='c', alpha=0.5)
# plt.plot(logM, 0.7151*(logM - 9.) + 3.3998,color='r',linestyle='--')
# plt.plot(logM, 0.6186*(logM - 9.) + 2.8454,color='b',linestyle='--')
# plt.xlabel('log(M/M$_\odot$)')
# plt.ylabel('log($\\rho_{5pc}$ [M$_\odot$/pc$^3$])')
# =============================================================================


# =============================================================================
# =============================================================================
# =============================================================================



#%%
# =============================================================================
# ================== 2D PLOTS OF MODEL PARAMETER SPACE ========================
# =============================================================================

min_BH = np.min(np.concatenate((BH_masses_r,BH_masses_b)))
max_BH = np.max(np.concatenate((BH_masses_r,BH_masses_b)))

# plot the slopes vs density colored by BH mass for Early-types
plt.figure(dpi=500)
plt.scatter(slopes_b, densities_b ,c=BH_masses_b,marker='.',s=0.7,cmap='winter', alpha=0.6)
plt.colorbar(label='log(M$_{BH}$/M$_\odot$)', pad=-0.106)
plt.clim(min_BH,max_BH)
plt.scatter(slopes_r, densities_r ,c=BH_masses_r,marker='.',s=0.7,cmap='autumn', alpha=0.6)
plt.colorbar(pad=0.03).set_ticks([])
plt.clim(min_BH,max_BH)
plt.xlabel('$\gamma$')
plt.ylabel('log($\\rho_{5pc}$ [M$_\odot$/pc$^3$])')

plt.savefig('../Plots/Model_Galaxy_Sample/slopes_v_densities_colored_by_BH_mass_{:.2f}_Mpc3.png'.format(tot_vol),
            bbox_inches='tight', pad_inches=0.1, dpi=500)

min_galm = np.min(np.concatenate((logmasses_r,logmasses_b)))
max_galm = np.max(np.concatenate((logmasses_r,logmasses_b)))

# plot the slopes vs density colored by BH mass for Early-types
plt.figure(dpi=500)
plt.scatter(slopes_b, densities_b ,c=logmasses_b,marker='.',s=0.7,cmap='winter', alpha=0.6)
plt.colorbar(label='log(M$_{\star}$/M$_\odot$)', pad=-0.106)
plt.clim(min_galm,max_galm)
plt.scatter(slopes_r, densities_r ,c=logmasses_r,marker='.',s=0.7,cmap='autumn', alpha=0.6)
plt.colorbar(pad=0.03).set_ticks([])
plt.clim(min_galm,max_galm)
plt.xlabel('$\gamma$')
plt.ylabel('log($\\rho_{5pc}$ [M$_\odot$/pc$^3$])')

plt.savefig('../Plots/Model_Galaxy_Sample/slopes_v_densities_colored_by_galmass_{:.2f}_Mpc3.png'.format(tot_vol),
            bbox_inches='tight', pad_inches=0.1, dpi=500)

# =============================================================================
# =============================================================================
# =============================================================================



#%%

# =============================================================================
# ================== 3D PLOTS OF MODEL PARAMETER SPACE ========================
# =============================================================================

points = np.zeros((len(slopes_r)+len(slopes_b),3))
points[:,0] = np.concatenate((slopes_r,slopes_b))
points[:,1] = np.concatenate((densities_r, densities_b))
points[:,2] = np.concatenate((BH_masses_r, BH_masses_b))
hull = ConvexHull(points)

vert_x = np.zeros(len(hull.vertices))
vert_y = np.zeros(len(hull.vertices))
vert_z = np.zeros(len(hull.vertices))
count = 0
for vertices in hull.vertices:
    vert_x[count] = points[vertices, 0]
    vert_y[count] = points[vertices, 1]
    vert_z[count] = points[vertices, 2]
    count+=1

#get convex hull
hullv = np.zeros((len(hull.vertices),3))
hullv[:,0] = vert_x
hullv[:,1] = vert_y
hullv[:,2] = vert_z
hullv = np.transpose(hullv)         
            
#fit ellipsoid on convex hull
eansa = u.ls_ellipsoid(hullv[0],hullv[1],hullv[2]) #get ellipsoid polynomial coefficients
center,axes,inve = u.polyToParams3D(eansa,False)   #get ellipsoid 3D parameters

plt.figure(dpi=500)
ax = plt.axes(projection='3d')
ax.scatter3D(points[:len(slopes_r),0],points[:len(slopes_r),1],points[:len(slopes_r),2], 
             s=0.5, color='', alpha=0.5)
ax.scatter3D(points[len(slopes_r):,0],points[len(slopes_r):,1],points[len(slopes_r):,2], 
             s=0.5, color='', alpha=0.5)


# let's rotate our vertices onto the fitted ellipse to increase size until all 
# vertices are enclosed
vert_x_rot = np.zeros_like(vert_x)
vert_y_rot = np.zeros_like(vert_y)
vert_z_rot = np.zeros_like(vert_z)
for i in range(len(vert_x)):
    b = np.zeros((3,1))
    b[0,0] = vert_x[i] - center[0]
    b[1,0] = vert_y[i] - center[1]
    b[2,0] = vert_z[i] - center[2]
    vert_x_rot[i], vert_y_rot[i], vert_z_rot[i] = inve.dot(b) 

# we will grow the size of the ellipse axes by 0.01% until all vertices are contained
def ell_value(x,y,z,axes):
    return (x/axes[0])**2+(y/axes[1])**2+(z/axes[2])**2

all_fit = False
increase = 0.0001
axes_temp = axes
while not all_fit:
    for i in range(len(vert_x_rot)):
        if ell_value(vert_x_rot[i],vert_y_rot[i],vert_z_rot[i],axes_temp) > 1:
            axes_temp += axes_temp*increase
            break
        if i == len(vert_x_rot)-1:
            all_fit = True
            axes_final = axes_temp
        
rx = axes_final[0]
ry = axes_final[1]
rz = axes_final[2]
phi = np.linspace(0, 2*np.pi, 100)
theta = np.linspace(0, np.pi, 100)

x = rx*np.outer(np.cos(phi), np.sin(theta)) 
y = ry*np.outer(np.sin(phi), np.sin(theta)) 
z = rz*np.outer(np.ones_like(phi), np.cos(theta)) 
x_1 = np.zeros_like(x)
y_1 = np.zeros_like(y)
z_1 = np.zeros_like(z)

for i in range(len(x)):
    for j in range(len(x)):
        b = np.zeros((3,1))
        b[0,0] = x[i,j]
        b[1,0] = y[i,j]
        b[2,0] = z[i,j]
        x_1[i,j], y_1[i,j], z_1[i,j] = inve.transpose().dot(b) + np.array([[center[0]],[center[1]],[center[2]]])

ax.plot_surface(x_1,y_1,z_1,color='c',alpha=0.3)


# let's make a grid of points for the models we will be running
max_slope = np.max(points[:,0])
min_slope = np.min(points[:,0])

max_dens = np.max(points[:,1])
min_dens = np.min(points[:,1])

max_bh = np.max(points[:,2])
min_bh = np.min(points[:,2])

num_grid = 10
ss = np.linspace(min_slope,max_slope,num=num_grid)
ds = np.linspace(min_dens,max_dens,num=num_grid)
ms = np.linspace(min_bh,max_bh,num=num_grid)


s = np.repeat(ss,num_grid**2)
d = np.tile(np.repeat(ds,num_grid),num_grid)
m = np.tile(ms,num_grid**2)
#d = np.repeat(ds,num_grid**2)
#m = np.repeat(ms,num_grid**2)
    

flags = []
for i in range(len(s)):
    x = s[i]
    y = d[i]
    z = m[i]
    b = np.zeros((3,1))
    b[0,0] = x - center[0]
    b[1,0] = y - center[1]
    b[2,0] = z - center[2]
    x_rot, y_rot, z_rot = inve.dot(b) 
            
    if  ell_value(x_rot,y_rot,z_rot,axes_final) > 1:
        flags.append(i)
flags = np.array(flags)

# =============================================================================
# s_flags = []
# d_flags = []
# m_flags = []
# for i in range(len(s)):
#     for j in range(len(d)):
#         for k in range(len(m)):
#             x = s[i]
#             y = d[j]
#             z = m[k]
#         
#             b = np.zeros((3,1))
#             b[0,0] = x - center[0]
#             b[1,0] = y - center[1]
#             b[2,0] = z - center[2]
#             x_rot, y_rot, z_rot = inve.dot(b) 
#             
#             if  ell_value(x_rot,y_rot,z_rot,axes_final) > 1:
#                 s_flags.append(i) 
#                 d_flags.append(j)  
#                 m_flags.append(k)  
# 
# s_flags = np.array(s_flags)
# d_flags = np.array(d_flags)
# m_flags = np.array(m_flags)
# =============================================================================

s_new = np.delete(s,flags)
d_new = np.delete(d,flags)
m_new = np.delete(m,flags)

points = np.zeros((len(s_new),3))
points[:,0] = s_new
points[:,1] = d_new
points[:,2] = m_new

         
ax.scatter3D(points[:,0],points[:,1],points[:,2], 
             s=1.5, color='b', alpha=0.9)





ax.set_xlabel('$\gamma$')
ax.set_ylabel('log($\\rho_{5pc}$ [M$_\odot$/pc$^3$])')
ax.set_zlabel('log(M$_{BH}$/M$_\odot$)')

# uncomment below to show vertices of convex hull
#for vertices in hull.vertices:
#    ax.plot(points[vertices, 0], points[vertices, 1], points[vertices,2], 'k.')

# uncomment below to alter viewing angle
#ax.view_init(20,125)

plt.savefig('../Plots/Model_Galaxy_Sample/3D_parameter_ellipsoid_{:.2f}_Mpc3_with_model_points.png'.format(tot_vol),
            bbox_inches='tight', pad_inches=0.1, dpi=500)
# =============================================================================
# =============================================================================
# =============================================================================

#%%

# =============================================================================
# ================= SAVE MODEL SAMPLE DATA TO FITS FILE =======================
# =============================================================================

numbers = np.arange(ngal_r+ngal_b)+1

# type = 0 -> early; type = 1 -> late
types = np.zeros(ngal_r+ngal_b).astype(int)
types[ngal_r:] = 1
all_radii = np.concatenate((radii_r,radii_b))
all_galmass = np.concatenate((logmasses_r,logmasses_b))
all_bhmass = np.concatenate((BH_masses_r, BH_masses_b))
all_gamma = np.concatenate((slopes_r,slopes_b))
all_rho = np.concatenate((densities_r, densities_b))


c1 = fits.Column(name='no.', array=numbers, format='I')
c2 = fits.Column(name='type', array=types, format='I', unit='0=early,1=late')
c3 = fits.Column(name='dist', array=all_radii/10**6, format='D', unit='Mpc')
c4 = fits.Column(name='galmass', array=all_galmass, format='D', unit='log(M_sol)')
c5 = fits.Column(name='bhmass', array=all_bhmass, format='D', unit='log(M_sol)')
c6 = fits.Column(name='gamma', array=all_gamma, format='D')
c7 = fits.Column(name='rho5pc', array=all_rho, format='D', unit='M_sol/pc^3')

t = fits.BinTableHDU.from_columns([c1,c2,c3,c4,c5,c6,c7])
t.writeto('../Result_Tables/model_galaxy_sample_'+gsmf_ext+
          BH_mass_ext+vol_ext+'.fits',clobber=True)

# =============================================================================
# =============================================================================
# =============================================================================





