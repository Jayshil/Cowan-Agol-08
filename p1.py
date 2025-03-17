import numpy as np
import matplotlib.pyplot as plt
import plotstyles
from kelp.jax import thermal_phase_curve
from utils import CowanPC_model
import astropy.units as u
from kelp import Filter
import batman

# Simulated data for GJ-367 b (From Zhang et al. 2024)

# Planetary parameters Goggo et al. (2023) unless otherwise noted
per, per_err = 0.3219225, 0.0000002
tc, tc_err = 2458544.13635, 0.00040
bb, bb_err = 0.584, np.sqrt(0.034**2 + 0.037**2)
inc, inc_err = 79.89, np.sqrt(0.87**2 + 0.85**2)
rprs, rprs_err = 0.01399, 0.00028
rho, rho_err = 6.70*1e3, np.sqrt(0.62**2 + 0.55**2) * 1e3
rst = 0.458
hotoff, c11, fprime = -10., 0.4, 1/np.sqrt(2)

ar_post = np.random.normal(bb, bb_err, 10000) / np.cos(np.radians(np.random.normal(inc, inc_err, 10000)))
ar, ar_err = np.nanmedian(ar_post), np.nanstd(ar_post)

tims = np.linspace(tc-per, tc+per, 10000)

# Transit and occultation models
pars = batman.TransitParams()
pars.t0 = tc
pars.per = per
pars.rp = rprs
pars.a = ar
pars.inc = np.rad2deg(np.arccos(bb / ar))
pars.ecc = 0.
pars.w = 90.
pars.u = [0.1, 0.3]
pars.limb_dark = 'quadratic'

m = batman.TransitModel(pars, tims)
flux_tra = m.light_curve(pars)

# Occultation model
pars.fp = 100e-6
pars.t_secondary = tc + (per/2)

m1 = batman.TransitModel(pars, t=tims, transittype='secondary')
flux_ecl = m1.light_curve(pars)
flux_ecl = (flux_ecl - 1.) / 100e-6

### Filter
filt_kelp = Filter.from_name('IRAC 2')
filt_wavelength, filt_trans = filt_kelp.wavelength.to(u.m).value, filt_kelp.transmittance

### Reflected light phase curve (homogeneous pc)
# Converting times to phases:
phases_unsorted = ((tims- tc) % per) / per         ## Un-sorted phases
idx_phase_sort = np.argsort(phases_unsorted)          ## This would sort any array acc to phase
phases_sorted = phases_unsorted[idx_phase_sort]       ## Sorted phase array
times_sorted_acc_phs = tims[idx_phase_sort]          ## Time array sorted acc to phase
idx_that_sort_arr_acc_times = np.argsort(times_sorted_acc_phs)   ## This array would sort array acc to time

## Phases
xi = 2 * np.pi * (phases_sorted - 0.5)

## Parameters for creating meshgrid
phi_ang = np.linspace(-2 * np.pi, 2 * np.pi, 100)
theta_ang = np.linspace(0, np.pi, 100)
theta2d, phi2d = np.meshgrid(theta_ang, phi_ang)

thermal_pc, TMap = thermal_phase_curve(
    xi=xi, hotspot_offset=np.radians(hotoff), omega_drag=4.5, alpha=0.575, C_11=c11, T_s=3522., a_rs=ar, rp_a=rprs/ar,\
    A_B=0., theta2d=theta2d, phi2d=phi2d, filt_wavelength=filt_wavelength, filt_transmittance=filt_trans, f=fprime
)

thm_pc_sorted_acc_time = thermal_pc[idx_that_sort_arr_acc_times]

total_pc_ecl_model = (thm_pc_sorted_acc_time*flux_ecl) + 1

# Cowan PC model
cowan_pc_model = CowanPC_model(times=tims, te=tc+(per/2), per=per, E=78.81e-6, C1=37.68e-6, D1=-7.89e-6, C2=5.25e-6, D2=0.)
total_pc_ecl_cowan_model = (cowan_pc_model*flux_ecl) + 1

# Total physical model
phy_model = flux_tra * total_pc_ecl_model
phy_model_cowan = flux_tra * total_pc_ecl_cowan_model

# Plotting the thermal PC and full phase curve and temperature map
plt.plot(tims, (total_pc_ecl_model-1.)*1e6, 'k-')
plt.plot(tims, (total_pc_ecl_cowan_model-1.)*1e6, 'r-')
plt.xlabel('Time')
plt.ylabel('Relative flux [ppm]')
plt.show()


plt.plot(tims, (phy_model-1.)*1e6, 'k-')
plt.plot(tims, (phy_model_cowan-1.)*1e6, 'r-')
plt.xlabel('Time')
plt.ylabel('Relative flux [ppm]')
plt.show()

fig = plt.figure()
ax = fig.add_subplot(111, projection='mollweide')
cax = ax.pcolormesh(phi2d, theta2d - np.pi/2, TMap[:,:,0,0])
plt.colorbar(cax, label='T [K]')
plt.show()

# Simulated data
fl_sim, fle_sim = np.zeros(len(tims)), np.zeros(len(tims))
mflx = 0.
for i in range(len(tims)):
    fl_sim[i] = np.random.normal(phy_model[i], 100e-6) / (1 + mflx)
    fle_sim[i] = np.abs(np.random.normal(40e-6, 1e-6))

plt.errorbar(tims, fl_sim, yerr=fle_sim, fmt='.')
plt.xlabel('Time')
plt.ylabel('Relative flux')
plt.show()

fname = open('Analysis/sim_data.dat', 'w')
for i in range(len(tims)):
    fname.write(str(tims[i]) + '\t' + str(fl_sim[i]) + '\t' + str(fle_sim[i]) + '\n')
fname.close()