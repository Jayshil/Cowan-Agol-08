import numpy as np
import matplotlib.pyplot as plt
import plotstyles
from kelp.jax import thermal_phase_curve
from utils import evaluate_CowanPC
import astropy.units as u
import pickle
import os
from kelp import Filter
import batman

# Compare the fitted models with the ingested model

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

# Total physical model
phy_model = flux_tra * total_pc_ecl_model

# ----------------------------------------------
#            For fitted models
# ----------------------------------------------
instrument = 'IRAC2'
pout = os.getcwd() + '/Analysis/Test1'
post = pickle.load(open(pout + '/_dynesty_DNS_posteriors.pkl', 'rb'))
samples = post['posterior_samples']
dummy_fl, dummy_fle = np.zeros(len(tims)), np.zeros(len(tims))

plt.plot(tims, (phy_model-1.)*1e6, 'k-')
## Random models
for i in range(50):
    random_model, _, _, _ = \
            evaluate_CowanPC(times=tims, fluxes=dummy_fl, errors=dummy_fle, t0=np.random.choice(samples['t0_p1']),\
                             per=np.random.choice(samples['P_p1']), rprs=np.random.choice(samples['p_p1']), bb=np.random.choice(samples['b_p1']),\
                             ar=np.nanmedian(ar), ecc=0., omega=90., q1=np.random.choice(samples['q1_' + instrument]),\
                             q2=np.random.choice(samples['q2_' + instrument]), E=np.random.choice(samples['E_p1_' + instrument]),\
                             C1=np.random.choice(samples['C1_p1_' + instrument]), D1=np.random.choice(samples['D1_p1_' + instrument]),\
                             C2=0., D2=0., mflx=np.random.choice(samples['mflux_' + instrument]),\
                             sigw=np.random.choice(samples['sigma_w_' + instrument]), LTTD=True, rst=rst)
    plt.plot(tims, (random_model-1.)*1e6, 'r-', alpha=0.1)
plt.xlabel('Time')
plt.ylabel('Relative flux [ppm]')
plt.show()