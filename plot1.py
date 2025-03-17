import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gd
import astropy.constants as con
from kelp import Filter
import juliet
from utils import lcbin, evaluate_CowanPC, lttd
import pickle
import os

# This file is to plot phase curve models (will plot 68% region and not random models)
instrument = 'TESS14'
pout = os.getcwd() + '/Analysis/Test2_kelt9'

G = con.G.value

## Stellar radius (in units of the Solar radius, to correct for light travel time delay)
rst = 1.458

## Loading the parameters
post = pickle.load(open(pout + '/_dynesty_DNS_posteriors.pkl', 'rb'))
samples = post['posterior_samples']

try:
    C2 = samples['C2_p1_' + instrument]
except:
    C2 = np.zeros(100)

try:
    D2 = samples['D2_p1_' + instrument]
except:
    D2 = np.zeros(100)

try:
    GP_S0, GP_Q, GP_rho = samples['GP_S0'], samples['GP_Q'], samples['GP_rho']
    GP = True
except:
    GP_S0, GP_Q, GP_rho = np.zeros(100), np.zeros(100), np.zeros(100)
    GP = False

## Dummy times (we will need these dummy data only for non-juliet models: 
# for juliet models, we will replace the dummy data with the real data later)
dummy_tim = np.linspace( np.nanmedian(samples['t0_p1'] - 0.499*samples['P_p1']), np.nanmedian(samples['t0_p1'] + 0.499*samples['P_p1']), 10000)
dummy_fl, dummy_fle = np.ones(len(dummy_tim)), 0.1 * np.ones(len(dummy_tim))

dummy_phs_tra = juliet.utils.get_phases(t=dummy_tim, P=np.nanmedian(samples['P_p1']), t0=np.nanmedian(samples['t0_p1']), phmin=0.5)
dummy_phs_pc = juliet.utils.get_phases(t=dummy_tim, P=np.nanmedian(samples['P_p1']), t0=np.nanmedian(samples['t0_p1']), phmin=1.)


# We need to store the data first:
## We need to save: 1) detrended data (flux, and residuals): for every instruments
##                  2) quantile models: median and 68% lower and upper quantiles, for every instruments
##                  3) binned data (tim, flux, errors, residuals, resid_errs): for transit and pc, combined for all instruments
##                  4) phases for every instruments: for transits and pc

detrended_fl, residuals = {}, {}
phases_tra, phases_pc = {}, {}


## Load the data
data = pickle.load(open(pout + '/_lc_.data', 'rb'))
par_tot = [i for i in samples.keys()]

# Loading the data first
tim7, fl7, fle7 = data['times'][instrument], data['fluxes'][instrument], data['errors'][instrument] 

## Computing ar from rho
ar = ((samples['rho'] * G * ((samples['P_p1'] * 24. * 3600.)**2)) / (3. * np.pi))**(1. / 3.)

## Let's correct for light travel time
tim7 = lttd(times=tim7, t0=np.nanmedian(samples['t0_p1']), per=np.nanmedian(samples['P_p1']), ar=np.nanmedian(ar), bb=np.nanmedian(samples['b_p1']), rst=rst)

## Now, computing phases
phases_tra[instrument] = juliet.utils.get_phases(t=tim7, P=np.nanmedian(samples['P_p1']), t0=np.nanmedian(samples['t0_p1']), phmin=0.5)
phases_pc[instrument] = juliet.utils.get_phases(t=tim7, P=np.nanmedian(samples['P_p1']), t0=np.nanmedian(samples['t0_p1']), phmin=1.)

## mflux
mflx = np.nanmedian(samples['mflux_' + instrument])

## And the GP and linear models (to detrend the data)
_, flux_norm, gp_model, total_model, _ = \
    evaluate_CowanPC(times=tim7, fluxes=fl7, errors=fle7, t0=np.nanmedian(samples['t0_p1']),\
                     per=np.nanmedian(samples['P_p1']), rprs=np.nanmedian(samples['p_p1']), bb=np.nanmedian(samples['b_p1']),\
                     ar=np.nanmedian(ar), ecc=0., omega=90., q1=np.nanmedian(samples['q1_' + instrument]),\
                     q2=np.nanmedian(samples['q2_' + instrument]), E=np.nanmedian(samples['E_p1_' + instrument]),\
                     C1=np.nanmedian(samples['C1_p1_' + instrument]), D1=np.nanmedian(samples['D1_p1_' + instrument]),\
                     C2=np.nanmedian(C2), D2=np.nanmedian(D2), mflx=np.nanmedian(samples['mflux_' + instrument]),\
                     sigw=np.nanmedian(samples['sigma_w_' + instrument]), GP=GP, GP_S0=np.nanmedian(GP_S0),\
                     GP_Q=np.nanmedian(GP_Q), GP_rho=np.nanmedian(GP_rho), LTTD=True, rst=rst)

detrended_fl[instrument] = (fl7 - gp_model) * (1 + mflx)
residuals[instrument] = (fl7 - total_model) * 1e6


# Another loop to collect binned data points because if there are more than one instruments,
# binned data points will be common for all instruments
all_phs_tra, all_fl, all_resid = [], [], []
all_phs_pc = []

## Saving the data
### Transit data
all_phs_tra = all_phs_tra + [ phases_tra[instrument] ]
### Phase curve data
all_phs_pc = all_phs_pc + [ phases_pc[instrument] ]

### Flux and residuals
all_fl = all_fl + [ detrended_fl[instrument] ]
all_resid = all_resid + [ residuals[instrument] ]

## Binning for transits
bin_phs_tra, bin_fl_tra, bin_fle_tra, _ = lcbin(time=np.hstack(all_phs_tra), flux=np.hstack(all_fl), binwidth=0.01)
_, bin_resid_tra, bin_reserr_tra, _ = lcbin(time=np.hstack(all_phs_tra), flux=np.hstack(all_resid), binwidth=0.01)
## Binned for phase curves
bin_phs_pc, bin_fl_pc, bin_fle_pc, _ = lcbin(time=np.hstack(all_phs_pc), flux=np.hstack(all_fl), binwidth=0.03)
_, bin_resid_pc, bin_reserr_pc, _ = lcbin(time=np.hstack(all_phs_pc), flux=np.hstack(all_resid), binwidth=0.03)

# Running separate loop for quantile models 
# (because if there are more than one instruments, then it doesn't make sense to run this
#  for every instruments -- it takes time to compute these models)
random_models = np.zeros((50, len(dummy_tim)))
## Computing ar from rho
ar = ((samples['rho'] * G * ((samples['P_p1'] * 24. * 3600.)**2)) / (3. * np.pi))**(1. / 3.)

## First computing median models
median_model, _, _, _, _ = \
    evaluate_CowanPC(times=dummy_tim, fluxes=dummy_fl, errors=dummy_fle, t0=np.nanmedian(samples['t0_p1']),\
                     per=np.nanmedian(samples['P_p1']), rprs=np.nanmedian(samples['p_p1']), bb=np.nanmedian(samples['b_p1']),\
                     ar=np.nanmedian(ar), ecc=0., omega=90., q1=np.nanmedian(samples['q1_' + instrument]),\
                     q2=np.nanmedian(samples['q2_' + instrument]), E=np.nanmedian(samples['E_p1_' + instrument]),\
                     C1=np.nanmedian(samples['C1_p1_' + instrument]), D1=np.nanmedian(samples['D1_p1_' + instrument]),\
                     C2=np.nanmedian(C2), D2=np.nanmedian(D2), mflx=np.nanmedian(samples['mflux_' + instrument]),\
                     sigw=np.nanmedian(samples['sigma_w_' + instrument]), LTTD=True, rst=rst)

## Random models
for i in range(50):
    random_models[i,:], _, _, _, _ = \
            evaluate_CowanPC(times=dummy_tim, fluxes=dummy_fl, errors=dummy_fle, t0=np.random.choice(samples['t0_p1']),\
                             per=np.random.choice(samples['P_p1']), rprs=np.random.choice(samples['p_p1']), bb=np.random.choice(samples['b_p1']),\
                             ar=np.nanmedian(ar), ecc=0., omega=90., q1=np.random.choice(samples['q1_' + instrument]),\
                             q2=np.random.choice(samples['q2_' + instrument]), E=np.random.choice(samples['E_p1_' + instrument]),\
                             C1=np.random.choice(samples['C1_p1_' + instrument]), D1=np.random.choice(samples['D1_p1_' + instrument]),\
                             C2=np.random.choice(C2), D2=np.random.choice(D2), mflx=np.random.choice(samples['mflux_' + instrument]),\
                             sigw=np.random.choice(samples['sigma_w_' + instrument]), LTTD=True, rst=rst)

# Two sorting array that will sort dummy phases
idx_phs_tra, idx_phs_pc = np.argsort(dummy_phs_tra), np.argsort(dummy_phs_pc)

# Figure codes starts from here

fig = plt.figure(figsize=(16/1.5,9/1.5))
gs = gd.GridSpec(2, 2, height_ratios=[2,1], width_ratios=[1,3])#, wspace=0.1)


# Transit
xlim1, xlim2 = -0.07, 0.07
## -- Upper panel
ax1 = plt.subplot(gs[0,0])
ax1.errorbar(phases_tra[instrument], (detrended_fl[instrument]-1.)*1e6, fmt='.', alpha=0.25, c='cornflowerblue', zorder=1)
ax1.errorbar(bin_phs_tra, (bin_fl_tra-1.)*1e6, yerr=bin_fle_tra*1e6, fmt='o', c='navy', elinewidth=2, capthick=2, capsize=3, mfc='white', zorder=100)
ax1.plot(dummy_phs_tra[idx_phs_tra], (median_model[idx_phs_tra]-1.)*1e6, color='navy', lw=2., zorder=50)
### Errorbars (on models)
for i in range(50):
    ax1.plot(dummy_phs_tra[idx_phs_tra], (random_models[i,:][idx_phs_tra]-1.)*1e6, color='orangered', alpha=0.5, lw=0.7, zorder=10)

ax1.set_ylabel('Relative flux [ppm]', fontsize=14, fontfamily='serif')
ax1.set_xlim([xlim1, xlim2])
#ax1.set_ylim([-300, 50])
ax1.set_ylim([-8000, 1000])

ax1.tick_params(labelfontfamily='serif')
ax1.set_xticks(ticks=np.array([-0.04, 0.0, 0.04]))
plt.setp(ax1.get_yticklabels(), fontsize=12)
ax1.xaxis.set_major_formatter(plt.NullFormatter())

## -- Bottom panel
ax2 = plt.subplot(gs[1,0])
ax2.errorbar(phases_tra[instrument], residuals[instrument], fmt='.', alpha=0.1, c='cornflowerblue')
ax2.errorbar(bin_phs_tra, bin_resid_tra, yerr=bin_reserr_tra, fmt='o', c='navy', elinewidth=2, capthick=2, capsize=3, mfc='white', zorder=20)
ax2.axhline(0.0, lw=2., color='navy', zorder=10)

ax2.set_xlabel('Orbital Phase', fontsize=14, fontfamily='serif')
ax2.set_ylabel('Residuals [ppm]', fontsize=14, fontfamily='serif')
ax2.set_xlim([xlim1, xlim2])
#ax2.set_ylim([-50., 50.])
ax2.set_ylim([-200., 200.])

ax2.set_xticks(ticks=np.array([-0.04, 0.0, 0.04]), labels=np.array([-0.04, 0.0, 0.04]))
ax2.tick_params(labelfontfamily='serif')
plt.setp(ax2.get_xticklabels(), fontsize=12)
plt.setp(ax2.get_yticklabels(), fontsize=12)


# Phase curve
xlim3, xlim4 = 0.01, 0.99
## Upper panel
ax3 = plt.subplot(gs[0,1])
ax3.errorbar(phases_pc[instrument], (detrended_fl[instrument]-1.)*1e6, alpha=0.1, fmt='.', c='cornflowerblue', zorder=1)
ax3.errorbar(bin_phs_pc, (bin_fl_pc-1.)*1e6, yerr=bin_fle_pc*1e6, fmt='o', c='navy', elinewidth=2, capthick=2, capsize=3, mfc='white', zorder=100)
ax3.plot(dummy_phs_pc[idx_phs_pc], (median_model[idx_phs_pc]-1.)*1e6, c='navy', lw=2., zorder=50)
### Errorbars (on models)
for i in range(50):
    ax3.plot(dummy_phs_pc[idx_phs_pc], (random_models[i,:][idx_phs_pc]-1.)*1e6, color='orangered', alpha=0.5, lw=0.7, zorder=10)

ax3.set_ylabel('Relative flux [ppm]', rotation=270, fontsize=14, labelpad=25, fontfamily='serif')
ax3.set_xlim([xlim3, xlim4])
#ax3.set_ylim([-50., 100.])
ax3.set_ylim([-100., 750.])

ax3.yaxis.tick_right()
ax3.tick_params(labelright=True, labelfontfamily='serif')
plt.setp(ax3.get_yticklabels(), fontsize=12)
ax3.yaxis.set_label_position('right')
ax3.xaxis.set_major_formatter(plt.NullFormatter())

## Bottom panel
ax4 = plt.subplot(gs[1,1])
ax4.errorbar(phases_pc[instrument], residuals[instrument], alpha=0.1, fmt='.', color='cornflowerblue')
ax4.errorbar(bin_phs_pc, bin_resid_pc, yerr=bin_reserr_pc, fmt='o', c='navy', elinewidth=2, capthick=2, capsize=3, mfc='white', zorder=100)
ax4.axhline(0., lw=2., color='navy')

ax4.set_xlabel('Orbital Phase', fontsize=14, fontfamily='serif')
ax4.set_ylabel('Residuals [ppm]', rotation=270, fontsize=14, labelpad=25, fontfamily='serif')
ax4.set_xlim([xlim3, xlim4])
#ax4.set_ylim([-30., 30.])
ax4.set_ylim([-70., 70.])

ax4.yaxis.tick_right()
ax4.tick_params(labelright=True, labelfontfamily='serif')
plt.setp(ax4.get_xticklabels(), fontsize=12)
plt.setp(ax4.get_yticklabels(), fontsize=12)
ax4.yaxis.set_label_position('right')

plt.tight_layout()
plt.show()
#plt.savefig(pout + '/phase_folded_random_lc' + '_'.join(instrument) + '.png', dpi=500)