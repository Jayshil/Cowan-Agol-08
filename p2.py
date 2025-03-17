import numpy as np
import matplotlib.pyplot as plt
from exotoolbox.utils import get_quantiles
import matplotlib.gridspec as gd
import astropy.constants as con
import plotstyles
import dynesty
from dynesty.utils import resample_equal
from utils import evaluate_CowanPC, computeRMS
from utils import corner_plot
from scipy import stats
import os
import pickle
from pathlib import Path
from multiprocessing import Pool
import contextlib

import multiprocessing
multiprocessing.set_start_method('fork')

# This file is to fit sinusoidal phase curve (juliet) to CHEOPS data
# Roll angle modulation is fit using sinusoidal models (up to third order sin)
# No GP in time, fitted for phase offset

G = con.G.value

# Output folder
pout = os.getcwd() + '/Analysis/Test1'
if not Path(pout).exists():
    os.mkdir(pout)

# Number of threads
nthreads=8

# ------------------------------------------------------------------
#
#                        Planetary parameters
#
# ------------------------------------------------------------------

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

# ------------------------------------------------------------------
#
#                        Loading the data
#
# ------------------------------------------------------------------

instrument = 'IRAC2'

# Loading the data in a way juliet understands
tim, fl, fle = {}, {}, {}
tim7, fl7, fle7 = np.loadtxt(os.getcwd() + '/Analysis/sim_data.dat', usecols=(0,1,2), unpack=True)
tim[instrument], fl[instrument], fle[instrument] = tim7, fl7, fle7

## Predicting transit time for the data
cycle = round((tim[instrument][-1] - tc)/per)
tc1, tc1_err = tc + (cycle*per), np.sqrt(tc_err**2 + (cycle*per_err)**2)

# ------------------------------------------------------------------
#
#                        Defining the priors
#
# ------------------------------------------------------------------

## Planetary priors
par_P = ['P_p1', 't0_p1', 'p_p1', 'b_p1', 'q1_' + instrument, 'q2_' + instrument, 'ecc_p1', 'omega_p1', 'rho']
dist_P = ['normal', 'normal', 'normal', 'normal', 'uniform', 'uniform', 'fixed', 'fixed', 'normal']
hyper_P = [[per, per_err], [tc1, tc1_err], [rprs, rprs_err], [bb, bb_err], [0., 1.], [0., 1.], 0., 90., [rho, rho_err]]

par_pc = ['E_p1_' + instrument, 'C1_p1_' + instrument, 'D1_p1_' + instrument, 'C2_p1_' + instrument, 'D2_p1_' + instrument]
dist_pc = ['uniform', 'uniform', 'uniform', 'fixed', 'fixed']
hyper_pc = [[0.e-6, 500.e-6], [-1., 1.], [-1., 1.], 0., 0.]

## Instrumental priors
par_ins = ['mflux_' + instrument, 'sigma_w_' + instrument]
dist_ins = ['normal', 'loguniform']
hyper_ins = [[0., 0.1], [0.1, 10000.]]

## Total priors
par_tot = par_P + par_pc + par_ins
dist_tot = dist_P + dist_pc + dist_ins
hyper_tot = hyper_P + hyper_pc + hyper_ins

## Prior transform
def uniform(t, a, b):
    return (b-a)*t + a
def stand(a, loc, scale):
    return (a-loc)/scale

# Saving prior file
f11 = open(pout + '/priors.dat', 'w')
for i in range(len(par_tot)):
    f11.write(par_tot[i] + '\t' + dist_tot[i] + '\t' + str(hyper_tot[i]) + '\n')
f11.close()

# Also saving the data
data_all = {}
data_all['times'], data_all['fluxes'], data_all['errors'] = tim, fl, fle
pickle.dump(data_all, open(pout + '/_lc_.data','wb'))

# First leave the fixed parameters
free_params, free_dists, free_hypers = [], [], []
fixed_params, fixed_vals = [], []
for i in range(len(par_tot)):
    if dist_tot[i] != 'fixed':
        free_params.append(par_tot[i])
        free_dists.append(dist_tot[i])
        free_hypers.append(hyper_tot[i])
    else:
        fixed_params.append(par_tot[i])
        fixed_vals.append(hyper_tot[i])

# Prior cube for only free parameters
def prior_transform(ux):
    x = np.array(ux)
    for i in range(len(free_params)):
        if free_dists[i] == 'uniform':
            x[i] = uniform(ux[i], free_hypers[i][0], free_hypers[i][1])
        elif free_dists[i] == 'normal':
            x[i] = stats.norm.ppf(ux[i], loc=free_hypers[i][0], scale=free_hypers[i][1])
        elif free_dists[i] == 'truncatednormal':
            x[i] = stats.truncnorm.ppf(ux[i], a=stand(free_hypers[i][2], free_hypers[i][0], free_hypers[i][1]), b=stand(free_hypers[i][3], free_hypers[i][0], free_hypers[i][1]), loc=free_hypers[i][0], scale=free_hypers[i][1])
        elif free_dists[i] == 'loguniform':
            x[i] = stats.loguniform.ppf(ux[i], a=free_hypers[i][0], b=free_hypers[i][1])
        else:
            raise Exception('Please use proper distribution!')
    return x


# ------------------------------------------------------------------
#
#               Defining the log likelihood function
#
# ------------------------------------------------------------------


def loglike(x):
    global tim, fl, fle, par_tot, free_params, fixed_params, fixed_vals

    # Saving values of parameters in a dictionary

    parameters = {}
    for p in range(len(free_params)):
        parameters[free_params[p]] = x[p]
    for p in range(len(fixed_params)):
        parameters[fixed_params[p]] = fixed_vals[p]

    # Computing ar from rho
    ar = ((parameters['rho'] * G * ((parameters['P_p1'] * 24. * 3600.)**2)) / (3. * np.pi))**(1. / 3.)

    # And computing log-likelihood
    _, _, _, log_like = \
        evaluate_CowanPC(times=tim[instrument], fluxes=fl[instrument], errors=fle[instrument], t0=parameters['t0_p1'],\
                         per=parameters['P_p1'], rprs=parameters['p_p1'], bb=parameters['b_p1'], ar=ar, ecc=parameters['ecc_p1'],\
                         omega=parameters['omega_p1'], q1=parameters['q1_' + instrument], q2=parameters['q2_' + instrument],\
                         E=parameters['E_p1_' + instrument], C1=parameters['C1_p1_' + instrument], D1=parameters['D1_p1_' + instrument],\
                         C2=parameters['C2_p1_' + instrument], D2=parameters['D2_p1_' + instrument],\
                         mflx=parameters['mflux_' + instrument], sigw=parameters['sigma_w_' + instrument],\
                         LTTD=True, rst=rst)
    
    return log_like


# ------------------------------------------------------------------
#
#                      Sampling with dynesty
#
# ------------------------------------------------------------------

out_files = Path(pout + '/_dynesty_DNS_posteriors.pkl')
## Only start sampler if dynesty output files are not detected
## Otherwise, just load them
if not out_files.exists():
    with contextlib.closing(Pool(processes=nthreads-1)) as executor:
        dsampler = dynesty.DynamicNestedSampler(loglikelihood=loglike, prior_transform=prior_transform,\
            ndim=len(free_params), nlive=500, bound='single', sample='rwalk', pool=executor, queue_size=nthreads)
        dsampler.run_nested()
    dres = dsampler.results

    weights = np.exp(dres['logwt'] - dres['logz'][-1])
    posterior_samples = resample_equal(dres.samples, weights)

    f22 = open(pout + '/posteriors.dat', 'w')
    post_samps = {}
    post_samps['posterior_samples'] = {}
    for i in range(len(free_params)):
        post_samps['posterior_samples'][free_params[i]] = posterior_samples[:, i]
        qua = get_quantiles(posterior_samples[:, i])
        f22.write(free_params[i] + '\t' + str(qua[0]) + '\t' + str(qua[1]-qua[0]) + '\t' + str(qua[0]-qua[2]) + '\n')
    f22.close()

    # logZ
    post_samps['lnZ'] = dres.logz
    post_samps['lnZ_err'] = dres.logzerr

    # Dumping a pickle
    pickle.dump(post_samps,open(pout + '/_dynesty_DNS_posteriors.pkl','wb'))
else:
    print('>>>> --- Dynesty sampler files are detected!!!')
    print('>>>> --- Loading them...')
    post_samps = pickle.load(open(pout + '/_dynesty_DNS_posteriors.pkl', 'rb'))
    print('>>>> --- Done!')

# ------------------------------------------------------------------
#
#                          Some plots
#
# ------------------------------------------------------------------

# Model
samples = post_samps['posterior_samples']
## Adding fixed parameters to samples
for p in range(len(fixed_params)):
    samples[fixed_params[p]] = fixed_vals[p]

# And plotting some cool results

ar = ((np.nanmedian(samples['rho']) * G * ((np.nanmedian(samples['P_p1']) * 24. * 3600.)**2)) / (3. * np.pi))**(1. / 3.)

_, _, total_model, _ = \
    evaluate_CowanPC(times=tim[instrument], fluxes=fl[instrument], errors=fle[instrument], t0=np.nanmedian(samples['t0_p1']),\
                     per=np.nanmedian(samples['P_p1']), rprs=np.nanmedian(samples['p_p1']), bb=np.nanmedian(samples['b_p1']), ar=ar,\
                     ecc=np.nanmedian(samples['ecc_p1']), omega=np.nanmedian(samples['omega_p1']), q1=np.nanmedian(samples['q1_' + instrument]),\
                     q2=np.nanmedian(samples['q2_' + instrument]), E=np.nanmedian(samples['E_p1_' + instrument]),\
                     C1=np.nanmedian(samples['C1_p1_' + instrument]), D1=np.nanmedian(samples['D1_p1_' + instrument]),\
                     C2=np.nanmedian(samples['C2_p1_' + instrument]), D2=np.nanmedian(samples['D2_p1_' + instrument]),\
                     mflx=np.nanmedian(samples['mflux_' + instrument]), sigw=np.nanmedian(samples['sigma_w_' + instrument]),\
                     LTTD=True, rst=rst)


# And plotting
fig = plt.figure(figsize=(16/1.5,9/1.5))
gs = gd.GridSpec(2,1, height_ratios=[2,1])

# Sort according to time
idx_tim = np.argsort(tim[instrument])

# Top panel
ax1 = plt.subplot(gs[0])
ax1.errorbar(tim[instrument][idx_tim], fl[instrument][idx_tim], yerr=fle[instrument][idx_tim], fmt='.')#, alpha=0.3)
ax1.plot(tim[instrument][idx_tim], total_model[idx_tim], c='k', zorder=100)
ax1.set_ylabel('Relative Flux')
ax1.set_xlim(np.min(tim[instrument]), np.max(tim[instrument]))
ax1.xaxis.set_major_formatter(plt.NullFormatter())

# Bottom panel
ax2 = plt.subplot(gs[1])
ax2.errorbar(tim[instrument][idx_tim], (fl[instrument][idx_tim]-total_model[idx_tim])*1e6, yerr=fle[instrument][idx_tim]*1e6, fmt='.')#, alpha=0.3)
ax2.axhline(y=0.0, c='black', ls='--', zorder=100)
ax2.set_ylabel('Residuals (ppm)')
ax2.set_xlabel('Time (BJD)')
ax2.set_xlim(np.min(tim[instrument]), np.max(tim[instrument]))
#plt.show()
plt.savefig(pout + '/full_model_' + instrument + '.png')

# Alan deviation plot

residuals = fl[instrument][idx_tim] - total_model[idx_tim]
rms, stderr, binsz = computeRMS(residuals, binstep=1)
normfactor = 1e-6

fig = plt.figure(figsize=(8,6))
plt.plot(binsz, rms / normfactor, color='black', lw=1.5,
                label='Fit RMS', zorder=3)
plt.plot(binsz, stderr / normfactor, color='red', ls='-', lw=2,
                label=r'Std. Err. ($1/\sqrt{N}$)', zorder=1)
plt.xlim(0.95, binsz[-1] * 2)
plt.ylim(stderr[-1] / normfactor / 2., stderr[0] / normfactor * 2.)
plt.xlabel("Bin Size (N frames)", fontsize=14)
plt.ylabel("RMS (ppm)", fontsize=14)
plt.xscale('log')
plt.yscale('log')
plt.legend(loc='best')
#plt.show()
plt.savefig(pout + '/alan_deviation_' + instrument + '.png')

#fig, axes = dyplot.cornerplot(dres, show_titles=True, labels=free_params)
#plt.savefig(pout + '/corner.pdf')

corner_plot(pout, True)