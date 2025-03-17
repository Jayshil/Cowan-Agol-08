import numpy as np
import matplotlib.pyplot as plt
from celerite2.terms import SHOTerm
from celerite2 import GaussianProcess
import corner
import pickle
from glob import glob
import batman
import juliet
from astropy.stats import mad_std
import astropy.constants as con
import astropy.units as u

# To standardise the data
def standard(x):
    return (x - np.nanmedian(x)) / mad_std(x)

# Gaussian log-likelihood
def gaussian_log_likelihood(residuals, variances):
    taus = 1. / variances
    return -0.5 * (len(residuals) * np.log(2*np.pi) + np.sum(-np.log(taus.astype(float)) + taus * (residuals**2)))

# Computing light travel time delay:
def lttd(times, t0, per, ar, bb, rst):
    inc1 = np.arccos(bb / ar)
    term1 = 1 - np.cos(2 * np.pi * (times - t0) / per)
    abyc = ( ( ar * rst * u.R_sun / con.c ).to(u.d) ).value
    times = times - ( abyc * term1 * np.sin(inc1) )
    return times

def lcbin(time, flux, binwidth=0.06859, nmin=4, time0=None,
        robust=False, tmid=False):
    """
    This code is taken from the code `pycheops`
    Calculate average flux and error in time bins of equal width.
    The default bin width is equivalent to one CHEOPS orbit in units of days.
    To avoid binning data on either side of the gaps in the light curve due to
    the CHEOPS orbit, the algorithm searches for the largest gap in the data
    shorter than binwidth and places the bin edges so that they fall at the
    centre of this gap. This behaviour can be avoided by setting a value for
    the parameter time0.
    The time values for the output bins can be either the average time value
    of the input points or, if tmid is True, the centre of the time bin.
    If robust is True, the output bin values are the median of the flux values
    of the bin and the standard error is estimated from their mean absolute
    deviation. Otherwise, the mean and standard deviation are used.
    The output values are as follows.
    * t_bin - average time of binned data points or centre of time bin.
    * f_bin - mean or median of the input flux values.
    * e_bin - standard error of flux points in the bin.
    * n_bin - number of flux points in the bin.
    :param time: time
    :param flux: flux (or other quantity to be time-binned)
    :param binwidth:  bin width in the same units as time
    :param nmin: minimum number of points for output bins
    :param time0: time value at the lower edge of one bin
    :param robust: use median and robust estimate of standard deviation
    :param tmid: return centre of time bins instead of mean time value
    :returns: t_bin, f_bin, e_bin, n_bin
    """
    if time0 is None:
        tgap = (time[1:]+time[:-1])/2
        gap = time[1:]-time[:-1]
        j = gap < binwidth
        gap = gap[j]
        tgap = tgap[j]
        time0 = tgap[np.argmax(gap)]
        time0 = time0 - binwidth*np.ceil((time0-min(time))/binwidth)

    n = int(1+np.ceil(np.ptp(time)/binwidth))
    r = (time0,time0+n*binwidth)
    n_in_bin,bin_edges = np.histogram(time,bins=n,range=r)
    bin_indices = np.digitize(time,bin_edges)

    t_bin = np.zeros(n)
    f_bin = np.zeros(n)
    e_bin = np.zeros(n)
    n_bin = np.zeros(n, dtype=int)

    for i,n in enumerate(n_in_bin):
        if n >= nmin:
            j = bin_indices == i+1
            n_bin[i] = n
            if tmid:
                t_bin[i] = (bin_edges[i]+bin_edges[i+1])/2
            else:
                t_bin[i] = np.nanmean(time[j])
            if robust:
                f_bin[i] = np.nanmedian(flux[j])
                e_bin[i] = 1.25*np.nanmean(abs(flux[j] - f_bin[i]))/np.sqrt(n)
            else:
                f_bin[i] = np.nanmean(flux[j])
                e_bin[i] = np.std(flux[j])/np.sqrt(n-1)

    j = (n_bin >= nmin)
    return t_bin[j], f_bin[j], e_bin[j], n_bin[j]

def computeRMS(data, maxnbins=None, binstep=1, isrmserr=False):
    """Compute the root-mean-squared and standard error for various bin sizes.
    Parameters: This function is taken from the code `Eureka` -- please cite them!!
    ----------
    data : ndarray
        The residuals after fitting.
    maxnbins : int; optional
        The maximum number of bins. Use None to default to 10 points per bin.
    binstep : int; optional
        Bin step size. Defaults to 1.
    isrmserr : bool
        True if return rmserr, else False. Defaults to False.
    Returns
    -------
    rms : ndarray
        The RMS for each bin size.
    stderr : ndarray
        The standard error for each bin size.
    binsz : ndarray
        The different bin sizes.
    rmserr : ndarray; optional
        The uncertainty in the RMS. Only returned if isrmserr==True.
    Notes
    -----
    History:
    - December 29-30, 2021 Taylor Bell
        Moved code to separate file, added documentation.
    """
    data = np.ma.masked_invalid(np.ma.copy(data))
    
    # bin data into multiple bin sizes
    npts = data.size
    if maxnbins is None:
        maxnbins = npts / 10.
    binsz = np.arange(1, maxnbins + binstep, step=binstep, dtype=int)
    nbins = np.zeros(binsz.size, dtype=int)
    rms = np.zeros(binsz.size)
    rmserr = np.zeros(binsz.size)
    for i in range(binsz.size):
        nbins[i] = int(np.floor(data.size / binsz[i]))
        bindata = np.ma.zeros(nbins[i], dtype=float)
        # bin data
        # ADDED INTEGER CONVERSION, mh 01/21/12
        for j in range(nbins[i]):
            bindata[j] = np.ma.mean(data[j * binsz[i]:(j + 1) * binsz[i]])
        # get rms
        rms[i] = np.sqrt(np.ma.mean(bindata ** 2))
        rmserr[i] = rms[i] / np.sqrt(2. * nbins[i])
    # expected for white noise (WINN 2008, PONT 2006)
    stderr = (np.ma.std(data) / np.sqrt(binsz)) * np.sqrt(nbins / (nbins - 1.))
    if isrmserr is True:
        return rms, stderr, binsz, rmserr
    else:
        return rms, stderr, binsz
    

def CowanPC_model(times, te, per, E, C1, D1, C2, D2):
    omega_t = 2 * np.pi * (times - te) / per
    pc = E + ( C1 * (np.cos( omega_t ) - 1.) ) + ( D1 * np.sin( omega_t ) ) +\
             ( C2 * (np.cos( 2*omega_t ) - 1.) ) + ( D2 * np.sin( 2*omega_t ) )
    return pc


def evaluate_CowanPC(times, fluxes, errors, t0, per, rprs, bb, ar, ecc, omega, q1, q2, E, C1, D1, C2, D2,\
                     mflx, sigw, GP=False, GP_S0=None, GP_Q=None, GP_rho=None, LTTD=True, rst=None):
    
    times = times.astype('float64')

    if LTTD:
        ## This mean that we need to correct for the light travel time delay
        ### Computing inclination from bb and ar
        inc1 = np.arccos(bb / ar)
        term1 = 1 - np.cos(2 * np.pi * (times - t0) / per)
        abyc = ( ( ar * rst * u.R_sun / con.c ).to(u.d) ).value
        times = times - ( abyc * term1 * np.sin(inc1) )

    # Converting q1 and q2 to u1 and u2
    u1, u2 = juliet.utils.reverse_ld_coeffs('quadratic', q1, q2)
    
    # First batman transit model
    params = batman.TransitParams()
    params.t0 = t0
    params.per = per
    params.rp = rprs
    params.a = ar
    params.inc = np.rad2deg(np.arccos(bb / ar))
    params.ecc = ecc
    params.w = omega
    params.u = [u1, u2]
    params.limb_dark = 'quadratic'

    m = batman.TransitModel(params, times)
    flux_tra = m.light_curve(params)

    # Occultation model
    params.fp = 100e-6
    params.t_secondary = t0 + (per/2)

    m1 = batman.TransitModel(params, t=times, transittype='secondary')
    flux_ecl = m1.light_curve(params)
    flux_ecl = (flux_ecl - 1.) / 100e-6

    # Phase curve models
    ## Phase curve (juliet-like; copied from juliet)
    sine_model = CowanPC_model(times=times, te=t0+(per/2), per=per, E=E, C1=C1, D1=D1, C2=C2, D2=D2)

    ### phase curve + eclipse model
    sine_ecl_model = 1. + (sine_model * flux_ecl)

    # Total physical model
    phy_model = sine_ecl_model*flux_tra
    
    # Normalising the flux
    flux_norm = phy_model / ( 1 + mflx )

    # Computing residuals
    resids = fluxes - flux_norm

    # Now, if GP=True, compute the GP model (and also, loglikelihood)
    vars = errors**2 + (sigw*1e-6)**2
    if GP:
        ker = SHOTerm(S0=GP_S0, Q=GP_Q, rho=GP_rho)
        gp = GaussianProcess(ker, mean=0.)
        gp.compute(times, diag=vars, quiet=True)
        
        loglike = gp.log_likelihood(resids)
        gp_model = gp.predict(y=resids, t=times, return_cov=False)
    else:
        gp_model = np.zeros(len(times))
        loglike = gaussian_log_likelihood(resids, vars)

    total_model = flux_norm + gp_model

    return phy_model, flux_norm, gp_model, total_model, loglike

def corner_plot(folder, planet_only=False):
    pcl = glob(folder + '/*.pkl')[0]
    post = pickle.load(open(pcl, 'rb'), encoding='latin1')
    p1 = post['posterior_samples']
    lst = []
    if not planet_only:
        for i in p1.keys():
            lst.append(i)
    else:
        for i in p1.keys():
            gg = i.split('_')
            if 'p1' in gg or 'q1' in gg or 'q2' in gg or 'rho' in gg or 'mst' in gg or 'rst' in gg or 'vsini' in gg or 'tpole' in gg or 'phi' in gg or 'lamp' in gg:
                if not 'GP' in gg:
                    lst.append(i)
    if 't0' in lst[0].split('_'):
        t01 = np.floor(p1[lst[0]][0])
        cd = p1[lst[0]] - t01
        lst[0] = lst[0] + ' - ' + str(t01)
    elif 'fp' in lst[0].split('_'):
        cd = p1[lst[0]]*1e6
        lst[0] = lst[0] + ' (in ppm)'
    elif (lst[0][0:3] == 'p_p') or (lst[0][0:4] == 'p1_p') or (lst[0][0:4] == 'p2_p') or (lst[0][0:2] == 'GP'):
        cd = p1[lst[0]]
        if len(lst[0].split('_')) > 3:
            lst[0] = '_'.join(lst[0].split('_')[0:3]) + '_et al.'
        else:
            lst[0] = lst[0]
    elif (lst[0][0:2] == 'q1') or (lst[0][0:2] == 'q2'):
        cd = p1[lst[0]]
        if len(lst[0].split('_')) > 2:
            lst[0] = '_'.join(lst[0].split('_')[0:2]) + '_et al.'
        else:
            lst[0] = lst[0]
    else:
        cd = p1[lst[0]]
    for i in range(len(lst)-1):
        if 't0' in lst[i+1].split('_'):
            t02 = np.floor(p1[lst[i+1]][0])
            cd1 = p1[lst[i+1]] - t02
            cd = np.vstack((cd, cd1))
            lst[i+1] = lst[i+1] + ' - ' + str(t02)
        elif 'fp' in lst[i+1].split('_'):
            cd = np.vstack((cd, p1[lst[i+1]]*1e6))
            lst[i+1] = lst[i+1] + ' (in ppm)'
        elif (lst[i+1][0:3] == 'p_p') or (lst[i+1][0:4] == 'p1_p') or (lst[i+1][0:4] == 'p2_p') or (lst[i+1][0:2] == 'GP'):
            cd = np.vstack((cd, p1[lst[i+1]]))
            if len(lst[i+1].split('_')) > 3:
                lst[i+1] = '_'.join(lst[i+1].split('_')[0:3]) + '_et al.'
            else:
                lst[i+1] = lst[i+1]
        elif (lst[i+1][0:2] == 'q1') or (lst[i+1][0:2] == 'q2'):
            cd = np.vstack((cd, p1[lst[i+1]]))
            if len(lst[i+1].split('_')) > 2:
                lst[i+1] = '_'.join(lst[i+1].split('_')[0:2]) + '_et al.'
            else:
                lst[i+1] = lst[i+1]
        else:
            cd = np.vstack((cd, p1[lst[i+1]]))
    data = np.transpose(cd)
    value = np.median(data, axis=0)
    ndim = len(lst)
    fig = corner.corner(data, labels=lst)
    axes = np.array(fig.axes).reshape((ndim, ndim))

    for i in range(ndim):
        ax = axes[i,i]
        ax.axvline(value[i], color = 'r')

    for yi in range(ndim):
        for xi in range(yi):
            ax = axes[yi, xi]
            ax.axvline(value[xi], color = 'r')
            ax.axhline(value[yi], color = 'r')
            ax.plot(value[xi], value[yi], 'sr')

    fig.savefig(folder + "/corner.png")
    plt.close(fig)