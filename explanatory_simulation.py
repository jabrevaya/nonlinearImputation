#  -*- coding: utf-8 -*-

"""
09/16/2020
@author: Peter Toth, UNR

Explanatory Monte Carlo simulation for imputation project: probit model with a
single imputed occasionally missing RHS variable (x) using one, two, five other
(non-missing) RHS variables (z-s)

Packages needed: numpy, scipy

NOTES:
- we regenerate the missingness indicators every time
"""

import numpy as np
from scipy import optimize as opt
from scipy.stats import norm as normal
import time

#############################
#   SIMULATION PARAMETERS   #
#############################

alpha = 1
beta = [0.5, -2]
a0 = 0
b0 = [0, 0]
reps = 10
nlist = [100]
fname = 'proba.txt'
resultsFname = None
seed = 2433523
noise = True


#################
#   Functions   #
#################


def fromZToX(z):
    """ Helper for 'dgp'. Getting the x-values, given the z RHS variables.
     The length of z should be smaller than 10.
    """
    coeffs = np.array([0.6, -0.4, 0.7, 0.1, -0.3, 0.9, 1, -1, 0.5, -0.2]
                      [:z.shape[1]])
    index = np.sum(z * coeffs, axis=1, keepdims=True) \
        + np.random.normal(size=(z.shape[0], 1))
    x = 4 * np.exp(index) / (1 + np.exp(index)) - 2
    return x


def fromZToM(z):
    """ Helper for 'dgp'. Getting the missing indicators, given the z RHS
    variables. The length of z should be smaller than 10.
    """
    coeffs = np.array([0.7, -0.3, 0.75, 0.16, -0.21, 0.56, 0.76, -1.2, 0.54,
                      -0.42][:z.shape[1]])
    probit_index = np.sum(z * coeffs, axis=1, keepdims=True)\
        + np.random.normal(size=(z.shape[0], 1))
    m = ((probit_index < -0.8) | (probit_index > 0.8)).astype('int')
    return m


def dgp(alpha, beta, n, noise=False):
    """ Creating the full information data set (no missing values) assuming a
    probit model. The relationship between x and (the independent) z, also m
    and z is baked in (a weird nonparametric function described by 'fromZToX'
    and 'fromZToM', respectively).

    arguments:
    - coefficients: 'alpha' (float), 'beta' (array of floats with length k)
    - sample size: n

    returns the list of arrays:
    - y: n-by-1, the outcome variable (binary, dtype is integer)
    - x: n-by-1, the scalar, float RHS variable that has missing values later
    - z: n-by-k, the vector of RHS variables without missing values
    - m: n-by-1, vector of missing indicators (=1 when the obs. is missing)
    """
    z = np.hstack((np.ones((n, 1)),
                   4 * np.random.uniform(size=(n, len(beta)-1)) - 2))
    x = fromZToX(z)
    y = (x * alpha + np.sum(z * beta, axis=1, keepdims=True)
         > np.random.normal(size=(n, 1))) \
        .astype('int')
    m = fromZToM(z)

    if noise:
        print('Missingness rate: ', np.mean(m))
        print('Mean of y:', np.mean(y))

    return y, x, z, m


def fullDataMoments(coeffs, y, x, z):
    indices = y-normal.cdf(x * coeffs[0] + np.sum(z * coeffs[1:],
                           axis=1, keepdims=True))
    moments = np.matrix(np.mean(np.hstack((x * indices, z * indices)), axis=0))
    return (moments*moments.transpose())[0, 0]


def gmmFullData(y, x, z, a0, b0, noise=False):
    a0 = np.array(a0, ndmin=1)
    b0 = np.array(b0)
    coeffs0 = np.concatenate((a0, b0))
    optimum = opt.minimize(
                fullDataMoments, coeffs0, args=(y, x, z), method='BFGS')
    if noise:
        print(optimum.message)
    return optimum.x


def gmmNonMissingData(y, x, z, m, a0, b0, noise=False):
    a0 = np.array(a0, ndmin=1)
    b0 = np.array(b0)
    coeffs0 = np.concatenate((a0, b0))
    y = y[np.squeeze((m == 0))]
    x = x[np.squeeze((m == 0))]
    z = z[np.squeeze((m == 0))]
    optimum = opt.minimize(
                fullDataMoments, coeffs0, args=(y, x, z), method='BFGS')
    if noise:
        print(optimum.message)
    return optimum.x


def probXCondlZ(xx, zz, xVals, zs, bwidth):
    """ Takes xx and zz the nonmissing part of the sample, gridvalus from xVals
    and an n-by-k+1 array of (z, m) values (2D array) from the data set,
    and returns the kernel estimates for P[X=x|Z=z] for every x in xVals
    and z in zs as an n-by-xVals.shape[1] array.
    """
#    len(zz)
#    bwidth= (0.05, 0.05)
#    zs = np.hstack((z, m))
    if zz.shape[1] > 1:
        zz = np.stack([zz[:, 1:]] * zs.shape[0])
        zis = np.stack([zs[:, 1:-1]] * zz.shape[1])
    elif zz == np.ones(zz.shape):
        return np.mean(xx)
#    xVals = xGrid[0, :]
    xVals = np.stack([xVals] * zz.shape[1])
    kernel1 = np.prod(normal.pdf((zz - np.einsum('ijk->jik', zis))
                                 / bwidth[0]), axis=-1)
    kernel2 = normal.pdf((xx - xVals) / bwidth[1])
    results = np.einsum('ij, jk -> ik', kernel1, kernel2) \
        / np.sum(kernel1, axis=1, keepdims=True)
    return results/np.sum(results, axis=1, keepdims=True)


def yCondlOnZ(coeffs, probs, x, z, gridno):
    """ This is a not-so-dumb, but still very basic grid implementation of
    numerical integration for our simulation.
    """
    # grid: make it a n-by-gridno array
    xGrid = np.stack([np.linspace(start=-2, stop=2, num=gridno)] * len(x))
    expectedYs = normal.cdf(xGrid * coeffs[0] + np.sum(z * coeffs[1:],
                            axis=1, keepdims=True))
    return np.sum(probs * expectedYs, axis=1, keepdims=True)


def yCondlOnMarginalZs(coeffs, probsvector, x, z, gridno):
    """ This is a not-so-dumb, but still very basic grid implementation of
    numerical integration for our simulation. NEEDS WORK TO BRING UP TO v0.2
    """
    # grid: make it a n-by-gridno array
    xGrid = np.tile(np.linspace(start=-2, stop=2, num=gridno), len(x)) \
        .reshape((len(x), gridno))
    expectedYs = normal.cdf(xGrid * coeffs[0] + np.sum(z * coeffs[1:],
                            axis=1, keepdims=True))
    return np.hstack(tuple([np.sum(prob * expectedYs, axis=1, keepdims=True)
                            for prob in probsvector]))


# coeffs = [1, 0.5, -2]
# imputeMoments(coeffs, y, x, z, m, gridno, bwidth1)
# bwidth = bwidth1


def imputeMoments(coeffs, probs, y, x, z, m, gridno):
    residuals1 = (1 - m) * (y-normal.cdf(x * coeffs[0]
                                         + np.sum(z * coeffs[1:],
                                                  axis=1, keepdims=True)))
    residuals2 = m * (y - yCondlOnZ(coeffs, probs, x, z, gridno))
    moments = np.matrix(np.mean(np.hstack((x * residuals1,
                                           z * residuals1,
                                           z * residuals2)), axis=0))
    return (moments*moments.transpose())[0, 0]


def marginalizedImputeMoments(coeffs, probsvector, y, x, z, m, gridno):
    """ MAY NEED WORK TO BRING IT UP TO v0.2 """
    residuals1 = (1 - m) * (y-normal.cdf(x * coeffs[0]
                                         + np.sum(z * coeffs[1:],
                                                  axis=1, keepdims=True)))
    residuals2 = m * (y - yCondlOnMarginalZs(
                                coeffs, probsvector, x, z, gridno))
    moments = np.matrix(np.mean(np.hstack((x * residuals1,
                                           z * residuals1,
                                           z * residuals2)), axis=0))
    return (moments*moments.transpose())[0, 0]


def gmmImpute(y, x, z, m, a0, b0, gridno, bwidth, noise=False):
    # initializing variables
    a0 = np.array(a0, ndmin=1)
    b0 = np.array(b0)
    coeffs0 = np.concatenate((a0, b0))
    # nonmissing sample, grid, calculating probs (note this should be cached)
    xx = x[np.squeeze((m == 0))]
    zz = z[np.squeeze((m == 0))]
    xVals = np.linspace(start=-2, stop=2, num=gridno)
    probs = probXCondlZ(xx, zz, xVals, np.hstack((z, m)), bwidth)
    # optimization
    optimum = opt.minimize(
            imputeMoments, coeffs0, args=(probs, y, x, z, m, gridno),
            method='BFGS', options={'disp': True})
    if noise:
        print(optimum.message)
    return optimum.x


def gmmMarginalizedImpute(y, x, z, m, a0, b0, gridno, bwidth, noise=False):
    """ NEEDS WORK TO BRING IT UP TO v0.2 """
    a0 = np.array(a0, ndmin=1)
    b0 = np.array(b0)
    coeffs0 = np.concatenate((a0, b0))

    # nonmissing sample, grid, calculating probs (note this should be cached)
    xx = x[np.squeeze((m == 0))]
    zz = z[np.squeeze((m == 0))]
    xVals = np.linspace(start=-2, stop=2, num=gridno)
    probsvector = np.array([probXCondlZ(
                                xx, zz[:, i:i+1], xVals,
                                np.hstack((z[:, i:i+1], m)), bwidth)
                            for i in range(z.shape[1])])

    optimum = opt.minimize(
                marginalizedImputeMoments, coeffs0,
                args=(probsvector, y, x, z, m, gridno), method='BFGS',
                options={'disp': True})
    if noise:
        print(optimum.message)
    return optimum.x


def montecarlo(oldfuggveny):
    """ Monte Carlo decorator for a function containing one iteration of a
    simulation.

    Simulation parameters needed as global vars:
    - nlist: list(like) object containing sample sizes
    - reps: int, number of replications
    - fname: name of file where the aggregate results (mean, st. dev.)
             are to be printed
    - resultsFname: name of file where the list of estimate values are
                    to be printed

    EVERY TIME YOU RECYCLE, FILL IN THE 'meanlist' and 'stdlist' lines !!!
     """
    def iterations(*args):
        for n in nlist:
            t0 = time.time()
            results = [oldfuggveny(n, *args) for i in range(reps)]

            # THIS NEEDS TO BE CHANGED (only)
            meanlist = [0, 1, 2]
            stdlist = [2, 3, 5]

            if resultsFname:
                with open(resultsFname, 'a') as f:
                    f.write(results)
            print("")
            print('n=', n)
            print('Means:', meanlist)
            print('Std. devs.:', stdlist)
            print("")
            print("This took", time.time()-t0, 's')
            print("")
            print("")
            with open(fname, 'a') as f:
                f.write('\n\n\nn= '+str(n))
                f.write('\n\nmeans: \n'+str(meanlist))
                f.write('\n\nstandard deviations: \n' + str(stdlist))
                f.write('\n\nThis took ' + str(time.time() - t0) + ' s\n')

    return iterations


@montecarlo
def iteration(n, noise=False):
    """ One iteration of the simulation containing data generation and fitting
    the four estimators (full data set GMM, nonmissing data set GMM,
    AD imputation GMM, marginalized imputation GMM)
    Returns a numpy array.
    """
    n = 1000
    gridno = 500
    bwidth1 = [0.1, 0.1]
    bwidth2 = [0.1, 0.1]
    y, x, z, m = dgp(alpha, beta, n, noise)
    fullDataRes = gmmFullData(y, x, z, a0, b0, noise)
    nonMissingDataRes = gmmNonMissingData(y, x, z, m, a0, b0, noise)
    print(fullDataRes, nonMissingDataRes)
    imputeRes = gmmImpute(y, x, z, m, a0, b0, gridno, bwidth1, noise)
    print(imputeRes)
    marginalizedImputeRes = gmmMarginalizedImpute(y, x, z, m, a0, b0, gridno,
                                                  bwidth2, noise)
    print(imputeRes, marginalizedImputeRes)
    return np.array((fullDataRes, nonMissingDataRes, imputeRes,
                    marginalizedImputeRes))


###################
#    SIMULATION   #
###################

# with open(fname, 'w') as f:
#    f.write('beta= ' + str(beta) + '\nalpha= ' + str(alpha) + '\nseed= '
#            + str(seed) + '\nreps= ' + str(reps) + '\n')

# iteration()
