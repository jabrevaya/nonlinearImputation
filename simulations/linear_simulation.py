#  -*- coding: utf-8 -*-

"""
10/16/2020
@author: Peter Toth, UNR

Exploratory Monte Carlo simulation for imputation project: a linear model (in
the observable with missing values) with a single imputed occasionally missing
RHS variable (x) using one (non-missing) RHS variable (z)

Packages needed: numpy, scipy

NOTES:
- we regenerate the missingness indicators every time
"""

import numpy as np
from scipy import optimize as opt
from scipy import linalg
from scipy.stats import norm as normal
import time
import trueWeighting

#############################
#   SIMULATION PARAMETERS   #
#############################

# Structural PARAMETERS (missing at random indicator for now)

mar = False

# True values
alpha = 1
beta = [0.5, -0.9, 0.6, -0.3]

# Initial values
a0 = 0
b0 = [0, 0, 0, 0]

# Replication number, sample sizes
reps = 1000
nlist = [250, 500, 1000]

# File names (fname is req'd, stores the aggregate results, resultsFname
# can be set to False)
fname = 'Linear4NoWeight.txt'
resultsFname = 'Linear4NoWeight'

# Random seed (not implemented), noisiness (should be True only for dev)
seed = 2433523
noise = True


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


def dgp(alpha, beta, n, mar=False, noise=False):
    """ Creating the full information data set (no missing values) assuming a
    probit model. The relationship between x and (the independent) z, also m
    and z is baked in (a weird nonparametric function described by 'fromZToX'
    and 'fromZToM', respectively).

    arguments:
    - coefficients: 'alpha' (float), 'beta' (array of floats with length k)
    - sample size: n
    - mar: missing at random indicator

    returns the list of arrays:
    - y: n-by-1, the outcome variable (binary, dtype is integer)
    - x: n-by-1, the scalar, float RHS variable that has missing values later
    - z: n-by-k, the vector of RHS variables without missing values
    - m: n-by-1, vector of missing indicators (=1 when the obs. is missing)
    """
    z = np.hstack((np.ones((n, 1)),
                   4 * np.random.uniform(size=(n, len(beta)-1)) - 2))
    x = fromZToX(z)
    y = x * alpha + np.sum(z * beta, axis=1, keepdims=True) \
        + np.random.normal(size=(n, 1))
    if mar:
        m= (np.random.uniform(size=(n,1))>0.6) * 1
    else:
        m = fromZToM(z)
    if noise:
        print('Missingness rate: ', np.mean(m))
        print('Mean of y:', np.mean(y))
    return y, x, z, m


def fullDataMoments(coeffs, y, x, z, weight=None):
    """ Creates the objective function for the estimator that assumes that the
    full dat set is available without missing values.
    Its arguments are the coefficient values 'coeffs' (k+1 numpy array) and
    the data y, x, z in separate 2D arrays. Returns the value of the GMM
    objective function as a float.
    """
    indices = y - (x * coeffs[0]
                   + np.sum(z * coeffs[1:], axis=1, keepdims=True))
    moments = np.matrix(np.mean(np.hstack((x * indices, z * indices)), axis=0))
    if weight is None:
        weight = np.matrix(np.identity(x.shape[1]+z.shape[1]))
    return (moments * weight * moments.transpose())[0, 0]


def fullDataWeights(coeffs, y, x, z):
    """ Estimated optimal weights for the fully-observed moments as a function
    of (true) coefficients and the data.
    """
    indices = y - (x * coeffs[0]
                   + np.sum(z * coeffs[1:], axis=1, keepdims=True))
    moments = np.matrix(np.hstack((x * indices, z * indices)))
    return linalg.inv((moments.transpose() * moments) / y.shape[0])


def gmmFullData(y, x, z, a0, b0, weighting_iteration=1, noise=False):
    """ The infeasible GMM estimator that is applied for the full data set
    pretending the missing x values are there.
    Arguments:
    - y, x, z: variables from the data as (n-by-1, n-by-1, n-by-k+1) 2D arrays
    - a0, b0: the initial values for maximization (1D arrays) - the dimensions
              must agree with the number of columns in x and z
    - noise: boolean, set it to True if want to print the messages of the
             optimizer (automatically suppressed when iteration() is run with
             the MC decorator)
    """
    a0 = np.array(a0, ndmin=1)
    b0 = np.array(b0)
    coeffs0 = np.concatenate((a0, b0))
    weight = None
    for i in range(weighting_iteration):
        optimum = opt.minimize(
                    fullDataMoments, coeffs0, args=(y, x, z, weight),
                    method='BFGS')
        coeffs0 = optimum.x
        weight = fullDataWeights(coeffs0, y, x, z)
    optimum = opt.minimize(
                fullDataMoments, coeffs0, args=(y, x, z, weight),
                method='BFGS')
    if noise:
        print(optimum.message)
    return optimum.x


def gmmNonMissingData(y, x, z, m, a0, b0, weighting_iteration=1, noise=False):
    """ The feasible GMM estimator that uses the same moments as the infeasible
    estimator gmmFullData, but is only applied for the part of the data that
    is fully observed (non-missing x values, when m=0).
    Arguments:
    - y, x, z: variables from the data as (n-by-1, n-by-1, n-by-k+1) 2D arrays
    - m: missingness indicator, another 2D array (n-by-1)
    - a0, b0: the initial values for maximization (1D arrays) - the dimensions
              must agree with the number of columns in x and z
    - noise: boolean, set it to True if want to print the messages of the
             optimizer (automatically suppressed when iteration() is run with
             the MC decorator)
    """
    a0 = np.array(a0, ndmin=1)
    b0 = np.array(b0)
    y = y[np.squeeze((m == 0))]
    x = x[np.squeeze((m == 0))]
    z = z[np.squeeze((m == 0))]
    return gmmFullData(y, x, z, a0, b0, weighting_iteration, noise)


def xCondlOnZ(zz, xx, z, bwidth):
    if zz.shape[1] > 1:
        zz = np.stack([zz[:, 1:]] * z.shape[0])
        zis = np.stack([z[:, 1:]] * zz.shape[1])
    elif np.array_equal(zz, np.ones(zz.shape)):
        return np.mean(xx) * np.ones(z.shape)
    else:
        zz = np.stack([zz] * z.shape[0])
        zis = np.stack([z] * zz.shape[1])
    kernel1 = np.prod(normal.pdf((zz - np.einsum('ijk->jik', zis))
                                 / bwidth[0]), axis=-1)
    results = np.asarray(np.asmatrix(kernel1) * np.asmatrix(xx)) \
        / np.sum(kernel1, axis=1, keepdims=True)
    return results


def XCondlZOracle(z, xVals):
    """ Takes the z-s (1D array) and grid values from xVals (1D array),
    and returns the true value for P[X=x|Z=z] for every x in xVals and z in z
    as an n-by-xVals.shape[1] array.
    """
    if z.shape[1] > 1:
        zs = z[:, 1]
    else:
        zs = z
    z_stack = np.stack([zs] * xVals.shape[0], axis=1)
    x_stack = np.stack([xVals] * z.shape[0])
    results = normal.pdf(-np.log(4 / (x_stack + 2) - 1) - 0.6 + 0.4 * z_stack)\
        * 4 / (4 - x_stack**2)
    return np.sum(results/np.sum(results, axis=1, keepdims=True) * xVals,
                  axis=1, keepdims=True)


def xCondlOnZSpline(zz, xx, z):

    return None


def yCondlOnZ(coeffs, xCondlZ, z):
    """ Conditional expectation returns.
    """
    expectedYs = xCondlZ * coeffs[0] \
        + np.sum(z * coeffs[1:], axis=1, keepdims=True)
    return expectedYs


def imputeMoments(coeffs, xCondlZ, y, x, z, m, weight=None):
    """The objective function for the imputation estimator that uses the
    analogue of the AD 2017 moments in addition to the feasible moments in
    gmmNonMissingData.
    Its arguments are
    - coeffs: (k+1 numpy array) the coefficient values
    - probs: conditional probabilities for P[X=xgridval|Z=z] for some xgridvals
             from a grid of X values generated linearly based on gridno (array)
    - y, x, z: the data in separate arrays (n-by-1, n-by-1, n-by-k shapes)
    - m: missingness indicator as 2D array

    Returns the value of the GMM objective function as a float.
    """
    residuals1 = (1 - m) * (y - (x * coeffs[0]
                            + np.sum(z * coeffs[1:], axis=1, keepdims=True)))
    residuals2 = m * (y - yCondlOnZ(coeffs, xCondlZ, z))
    moments = np.matrix(np.mean(np.hstack((x * residuals1,
                                           z * residuals1,
                                           z * residuals2)), axis=0))
    if weight is None:
        weight = np.matrix(np.identity(moments.shape[1]))
    return (moments * weight * moments.transpose())[0, 0]


def imputeMomentsWeights(coeffs, xCondlOnZ, y, x, z, m):
    """ Estimated optimal weights for the fully-observed moments as a function
    of (true) coefficients and the data.
    """
    residuals1 = (1 - m) * (y - (x * coeffs[0]
                            + np.sum(z * coeffs[1:], axis=1, keepdims=True)))
    residuals2 = m * (y - yCondlOnZ(coeffs, xCondlOnZ, z))
    moments = np.matrix(np.hstack((x * residuals1,
                                   z * residuals1,
                                   z * residuals2)))
    return linalg.inv((moments.transpose() * moments) / y.shape[0])


def gmmImpute(y, x, z, m, a0, b0, w_iteration=1, method='NW', noise=False):
    """ The feasible GMM estimator that adds the analogues of the AD 2017
    moments to the moments of the gmmNonMissingData estimator.
    Arguments:
    - y, x, z: variables from the data as (n-by-1, n-by-1, n-by-k+1) 2D arrays
    - m: missingness indicator, another 2D array (n-by-1)
    - a0, b0: the initial values for maximization (1D arrays) - the dimensions
              must agree with the number of columns in x and z
    - bwidth: a list-like of two floats, where the first number is the (equal)
              bwidth for the kernels for the Z dimensions, and the second one
              is the bandwidth for the normal kernel for the pdf estimator for
              the X
    - noise: boolean, set it to True if want to print the messages of the
             optimizer (automatically suppressed when iteration() is run with
             the MC decorator)
    """
    # initializing variables
    a0 = np.array(a0, ndmin=1)
    b0 = np.array(b0)
    coeffs0 = np.concatenate((a0, b0))
    n = y.shape[0]
    # nonmissing sample, grid, calculating probs (note this should be cached)
    xx = x[np.squeeze((m == 0))]
    zz = z[np.squeeze((m == 0))]
    if method == 'spline':
        knotno = int(500 / (np.exp(n**1/2) + 1))
        xGivenZ = xCondlOnZSpline(zz, xx, z, knotno)
    elif method == 'NW':
        bwidth1 = [2.154 * n**(-1/3), 0]
        xGivenZ = xCondlOnZ(zz, xx, z, bwidth1)
    elif method == 'trueweight':
        bwidth1 = [2.154 * n**(-1/3), 1.077 * n**(-1/3)]
        xGivenZ = xCondlOnZ(zz, xx, z, bwidth1)
    elif method == 'oracle':
        gridno = 10000
        xVals = np.linspace(start=-1.999, stop=1.999, num=gridno)
        xGivenZ = XCondlZOracle(z, xVals)
    # optimization
    weight = None
    if method == 'trueweight':
        gridnoo = 1000
        nn = 10000
        weight = trueWeighting.getWeights([alpha] + beta, nn, gridnoo,
                                          noise=False)
    for i in range(w_iteration):
        optimum = opt.minimize(
                    imputeMoments, coeffs0,
                    args=(xGivenZ, y, x, z, m, weight),
                    method='BFGS')
        coeffs0 = optimum.x
        weight = imputeMomentsWeights(coeffs0, xGivenZ, y, x, z, m)
    optimum = opt.minimize(
            imputeMoments, coeffs0, args=(xGivenZ, y, x, z, m, weight),
            method='BFGS', options={'disp': noise})
    if noise:
        print(optimum.message)

    return optimum.x


def yCondlOnZk(coeffs, xCondlZk, z):
    """ Conditional expectation of Y given Z_k=z_k for every observation.

    Inputs:
     - coeffs: candidate coefficient probsvector (1D array-like)
     - xCondlZk: 2D-array (n-times-1) conditional expectation of X given
       Z-_k=z_k for every row
     - z: 2D array (n-times-K), the z observable values from the data set

    Returns: 2D array of E[Y_i|Z^k_i=z^k_i]
    """
    expectedYs = xCondlZk * coeffs[0] \
        + np.sum(z * coeffs[1:], axis=1, keepdims=True)
    return expectedYs


def mImputeMoments(coeffs, xCondlZk, y, x, z, m, weight=None, lst='all'):
    """The objective function for the imputation estimator that uses the
    analogue of the AD 2017 moments in addition to the feasible moments in
    gmmNonMissingData.
    Its arguments are
    - coeffs: (k+1 numpy array) the coefficient values
    - xCondlZk: conditional expectation E[X|Z_k=z_k] (n-times-len(list) or
                n-times-K 2D array)
    - y, x, z: the data in separate arrays (n-by-1, n-by-1, n-by-k shapes)
    - m: missingness indicator as 2D array
    - weight: weighting matrix for the moments (defaults to 'None' = identity)
    - list: the list of moments required (defaults to 'all': all Z-s)

    Returns the value of the GMM objective function as a float.
    """
    residuals1 = (1 - m) * (y - (x * coeffs[0]
                            + np.sum(z * coeffs[1:], axis=1, keepdims=True)))
    if lst == 'all':
        lst = range(xCondlZk.shape[1])
    moments2 = np.hstack([z[:, lst[i]] * m
                         * (y - yCondlOnZk(coeffs, xCondlZk[:, i:i+1], z))
                         for i in range(len(lst))])
    moments = np.matrix(np.mean(np.hstack((x * residuals1,
                                           z * residuals1,
                                           moments2)), axis=0))
    if weight is None:
        weight = np.matrix(np.identity(moments.shape[1]))

    return (moments * weight * moments.transpose())[0, 0]


def mImputeMomentsWeights(coeffs, xCondlZk, y, x, z, m, lst='all'):
    """ Estimates the optimal weighting matrix, based on coeffificent estimates
    and data for the marginalized imputation GMM estimator.

    Arguments:
    - coeffs: 1D-array or list-like
    - xCondlZk: matrix of conditional expectation of X_i given Z_k=z_k
                (2D array, n-by-len(list))
    - y,x,z,m: the data
    - list: the list of z-dimensions we use for the imputation

    Returns the estimated optimal weighting matrix as 2D matrix.
    """
#    coeffs=coeffs0
#    xCondlZk=xGivenZ
    residuals1 = (1 - m) * (y - (x * coeffs[0]
                            + np.sum(z * coeffs[1:], axis=1, keepdims=True)))
    if lst == 'all':
        lst = range(xCondlZk.shape[1])
    moments2 = np.hstack([z[:, lst[i]] * m
                         * (y - yCondlOnZk(coeffs, xCondlZk[:, i:i+1], z))
                         for i in range(len(lst))])
    moments = np.matrix(np.hstack((x * residuals1, z * residuals1, moments2)))

    return linalg.inv((moments.transpose() * moments) / y.shape[0])


def gmmMarginalizedImpute(y, x, z, m, a0, b0, lst='all', w_iteration=1, noise=False):
    """ The feasible GMM estimator that adds the analogues of the AD 2017
    moments to the moments of the gmmNonMissingData estimator but 1-by-1.
    Arguments:
    - y, x, z: variables from the data as (n-by-1, n-by-1, n-by-k+1) 2D arrays
    - m: missingness indicator, another 2D array (n-by-1)
    - a0, b0: the initial values for maximization (1D arrays) - the dimensions
              must agree with the number of columns in x and z
    - bwidth: a list-like of two floats, where the first number is the (equal)
              bwidth for the kernels for the Z dimensions, and the second one
              is the bandwidth for the normal kernel for the pdf estimator for
              the X
    - list: 1D-list of z-s that imputation for x is done based on; defaults to
            'all', when all marginal moments are included in the calculation
    - weighting iteration: number of times the weighting matrix should be
                           iterated, defaults to 1
    - noise: boolean, set it to True if want to print the messages of the
             optimizer (automatically suppressed when iteration() is run with
             the MC decorator)

    Returns the estimated values as a single 1D-array.
    """
    # initializing variables
    a0 = np.array(a0, ndmin=1)
    b0 = np.array(b0)
    coeffs0 = np.concatenate((a0, b0))
    n = y.shape[0]
    if lst == 'all':
        lst = range(z.shape[1])

    # nonmissing sample, grid, calculating conditional expectation
    xx = x[np.squeeze((m == 0))]
    zz = z[np.squeeze((m == 0))]
    bwidth1 = [1.4 * n**(-1/5), 0]
    xGivenZ = np.hstack([xCondlOnZ(zz[:, i:i+1], xx, z[:, i:i+1], bwidth1)
                        for i in lst])
    # optimization
    weight = None
    for i in range(w_iteration):
        optimum = opt.minimize(
                    mImputeMoments, coeffs0,
                    args=(xGivenZ, y, x, z, m, weight, lst),
                    method='BFGS')
        coeffs0 = optimum.x
        weight = mImputeMomentsWeights(coeffs0, xGivenZ, y, x, z, m, lst)
    optimum = opt.minimize(
            mImputeMoments, coeffs0, args=(xGivenZ, y, x, z, m, weight, lst),
            method='BFGS', options={'disp': noise})
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
    You may also want to add names of estimators to print into the file if
    there are too many to compare.

    NOTE: REORGANIZE estimation parameter printing too.
     """
    def iterations(*args):
        for n in nlist:
            t0 = time.time()
            results = np.stack([oldfuggveny(n, *args) for i in range(reps)])

            # THIS NEEDS TO BE CHANGED (only), add name if you would like to!
            meanlist = np.mean(results, axis=0)
            stdlist = np.std(results, axis=0)

            if resultsFname:
                with open(resultsFname + str(n), 'wb') as f:
                    np.save(f, results)
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
    y, x, z, m = dgp(alpha, beta, n, mar, noise)
    fullDataRes = gmmFullData(y, x, z, a0, b0, 0, noise)
    nonMissingDataRes = gmmNonMissingData(y, x, z, m, a0, b0, 0, noise)
# imputeTrueWeightRes = gmmImpute(y, x, z, m, a0, b0, 1, 'trueweight', noise)
    imputeNWRes = gmmImpute(y, x, z, m, a0, b0, 0, 'NW', noise)
#    marginalizedImputeRes = gmmMarginalizedImpute(y, x, z, m, a0, b0,
#                                                  [1, 2], 1, noise)
    # DO THE STUFF: BLOCK MATRIX
    return np.array((fullDataRes, nonMissingDataRes, imputeNWRes))
                    # marginalizedImputeRes))

###################
#    SIMULATION   #
###################


with open(fname, 'w') as f:
    f.write('beta= ' + str(beta) + '\nalpha= ' + str(alpha) + '\nseed= '
            + str(seed) + '\nreps= ' + str(reps) + '\n')

iteration()
print('\a')
