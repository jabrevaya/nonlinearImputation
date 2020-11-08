#  -*- coding: utf-8 -*-

"""
10/23/2020
@author: Peter Toth, UNR

Simulate the optimal weighting matrix for the imputation estimator when the DGP
has a linear second stage and the nonlinear first stage. Helper file to
linear_simulation.py.

Packages needed: numpy, scipy

NOTES:
"""

import numpy as np
from scipy import linalg
from scipy.stats import norm as normal


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
    y = x * alpha + np.sum(z * beta, axis=1, keepdims=True) \
        + np.random.normal(size=(n, 1))
    m = fromZToM(z)
    if noise:
        print('Missingness rate: ', np.mean(m))
        print('Mean of y:', np.mean(y))
    return y, x, z, m


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


def yCondlOnZ(coeffs, xCondlOnZ, z):
    """ Conditional expectation returns.
    """
    expectedYs = xCondlOnZ * coeffs[0] \
        + np.sum(z * coeffs[1:], axis=1, keepdims=True)
    return expectedYs


def imputeTrueWeights(coeffs, xCondlOnZ, y, x, z, m):
    """The objective function for the imputation estimator that uses the
    analogue of the AD 2017 moments in addition to the feasible moments in
    gmmNonMissingData.
    Its arguments are
    - coeffs: (k+1 numpy array) the coefficient values
    - probs: conditional probabilities for P[X=xgridval|Z=z] for some xgridvals
             from a grid of X values generated linearly based on gridno (array)
    - y, x, z: the data in separate arrays (n-by-1, n-by-1, n-by-k shapes)
    - m: missingness indicator as 2D array
    """
    residuals1 = (1 - m) * (y - (x * coeffs[0]
                            + np.sum(z * coeffs[1:], axis=1, keepdims=True)))
    residuals2 = m * (y - yCondlOnZ(coeffs, xCondlOnZ, z))
    moments = np.matrix(np.hstack((x * residuals1,
                                   z * residuals1,
                                   z * residuals2)))
    return linalg.inv((moments.transpose() * moments) / y.shape[0])


def getWeights(coeffs, n, gridno, noise=False):
    y, x, z, m = dgp(coeffs[0], coeffs[1:], n, noise)
    xVals = np.linspace(start=-1.999, stop=1.999, num=gridno)
    xCondlOnZ = XCondlZOracle(z, xVals)
    return imputeTrueWeights(coeffs, xCondlOnZ, y, x, z, m)
