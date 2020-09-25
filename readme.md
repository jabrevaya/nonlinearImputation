# Nonlinear imputation

This project is about the analogue of Abrevaya and Donald (2017) for a nonlinear parametric second stage model with nonparametric missigness structure.

So far we only have exploratory simulation in explanatory_simulation.py (a typo in the main file name, how elegant).

## Estimators

Given y,(x,z) LHS and set of RHS variables, m missingness indicator, known g function and parameters of interest \[a,b]...

1. The full data GMM moments: E[(x, z) (y-g(ax+bz)]=0
2. The non-missing data GMM: E[m(x, z)(y-g(ax+bz)]=0
3. Imputation GMM: Supposed to improve on the efficiency of the non-missing data GMM by adding the feasible moments
E[(1-m) z (y-E[g(ax+bz)|z]]=0
4. Marginalized imputation GMM: The same as 3., except we only condition on a single dimension of z

The estimators appear in this order in the simulation results (as of today, proba.txt, proba2.txt).

Currently, the estimator for cond'l expectation E[g(ax+bz)|z] is based on simple grid-based numerical integration coupled with a Nadaraya-Watson estimator estimating the probabilities P[X=x|Z-z]

## Notes

09/23/2020: We needed to impose missingness to around 50% (around 55%)
09/25/2020: Initial basic simulations are running, using the code,
but only for the 'joint' imputation estimator (and the full and nonmissing sample gmms)


### TODO?

- random seed needs to be implemented
- weighting matrix needs to be implemented
- optimal bandwidth selection?
- linearization of known function g?
- better nonparametric estimator (B-splines?) for conditional expectation?
- 2 moving parts: nonparametric missingness structure + nonlinear second stage; should we do only one of these at first? (It would help the computation, the very least)
