# Nonlinear imputation

This project is about the analogue of Abrevaya and Donald (2017) for a nonlinear parametric second stage model with nonparametric missigness structure.

So far we only have exploratory simulation in explanatory_simulation.py (a typo in the main file name, how elegant).

## Estimators

Given y,(x,z) LHS and set of RHS variables, m missingness indicator, known g function and parameters of interest \[a,b]...

1. The full data GMM moments: E[(x, z) (y-g(ax+bz)]=0
2. The non-missing data GMM: E[m(x, z)(y-g(ax+bz)]=0
3. Oracle Imputation GMM: Supposed to improve on the efficiency of the non-missing data GMM by adding the infeasible moments
E[(1-m) z (y-E[g(ax+bz)|z]]=0, after assuming that we know the true conditional distribution f_x|z (still includes numerical integration for the nonlinear second stage)
4. Imputation GMM: Supposed to improve on the efficiency of the non-missing data GMM by adding the feasible moments
E[(1-m) z (y-E[g(ax+bz)|z]]=0
5. Marginalized imputation GMM: The same as 3., except we only condition on a single dimension of z - not done!

The estimators appear in this order in the simulation results (as of today):
* proba.txt, proba2.txt (old, no weighting matrix)
* NaiveWeighting1000Sim.txt, 1000hist.png - results available
* OracleVsNW.txt (also report from cleaned results) - results available
* LinearNW.txt (also report from cleaned results) - results available

Currently, the estimator for cond'l expectation E[g(ax+bz)|z] is based on simple grid-based numerical integration coupled with a Nadaraya-Watson estimator estimating the probabilities P[X=x|Z-z], a spline implementation is STILL not done/

## Notes
10/07/2020: Take-aways:
1. In the first  basic simulation with weighting matrix the picture is what we expected (~25-30% improvement in the overall MSE)
2. Then problems arise with weighting matrix calculation (invertibility), which given the numerical errors, probably makes the next iteration of GMM minimization a concave problem
3. When cleaned (4 iterations out of 200), the Oracle performs (almost?) identically with even at n=1,000 in terms of alpha, and much better at the beta-s, pretty close to the full data GMM, so while numerical integration/nonlinearities do take away some of the info, the efficiency loss due to missingness can be mostly recovered
4. More troubles come for the linear case: when identity=weighting matrix, the imputation actually INCREASES variance, the biases are basically the same. When we have iterative GMM for 1 iteration, even the full info GMM breaks sometimes with n=500, but the imputation ones much more often even at n=1000 or 4000.
10/07/2020: Three new simulations
* once-iterated GMM with weighting matrix in NaiveWeighting1000Sim
* Oracle estimator GMM in Oracle*.txt
* Linear model (no numerical integration) in Linear*
09/25/2020: Initial basic simulations are running, using the code,
but only for the 'joint' imputation estimator (and the full and nonmissing sample gmms)
09/23/2020: We needed to impose missingness to around 50% (around 55%)

### TODO?

- random seed needs to be implemented
- Save results - DONE 10/06
- weighting matrix needs to be implemented - DONE 10/06
- optimal bandwidth selection? (better rule-of-thumb) (Voted down)
- better nonparametric estimator (B-splines?) for conditional expectation? At least Local linear regression!
- 2 moving parts: nonparametric missingness structure + nonlinear second stage; should we do only one of these at first? (It would help the computation, the very least) - DONE 10/07
- Are we interested in Oracle estimator for Linear second stage sim?
- What to do with the weighting matrix inversion problem? (Talk about directions)
