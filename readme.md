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
10/24/2020: In the linear case it ran relatively faster when we increased the number of Z variable to 5. The interesting take-away is that the imputation estimator is root-n consistent, seemingly, even though we did not adjust the rate of the bandwidth from n^-1/3. Results are in LinearNW5.txt. The imputation estimator is better at around n=10,000.
10/23/2020: The troubles for the linear case were due to a bug in the code. The sample is 'large enough' to get an overall-better performance at n=250, but the imputation estimator is better by any measure at n=10,000. The results are in LinearNW2.txt. We implemented the calculation of the true weighting matrix in trueWeighting.py. Only really usable for this simulation (it is being recalculated every time, not g).

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

- semi-marginalized imputation for linear case
- number of covariates 2-4 in the linear second stage case
- THEORY: derive the cov-var matrix
- random seed needs to be implemented - EVENTUALLY
- better nonparametric estimator (B-splines?) for conditional expectation? At least Local linear regression condl distribution of f_x|z! - LATER
- Are we interested in Oracle estimator for Linear second stage sim? - MAYBE LATER
