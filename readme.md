# Nonlinear imputation

This project is about the analogue of Abrevaya and Donald (2017) for a nonlinear
 parametric second stage model with nonparametric missingness structure.

We have exploratory simulation in explanatory_simulation.py for a nonlinear
second stage (between Y and (X,Z)) and linear_simulation.py in the 'simulation'
directory. The LaTeX write-ups with pdf-s are in the "latexfiles" library. For
reproducibility, a Singularity container with a version of Ubuntu and python3
where the code runs securely is available at TBA.

## Visions and AIs - 02/24/2022

* Jason: the linear model can be extended.
*  * Show that we get efficiency gains
*  * get the proofs done (model is fine)
*  * Calculate asymptotic variance in simple cases
*  * deciding the set of sims
*   * * simulations for the linear case

* Jiangang: 
* 1. model (general) ok
* 2. ok with estimators
* 3. ok with NW
* 4. 
* 5. Proofs: higher order kernel in estimation as question

intercept, one z and an x, it is a double-linear model (y~x,z; x~z, x is MAR, 50%)


AIs
[P] : review proof from Jiangang
[Ji] : simulations linear model, comparing AD directly with nonparametric
[P] : upload papers cited, make folders
[Ji] : uploads Wooldridge paper + latex he wrote wqith proofs
[P] : edit proofs, take stock whats missing
[Jason] : Intro, literature review?

Next meeting: [P] circle back for 03/17 Thursday


## Estimators

(A1) Given y,(x,z) LHS and set of RHS variables, m missingness indicator that is
1 if x is unobserved, a know population moment exists for which
     E[g(y,x,z; b)|x,z]=0 \\iff b=\\beta
x,z-as. (We assume identification.)

(A2) Further, we need support conditions on (x,z): sufficiently, the support of
the RHS variables is the same unconditionally an conditionally on m=0.

(A3) y is independent of m, given x,z

We establish 4 estimators:

1. The GMM based on the full data moments: E[g(y,x,z;b)]=0
2. The complete case GMM based on: E[(1-m) g(y,x,z;b)]=0
3. Oracle Imputation GMM: Supposed to improve on the efficiency of the
non-missing data GMM by adding the infeasible moments
E[(1-m) E[g(y,x,z,b)|z]]=0 where the LHS is viewed as as a function, after
assuming that we know the true conditional distribution f_x|z
(still includes numerical integration for the nonlinear second stage)
4. Imputation GMM: Supposed to improve on the efficiency of the non-missing data
GMM by adding the feasible moments when we plug in an estimator for the
conditional distribution E[g(y,x,z,b)|z]
5. Marginalized imputation GMM: The same as 3., except we only condition on k<3
dimensions of z - only done for the linear case!

Given (A1)-(A3), the GMM estimators 1-2. are root-n consistent
(after assuming random sampling and other usual regularity conditions).

If we further strengthen (A3) to

(A3') y,x is independent of m, conditional on z^1; for z^1 variables that are a
 subset of z,

then the GMM estimator 3-4) (and presumably, 5) are also consistent.

The estimators appear in this order in the simulation results (as of today):
* proba.txt, proba2.txt (old, no weighting matrix)
* NaiveWeighting1000Sim.txt, 1000hist.png
* OracleVsNW.txt
* LinearNW.txt - results available
* + a lot more - TO BE REDONE TO CLEAN UP

Currently, the estimator for cond'l expectation E[g(y,x,z;b)|z] is based on
simple grid-based numerical integration coupled with a Nadaraya-Watson estimator
estimating the probabilities P[X=x|Z=z] in the nonlinear case.

## Notes
* 11/08/2020: Some theoretical results proven, now nothing surprising about
Monte Carlos. Marginalized estimator needs more exclusion restriction, even
having trouble with making it work with MCAR though

* 11/08/2020: README file corrections (some) + reorganization of github
repo + slides uploaded from Wednesday

* 10/24/2020: In the linear case it ran relatively faster when we increased the
number of Z variable to 5. The interesting take-away is that the imputation
estimator is root-n consistent, seemingly, even though we did not adjust the
rate of the bandwidth from n^-1/3. Results are in LinearNW5.txt. The imputation
estimator is better at around n=10,000.

* 10/23/2020: The troubles for the linear case were due to a bug in the code.
The sample is 'large enough' to get an overall-better performance at n=250,
but the imputation estimator is better by any measure at n=10,000.
The results are in LinearNW2.txt. We implemented the calculation of the true
weighting matrix in trueWeighting.py. Only really usable for this simulation
(it is being recalculated every time, not g).

* 10/07/2020: Take-aways - SOME PROVED OUT TO BE FALSE:
1. In the first  basic simulation with weighting matrix the picture is what we
expected (~25-30% improvement in the overall MSE)
2. (FALSE) Then problems arise with weighting matrix calculation
(invertibility), which given the numerical errors, probably makes the next
iteration of GMM minimization a concave problem
3. When cleaned (4 iterations out of 200), the Oracle performs (almost?)
identically with even at n=1,000 in terms of alpha, and much better at the
beta-s, pretty close to the full data GMM, so while numerical
integration/nonlinearities do take away some of the info, the efficiency
loss due to missingness can be mostly recovered
4. More troubles come for the linear case: when identity=weighting matrix,
the imputation actually INCREASES variance, the biases are basically the same.
When we have iterative GMM for 1 iteration, even the full info GMM breaks
sometimes with n=500, but the imputation ones much more often even at
n=1000 or 4000.

* 10/07/2020: Three new simulations
   - once-iterated GMM with weighting matrix in NaiveWeighting1000Sim
   -  Oracle estimator GMM in Oracle*.txt
   - Linear model (no numerical integration) in Linear*

* 09/25/2020: Initial basic simulations are running, using the code,
but only for the 'joint' imputation estimator (and the full and nonmissing
sample gmms)
* 09/23/2020: We needed to impose missingness to around 50% (around 55%) -
questionable in the end

### TODO?
- Maybe the linear case as DGP and compare AD 2017 with nonparametric
- "complete case GMM"; tilde missing at (11), hats in (17)
- PETER: theory finish (define the estimator that only uses fully observed observations, write down proofs better, clean up)  NO PROOFS!!!
- JASON: lit review, introduction
- the Overleaf write-up actually contains 2 different ways to estimate f(a,b;z)=E[y|z,m=0]=E[y|z,m=1]; the second one may be better (1 bandwidth) - LATER
- random seed needs to be implemented - LATER
- redo the simulations that were interesting - LATER
- continuous updating (have more iterations for the weighting matrix in code) - LATER
- search for bugs in the marginalized case - MAYBE MUCH LATER
- better nonparametric estimator (B-splines?) for conditional expectation? At least Local linear regression condl distribution of f_x|z! - MUCH LATER
- Are we interested in Oracle estimator for Linear second stage sim? - MAYBE MUCH LATER
