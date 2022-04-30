# BoiLES
Bayesian optimization for implicit Large Eddy Simulation schemes, as well as for shock-capturing scheme optimization.

This repo is a an interface between ALPACA and Bayesian optimization tools ([Ax](https://ax.dev/), [botorch](https://botorch.org/), [gpytorch](https://gpytorch.ai/) ...).
ALPACA, is named by "Adaptive Level-set Parallel Code". It is a multiresolution compressible multiphase flow solver developed at Prof. Adams' [Chair of Aerodynamics and Fluid Mechanics (AER)](https://www.mw.tum.de/en/aer/home/), [Technical University of Munich](https://www.tum.de/en/).

## Reading Materials

### Gaussian Processes

Gaussian processes (GP) is a widely used surrogate model for Bayesian optimization. We use [gpytorch](https://gpytorch.ai/) to build and train GP. Recommended learning resources may include

- [*A Visual Exploration of Gaussian Processes*](https://distill.pub/2019/visual-exploration-gaussian-processes/) 

- [*Gaussian Processes for Machine Learning*](http://gaussianprocess.org/gpml/chapters/RW.pdf), Chapter 2, Regression

Practice [gpytorch](https://gpytorch.ai/) package with:

- [*GPyTorch Regression Tutorial*](https://docs.gpytorch.ai/en/stable/examples/01_Exact_GPs/Simple_GP_Regression.html)

- [*Saving and Loading Models*](https://docs.gpytorch.ai/en/stable/examples/00_Basic_Usage/Saving_and_Loading_Models.html)

### Bayesian Optimization

Bayesian optimization finds samples to evaluate based on posterior distribution of the surrogate model (GP). The "finding" algorithm can be formulated into the acquisition function.
Learning resources may include 

- [*Bayesian optimization*](http://krasserm.github.io/2018/03/21/bayesian-optimization/)

- [*Tutorial #8: Bayesian optimization*](https://www.borealisai.com/en/blog/tutorial-8-bayesian-optimization/)

- [*A Tutorial on Bayesian Optimization*](https://arxiv.org/pdf/1807.02811.pdf)

We perform Bayesian optimization through [Ax](https://ax.dev/), a platform for sequential experimentation. Practice [Ax](https://ax.dev/) with:

- [*Multi-Objective Optimization Ax API*](https://ax.dev/tutorials/multiobjective_optimization.html)

or practice [botorch](https://botorch.org/) with:

- [*Noisy, Parallel, Multi-Objective BO in BoTorch with qEHVI, qNEHVI, and qNParEGO*](https://botorch.org/tutorials/multi_objective_bo)

You may find the expected hypervolume improvement (EHVI) algorithm that is used for multi-objective BO in above tutorials. Unfortunately there are little resources explaining EHVI well. You may learn it from 

- [Multi-Objective Bayesian Global Optimization using expected hypervolume improvement gradient](https://www.sciencedirect.com/science/article/pii/S2210650217307861)

### Optimization Problem: Nonlinear Reconstruction Scheme

Our intention is to optimize nonlinear reconstruction scheme to achieve better performance using multi-objective Bayesian optimization. We studied the WENO-CU6-M1 scheme and have several publications as

- *Optimization of an Implicit Large-Eddy Simulation Method for Underresolved Incompressible Flow Simulations*.

    You may find it in [Felix's dissertation](https://mediatum.ub.tum.de/1327615) -> Major Peer-reviewed Journal Publications. 
The "Basic Equations" and "Numercal Methods" parts in the dissertation are also worth reading, only single-phase.

- [*Iterative Bayesian Optimization of an Implicit LES Method for Underresolved Simulations of Incompressible Flows*](http://www.tsfp-conference.org/proceedings/2017/2/207.pdf)

- [*A Multi-Objective Bayesian Optimization Environment for Systematic Design of Numerical Schemes for Compressible Flow*](https://papers.ssrn.com/sol3/papers.cfm?abstract_id=4028726)

    This is the preprint of our paper where we started using the multi-objective Bayesian optimization to deal with our problem.