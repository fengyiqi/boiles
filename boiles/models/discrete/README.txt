These scripts are used to assist the generation of the samples during optimization. The classical Bayesian optimization
is difficult to deal with discrete domain, resulting in repeat evaluation of some specific samples.

Hear we use two methods:
1. tree-based search
2. constrained minimization.

Tree-based search will detroy the danymic model of the minimization problem, so it is not recommended currently.