from ax import *
from ax.modelbridge import get_sobol
from ..config import *

def mo_config(metrics, ref_point):

    mo = MultiObjective(
        metrics=metrics,
    )

    objective_thresholds = [
        ObjectiveThreshold(metric=metric, bound=val, relative=False)
        for metric, val in zip(mo.metrics, ref_point)
    ]

    optimization_config = MultiObjectiveOptimizationConfig(
        objective=mo,
        objective_thresholds=objective_thresholds,
    )

    return optimization_config

def initialize_exp(exp: Experiment):
    sobol = get_sobol(exp.search_space, seed=OC.seed)
    for _ in range(OC.training_samples):
        exp.new_trial(sobol.gen(1)).run()
    return exp.fetch_data()