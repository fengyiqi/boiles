#!/usr/bin/env python
# coding: utf-8

from mytools.config import *
from mytools.solvers.solver_config import SC
from mytools.test_cases.sod_60 import SodDisper60
import os
import shutil

OC.seed = 1234
OC.initial_samples = 50
OC.opt_iterations = 50
OC.opt_bounds = [(28, 45), (1, 100), (1, 20)]
OC.fun_bounds = [(0.28, 0.45), (1, 100), (1, 20)]
OC.increment = (0.01, 1, 1)
OC.dim_inputs = 3
OC.dim_outputs = 3
OC.discrete = {
        "activate": False,
        "method": "constrained",  # selection from "tree" and "constrained"
        "plot": True,
        "max_level": 6,  # only valid if method is tree.
        "counter": 1
    }
OP.test_cases = [SodDisper60, SodDisper60, SodDisper60]
OP.ref_point = [SodDisper60.ref_point, SodDisper60.ref_point, SodDisper60.ref_point]
SC.Riemann = "roe"

if os.path.exists("log.txt"):
    os.remove("log.txt")
if os.path.exists("discrete"):
    shutil.rmtree("discrete")
os.makedirs("discrete")

from ax import *

import torch
from ax.metrics.noisy_function import NoisyFunctionMetric
from ax.plot.exp_utils import exp_to_df
from ax.runners.synthetic import SyntheticRunner
from mytools.models.factory import get_moo_ehvi_botorch
from mytools.test_problem.sod_tgv import SodTGV
import numpy as np
import os

case_num = 1
num_samples = OC.initial_samples
num_optimization = OC.opt_iterations

if not os.path.exists('runtime_data'):
    os.makedirs('runtime_data')

sod_tgv = SodTGV(negate=False, case_num=case_num).to(
    dtype=torch.double,
    device=torch.device("cpu"),
)

# In[ ]:


x1 = ChoiceParameter(
            name="x1",
            parameter_type=ParameterType.INT,
            values=torch.arange(OC.opt_bounds[0][0], OC.opt_bounds[0][1] + 1),
            is_ordered=True
        )
x2 = ChoiceParameter(
            name="x2",
            parameter_type=ParameterType.INT,
            values=torch.arange(OC.opt_bounds[1][0], OC.opt_bounds[1][1] + 1),
            is_ordered=True
        )
x3 = ChoiceParameter(
            name="x3",
            parameter_type=ParameterType.INT,
            values=torch.arange(OC.opt_bounds[2][0], OC.opt_bounds[2][1] + 1),
            is_ordered=True
        )

search_space = SearchSpace(
    parameters=[x1, x2, x3],
)


# In[ ]:


class MetricSodDisper(NoisyFunctionMetric):
    def f(self, x: np.ndarray) -> float:
        density, _, _ = sod_tgv.sod_disper_mul(x)
        return density


class MetricSodShock(NoisyFunctionMetric):
    def f(self, x: np.ndarray) -> float:
        _, pressure, _ = sod_tgv.sod_disper_mul(x)
        return pressure


class MetricTGV(NoisyFunctionMetric):
    def f(self, x: np.ndarray) -> float:
        _, _, velocity = sod_tgv.sod_disper_mul(x)
        return velocity


disper_density = MetricSodDisper("density", ["x1", "x2", "x3"], noise_sd=0.0, lower_is_better=True)
disper_pressure = MetricSodDisper("pressure", ["x1", "x2", "x3"], noise_sd=0.0, lower_is_better=True)
disper_velocity = MetricSodDisper("velocity", ["x1", "x2", "x3"], noise_sd=0.0, lower_is_better=True)

# In[ ]:
from ax.core.outcome_constraint import (
    ComparisonOp,
    ObjectiveThreshold,
    OutcomeConstraint,
)

mo = MultiObjective(
    metrics=[disper_density, disper_pressure, disper_velocity],
)

objective_thresholds = [
    ObjectiveThreshold(metric=metric, bound=val, relative=False)
    for metric, val in zip(mo.metrics, sod_tgv.ref_point)
]
from mytools.config import *
optimization_config = MultiObjectiveOptimizationConfig(
    objective=mo,
    objective_thresholds=objective_thresholds,
)

def build_experiment():
    experiment = Experiment(
        name="teno5sensor_opt",
        search_space=search_space,
        optimization_config=optimization_config,
        runner=SyntheticRunner(),
    )
    return experiment


# In[ ]:

from ax.modelbridge import get_sobol


def initialize_experiment(experiment):
    sobol = get_sobol(experiment.search_space, seed=OC.seed)

    for _ in range(num_samples):
        experiment.new_trial(sobol.gen(1)).run()

    return experiment.fetch_data()


ehvi_experiment = build_experiment()
ehvi_data = initialize_experiment(ehvi_experiment)

# In[ ]:


import warnings
from mytools.utils import plot_runtime_contour, write_to_csv, save_model_state, log
from mytools.models.botorch_defaults import get_and_fit_model
from mytools.postprocessing.cross_validation import runtime_cross_validation
warnings.filterwarnings('ignore', category=UserWarning)

ehvi_hv_list = np.array([])

start = 0
iteration = num_optimization

# used to generate the initial model
ehvi_model = get_moo_ehvi_botorch(
    experiment=ehvi_experiment,
    data=ehvi_data,
    model_constructor=get_and_fit_model,
    device=torch.device('cpu'),
)
# plot_runtime_contour(['disper', 'shock', 'tgv'], ehvi_model, 0)
# runtime_cross_validation(ehvi_model, 0)
save_model_state(ehvi_model, 0)

while True:
    for i in range(start, int(start) + int(iteration)):
        log(f"Start {i+1} iteration...")
        ehvi_model = get_moo_ehvi_botorch(
            experiment=ehvi_experiment,
            data=ehvi_data,
            model_constructor=get_and_fit_model,
            device=torch.device('cpu'),
        )
        generator_run = ehvi_model.gen(1)
        #    print(shuosher_tgv.case_num)
        trial = ehvi_experiment.new_trial(generator_run=generator_run)
        trial.run()
        ehvi_data = Data.from_multiple_data([ehvi_data, trial.fetch_data()])
        exp_df = exp_to_df(ehvi_experiment)
        exp_df.to_csv(f'{OC.case_folder}/opt_history.csv')

        try:
            hv = ehvi_model.observed_hypervolume()
        except:
            hv = 0
            log("Failed to compute hv")
        ehvi_hv_list = np.append(ehvi_hv_list, [hv])
        log(f"Iteration: {i + 1}, HV: {hv}")

        # plot_runtime_contour(['disper', 'shock', 'tgv'], ehvi_model, i+1)
        # runtime_cross_validation(ehvi_model, i+1)
        save_model_state(ehvi_model, i+1)
        # cross_validation(ehvi_model, i)

        write_to_csv(f'{OC.case_folder}/hv_list.csv', ehvi_hv_list.reshape(-1, 1))
    start = int(start) + int(iteration)
    continues = input("Continue? ")
    if continues == 'y' or continues == "Y":
        iteration = input("How many iterations? ")
    else:
        break






