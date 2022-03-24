#!/usr/bin/env python
# coding: utf-8

# In[ ]:
from mytools.config.opt_config import OC
from mytools.config.opt_problems import OP
from mytools.test_cases.sod_60 import SodDisper60, SodShock60
OC.initial_samples = 50
OC.opt_iterations = 50
OC.bounds = [(28, 45), (1, 100), (1, 20)]
OC.num_inputs = 3
OC.num_outputs = 2

OP.test_cases = [SodDisper60, SodShock60]
OP.ref_point = [SodDisper60.ref_point, SodShock60.ref_point]

from ax import *

import torch
from ax.metrics.noisy_function import NoisyFunctionMetric
from ax.plot.exp_utils import exp_to_df
from ax.runners.synthetic import SyntheticRunner
from mytools.models.factory import get_moo_ehvi_botorch
from mytools.test_problem.sod_tgv import SodTGV

import os
import numpy as np


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


x1 = RangeParameter(name="d1", lower=OC.bounds[0][0], upper=OC.bounds[0][1], parameter_type=ParameterType.INT)
x2 = RangeParameter(name="Cq", lower=OC.bounds[1][0], upper=OC.bounds[1][1], parameter_type=ParameterType.INT)
x3 = RangeParameter(name="q", lower=OC.bounds[2][0], upper=OC.bounds[2][1], parameter_type=ParameterType.INT)

search_space = SearchSpace(
    parameters=[x1, x2, x3],
)


# In[ ]:


class MetricSodDisper(NoisyFunctionMetric):
    def f(self, x: np.ndarray) -> float:
        sod_disper, _ = sod_tgv.sod_error(x)
        return sod_disper


class MetricSodShock(NoisyFunctionMetric):
    def f(self, x: np.ndarray) -> float:
        _, sod_shock = sod_tgv.sod_error(x)
        return sod_shock


class MetricTGV(NoisyFunctionMetric):
    def f(self, x: np.ndarray) -> float:
        return sod_tgv.tgv_error(x)


metric_sod_disper = MetricSodDisper("disper", ["d1", "Cq", "q"], noise_sd=0.0, lower_is_better=True)
metric_sod_shock = MetricSodShock("shock", ["d1", "Cq", "q"], noise_sd=0.0, lower_is_better=True)
metric_tgv = MetricTGV("tgv", ["d1", "Cq", "q"], noise_sd=0.0, lower_is_better=True)

# In[ ]:


mo = MultiObjective(
    metrics=[metric_sod_disper, metric_sod_shock],
)

objective_thresholds = [
    ObjectiveThreshold(metric=metric, bound=val, relative=False)
    for metric, val in zip(mo.metrics, sod_tgv.ref_point)
]

optimization_config = MultiObjectiveOptimizationConfig(
    objective=mo,
    objective_thresholds=objective_thresholds,
)


# In[ ]:


def build_experiment():
    experiment = Experiment(
        name="teno5_opt",
        search_space=search_space,
        optimization_config=optimization_config,
        runner=SyntheticRunner(),
    )
    return experiment


# In[ ]:

from ax.modelbridge import get_sobol


def initialize_experiment(experiment):
    sobol = get_sobol(experiment.search_space, seed=OC.random_seed)

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
ehvi_model = None

start = 0
iteration = num_optimization

# used to generate the initial model
ehvi_model = get_moo_ehvi_botorch(
    experiment=ehvi_experiment,
    data=ehvi_data,
    model_constructor=get_and_fit_model,
    device=torch.device('cpu'),
)
# plot_runtime_contour(['sod', 'tgv'], ehvi_model, 0)
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

        # plot_runtime_contour(['sod', 'tgv'], ehvi_model, i+1)
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






