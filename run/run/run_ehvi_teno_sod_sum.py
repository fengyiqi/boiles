# !/usr/bin/env python
# coding: utf-8

# In[ ]:

from mytools.config import *
from mytools.test_cases.sod_60 import SodDisper60
import os
import shutil

case_num = 1
OC.seed = 1234
OC.initial_samples = 50
OC.opt_iterations = 50
OC.opt_bounds = [(28, 45), (1, 100), (1, 20)]
OC.fun_bounds = [(0.28, 0.45), (1, 100), (1, 20)]
OC.increment = (0.01, 1, 1)
OC.dim_inputs = 3
OC.dim_outputs = 1
OC.discrete = {
        "activate": True,
        "method": "constrained",  # selection from "tree" and "constrained"
        "plot": True,
        "max_level": 6,  # only valid if method is tree.
        "counter": 1
    }
OP.test_cases = [SodDisper60]
OP.ref_point = [SodDisper60.ref_point]

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
from mytools.models.factory import get_moo_ehvi_botorch, get_botorch
from mytools.test_problem.sod_tgv import SodTGV

import os
import numpy as np

import shutil
from mytools.utils import log

if not os.path.exists('runtime_data'):
    os.makedirs('runtime_data')

sod_tgv = SodTGV(negate=False, case_num=case_num).to(
    dtype=torch.double,
    device=torch.device("cpu"),
)

search_space = SearchSpace(
    parameters=[
        ChoiceParameter(
            name="x1",
            parameter_type=ParameterType.INT,
            values=torch.arange(OC.opt_bounds[0][0], OC.opt_bounds[0][1] + 1),
            is_ordered=True
        ),
        ChoiceParameter(
            name="x2",
            parameter_type=ParameterType.INT,
            values=torch.arange(OC.opt_bounds[1][0], OC.opt_bounds[1][1] + 1),
            is_ordered=True
        ),
        ChoiceParameter(
            name="x3",
            parameter_type=ParameterType.INT,
            values=torch.arange(OC.opt_bounds[2][0], OC.opt_bounds[2][1] + 1),
            is_ordered=True
        ),
    ]
)


def sod_disper_sum_error_(x: np.ndarray):
    return {"tgv": (sod_tgv.sod_disper_sum_error(x, props=["density"]), 0)}


experiment = SimpleExperiment(
    name="teno5_opt",
    search_space=search_space,
    evaluation_function=sod_disper_sum_error_,
    objective_name='tgv',
    minimize=True,
)

from ax.modelbridge import get_sobol

sobol = get_sobol(experiment.search_space, seed=OC.seed)
experiment.new_batch_trial(generator_run=sobol.gen(OC.initial_samples))

import warnings
from mytools.utils import save_single_model_state
from mytools.models.botorch_defaults import get_and_fit_model

warnings.filterwarnings('ignore', category=UserWarning)

start = 0
iteration = OC.opt_iterations

# used to generate the initial model
ei_model = get_botorch(
    experiment=experiment,
    data=experiment.eval(),
    model_constructor=get_and_fit_model,
    device=torch.device('cpu'),
)
# print(ei_model.model.model)
# plot_runtime_contour(['sod', 'tgv'], ehvi_model, 0)
# runtime_cross_validation(ehvi_model, 0)
save_single_model_state(ei_model, 0)

while True:
    for i in range(start, int(start) + int(iteration)):
        log(f"Start {i + 1} iteration...")
        ei_model = get_botorch(
            experiment=experiment,
            data=experiment.eval(),
            model_constructor=get_and_fit_model,
            device=torch.device('cpu'),
        )

        trial = experiment.new_trial(generator_run=ei_model.gen(1))
        #    print(shuosher_tgv.case_num)
        # trial = experiment.new_trial(generator_run=generator_run)
        # trial.run()
        ei_data = Data.from_multiple_data([experiment.eval(), trial.fetch_data()])
        exp_df = exp_to_df(experiment)
        exp_df.to_csv(f'{OC.case_folder}/opt_history.csv')

        # plot_runtime_contour(['sod', 'tgv'], ehvi_model, i+1)
        # runtime_cross_validation(ehvi_model, i+1)
        save_single_model_state(ei_model, i + 1)
        # cross_validation(ehvi_model, i)
        if OC.discrete["activate"]:
            OC.discrete["counter"] += 1

        # write_to_csv(f'{OC.case_folder}/hv_list.csv', ehvi_hv_list.reshape(-1, 1))
    start = int(start) + int(iteration)
    continues = input("Continue? ")
    if continues == 'y' or continues == "Y":
        iteration = input("How many iterations? ")
    else:
        break
