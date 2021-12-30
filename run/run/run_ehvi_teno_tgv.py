# !/usr/bin/env python
# coding: utf-8

from mytools.config import *
from mytools.test_cases.tgv_teno5 import TGVTENO5
case_num = 1

OC.initial_samples = 50
OC.opt_iterations = 50
OC.bounds = [(20, 60), (1, 20), (1, 20)]
OC.num_inputs = 3
OC.num_outputs = 1

TC.using_tree = True
TC.opt_bounds = OC.bounds
TC.function = "tgv"
TC.step = [0.01, 1000, 1]
TC.func_bounds = [(0.2, 0.6), (1000, 20000), (1, 20)]
TC.count = 0
TC.max_level = 5

OP.test_cases = [TGVTENO5]
OP.ref_point = [TGVTENO5.ref_point]

# currently this parameter doesn't work
SC.Riemann = "hllc"

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




if os.path.exists(f"{TC.function}.csv"):
    os.remove(f"{TC.function}.csv")
if os.path.exists("log.txt"):
    os.remove("log.txt")
if os.path.exists("plotly"):
    shutil.rmtree("plotly")
os.makedirs("plotly")

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
            values=torch.arange(TC.opt_bounds[0][0], TC.opt_bounds[0][1]+1),
            is_ordered=True
        ),
        ChoiceParameter(
            name="x2",
            parameter_type=ParameterType.INT,
            values=torch.arange(TC.opt_bounds[1][0], TC.opt_bounds[1][1]+1),
            is_ordered=True
        ),
        ChoiceParameter(
            name="x3",
            parameter_type=ParameterType.INT,
            values=torch.arange(TC.opt_bounds[2][0], TC.opt_bounds[2][1]+1),
            is_ordered=True
        ),
    ]
)


def tgv_error(x: np.ndarray):
    return {"tgv": (sod_tgv.tgv_error(x), 0)}


experiment = SimpleExperiment(
    name="teno5_opt",
    search_space=search_space,
    evaluation_function=tgv_error,
    objective_name='tgv',
    minimize=True,
)


from ax.modelbridge import get_sobol

sobol = get_sobol(experiment.search_space, seed=OC.random_seed)
experiment.new_batch_trial(generator_run=sobol.gen(OC.initial_samples))

import warnings
from mytools.utils import plot_runtime_contour, write_to_csv, save_single_model_state
from mytools.models.botorch_defaults import get_and_fit_model
from mytools.postprocessing.cross_validation import runtime_cross_validation
warnings.filterwarnings('ignore', category=UserWarning)

ehvi_hv_list = np.array([])
ehvi_model = None

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
        log(f"Start {i+1} iteration...")
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
        save_single_model_state(ei_model, i+1)
        # cross_validation(ehvi_model, i)
        TC.count += 1

        # write_to_csv(f'{OC.case_folder}/hv_list.csv', ehvi_hv_list.reshape(-1, 1))
    start = int(start) + int(iteration)
    continues = input("Continue? ")
    if continues == 'y' or continues == "Y":
        iteration = input("How many iterations? ")
    else:
        break






