#!/usr/bin/env python
# coding: utf-8

from boiles.test_cases.shuosher_200 import ShuDisper200, ShuShock200
from boiles.test_cases.tgv import TGV
from boiles.config.opt_problems import OP
from boiles.config.opt_config import OC
from boiles.solvers.solver_config import SC
# OP, OC and SC are Singleton classes. They should be configured at the very beginning of the script
OP.test_cases = [ShuDisper200, ShuShock200, TGV]
OP.ref_point = [ShuDisper200.ref_point, ShuShock200.ref_point, TGV.ref_point]

OC.seed = 1234
OC.training_samples = 30
OC.opt_iterations = 30
OC.dim_inputs = 2
OC.dim_outputs = 3
OC.opt_bounds = [(20, 400), (1, 20)]
OC.fun_bounds = [(1000, 20000), (1, 20)]
OC.increment = (50, 1)
OC.case_folder = 'runtime_data'
OC.debug = False

SC.solver_folder = "~/alpaca/bayesian/alpaca_mobo"
SC.Riemann = "hllc"

from ax import *

import torch
import numpy as np
import os
from ax.metrics.noisy_function import NoisyFunctionMetric
from ax.modelbridge.modelbridge_utils import observed_hypervolume
from ax.service.utils.report_utils import exp_to_df
from ax.runners.synthetic import SyntheticRunner

from boiles.models.factory import get_moo_parego_botorch
from boiles.test_problem.shu_tgv import ShuOsherTGV
from boiles.config import OC, OP
from boiles.helpers.mo_helper import mo_config, initialize_exp
from boiles.utils import append_to_csv, save_model_state, log
from boiles.models.botorch_defaults import get_and_fit_model

# a counter to count the number of thr cases
case_num = 1
# create data holder directory
if not os.path.exists('runtime_data'):
    os.makedirs('runtime_data')

# ChoiceParameter is used for discrete input domain. One could use RangeParameter for continuous domain
x1 = ChoiceParameter(name="cq", values=torch.arange(OC.opt_bounds[0][0], OC.opt_bounds[0][1]+1).numpy().tolist(), is_ordered=True, parameter_type=ParameterType.INT)
x2 = ChoiceParameter(name="q", values=torch.arange(OC.opt_bounds[1][0], OC.opt_bounds[1][1]+1).numpy().tolist(), is_ordered=True, parameter_type=ParameterType.INT)

# instantiate the optimization problem
search_space = SearchSpace(parameters=[x1, x2])
shuosher_tgv = ShuOsherTGV(negate=False, case_num=case_num).to(
    dtype=torch.double,
    device=torch.device("cpu"),
)

# create three metric classes, required by the package.
class MetricShuOsherDisper(NoisyFunctionMetric):
    def f(self, x: np.ndarray) -> float:
        error_disper, _ = shuosher_tgv.shu_error(x)
        return error_disper
class MetricShuOsherShock(NoisyFunctionMetric):
    def f(self, x: np.ndarray) -> float:
        _, error_shock = shuosher_tgv.shu_error(x)
        return error_shock
class MetricTGV(NoisyFunctionMetric):
    def f(self, x: np.ndarray) -> float:
        return shuosher_tgv.tgv_error(x)


metric_shu_disper = MetricShuOsherDisper("disper", ["cq", "q"], noise_sd=0.0, lower_is_better=True)
metric_shu_shock = MetricShuOsherShock("shock", ["cq", "q"], noise_sd=0.0, lower_is_better=True)
metric_tgv = MetricTGV("tgv", ["cq", "q"], noise_sd=0.0, lower_is_better=True)

# a helper function to create multiobjective optimization config
optimization_config = mo_config(
    metrics=[metric_shu_disper, metric_shu_shock, metric_tgv],
    ref_point=shuosher_tgv._ref_point
)

# create an experiment
ehvi_experiment = Experiment(
    name="m1_opt",
    search_space=search_space,
    optimization_config=optimization_config,
    runner=SyntheticRunner(),
)
# generate initial data
ehvi_data = initialize_exp(ehvi_experiment)

# core of the optimization
for i in range(OC.opt_iterations):
    log(f"Start {i+1} iteration...")

    ehvi_model = get_moo_parego_botorch(
        experiment=ehvi_experiment,
        data=ehvi_data,
        model_constructor=get_and_fit_model,
        device=torch.device('cpu'),
    )
    # generate a candidate
    generator_run = ehvi_model.gen(1)
    # evaluate the candidate
    trial = ehvi_experiment.new_trial(generator_run=generator_run)
    trial.run()
    # output data during optimization
    ehvi_data = Data.from_multiple_data([ehvi_data, trial.fetch_data()])
    exp_df = exp_to_df(ehvi_experiment)
    exp_df.to_csv(f'{OC.case_folder}/opt_history.csv')

    try:
        hv = observed_hypervolume(modelbridge=ehvi_model)
    except:
        hv = 0
        log("Failed to compute hv")

    log(f"Iteration: {i + 1}, HV: {hv}")
    # save the gp model state dict for post processing
    save_model_state(ehvi_model, i+1)
    # save hv history
    append_to_csv(f'{OC.case_folder}/hv_list.csv', [hv])






