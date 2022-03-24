import os
import shutil
from boiles.config.opt_config import OC

OC.seed = 1234
OC.training_samples = 10
OC.opt_iterations = 20
OC.dim_inputs = 2
OC.dim_outputs = 1
OC.opt_bounds = [(0, 50), (0, 50)]
OC.increment = (0.3, 0.3)
OC.fun_bounds = [(-5, 10), (0, 15)]

OC.discrete = {
        "activate": True,
        "method": "constrained",  # selection from "tree" and "constrained"
        "plot": True,
        "max_level": 6,  # only valid if method is tree.
        "counter": 1
    }

if os.path.exists("log.txt"):
    os.remove("log.txt")
if os.path.exists("runtime_data.csv"):
    os.remove("runtime_data.csv")
if os.path.exists("discrete"):
    shutil.rmtree("discrete")
if OC.discrete["activate"]:
    os.makedirs("discrete")


from botorch.models.gpytorch import GPyTorchModel
from gpytorch.distributions import MultivariateNormal
from gpytorch.means import ConstantMean
from gpytorch.models import ExactGP
from gpytorch.kernels import RBFKernel, ScaleKernel
from gpytorch.likelihoods.gaussian_likelihood import (
    GaussianLikelihood
)
from gpytorch.mlls import ExactMarginalLogLikelihood
from botorch.fit import fit_gpytorch_model
from ax import SimpleExperiment
from ax.modelbridge import get_sobol
from ax import ParameterType, SearchSpace, ChoiceParameter
import numpy as np
import torch
import random
import warnings
from gpytorch.utils.warnings import *
from boiles.models.factory import get_botorch
from boiles.utils import log, append_to_csv

warnings.filterwarnings('ignore', category=UserWarning)
warnings.filterwarnings('ignore', category=NumericalWarning)
warnings.filterwarnings('ignore', category=GPInputWarning)
warnings.filterwarnings('ignore', category=OldVersionWarning)


# define a GP surrogate model
class SimpleCustomGP(ExactGP, GPyTorchModel):
    _num_outputs = 1  # to inform GPyTorchModel API

    def __init__(self, train_X, train_Y):
        # squeeze output dim before passing train_Y to ExactGP
        super().__init__(train_X, train_Y.squeeze(-1), GaussianLikelihood())
        self.mean_module = ConstantMean()
        self.covar_module = ScaleKernel(
            base_kernel=RBFKernel(ard_num_dims=train_X.shape[-1]),
        )
        self.to(train_X)  # make sure we're on the right device/dtype

    def forward(self, x):
        mean_x = self.mean_module(x)
        covar_x = self.covar_module(x)
        return MultivariateNormal(mean_x, covar_x)


# define a function to build GP and fit GP, return the fitted GP
def _get_and_fit_simple_custom_gp(Xs, Ys, **kwargs):
    model = SimpleCustomGP(Xs[0], Ys[0])
    mll = ExactMarginalLogLikelihood(model.likelihood, model)
    fit_gpytorch_model(mll)
    return model


# one can adapt this function based on the problem you would like to optimize, just keep the arguments and return.
def branin(parameterization, *args):
    x1, x2 = parameterization["x1"], parameterization["x2"]
    # if we use the discrete assistant, the "parameterization" is a int value, we have to perform a linear
    # transformation to get the true input for the function.
    if OC.discrete["activate"]:
        x1 = x1 * OC.increment[0] + OC.intercept[0]
        x2 = x2 * OC.increment[1] + OC.intercept[1]
    y = (x2 - 5.1 / (4 * np.pi ** 2) * x1 ** 2 + 5 * x1 / np.pi - 6) ** 2
    y += 10 * (1 - 1 / (8 * np.pi)) * np.cos(x1) + 10
    # let's add some synthetic observation noise
    y += random.normalvariate(0, 0.1)
    # if we use the discrete assistant, we have to pass the data from outside
    append_to_csv(f"runtime_data.csv", [x1, x2, y])
    if OC.discrete["activate"]:
        append_to_csv(f"discrete/data.csv", [parameterization["x1"], parameterization["x2"], y])
    return {"branin": (y, 0.0)}


# define a experiment container, return this experiment
def build_exp(search_space, seed, minimize=True):
    name = "branin"
    exp = SimpleExperiment(
        name=name,
        search_space=search_space,
        evaluation_function=branin,
        objective_name=name,
        minimize=minimize,
    )
    sobol = get_sobol(exp.search_space, seed=seed)
    exp.new_batch_trial(generator_run=sobol.gen(OC.training_samples))
    return exp


def optimize(search_space, seed, minimize=True):
    exp = build_exp(search_space, seed, minimize)

    for i in range(OC.opt_iterations):
        model = get_botorch(
            experiment=exp,
            data=exp.eval(),
            search_space=exp.search_space,
            model_constructor=_get_and_fit_simple_custom_gp,
        )

        generator_run = model.gen(1)
        exp.new_trial(generator_run=generator_run)
        log(f"Running optimization batch {i + 1}/{OC.opt_iterations}...")
        if OC.discrete["activate"]:
            OC.discrete["counter"] += 1

    log("Done!")

    return model


if __name__ == "__main__":

    parameter_space = SearchSpace(
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
        ]
    )
    model = optimize(
        search_space=parameter_space,
        seed=OC.seed
    )

