# Optimizing TENO5 with Sods Shock Tube
# Discrete Inputs

# 1 Import necessary modules
from boiles.test_cases.sod_60 import SodDisper60
from boiles.config import OC, OP

OP.test_cases = [SodDisper60]
# above imports is a legacy problem, will be changed in future version

import warnings
from gpytorch.utils.warnings import NumericalWarning

warnings.filterwarnings('ignore', category=UserWarning)
warnings.filterwarnings('ignore', category=NumericalWarning)

#1 Import
from botorch.models.gpytorch import GPyTorchModel
from gpytorch.distributions import MultivariateNormal
from gpytorch.means import ConstantMean
from gpytorch.models import ExactGP
from gpytorch.kernels import RBFKernel, ScaleKernel, MaternKernel
from gpytorch.likelihoods.gaussian_likelihood import GaussianLikelihood, FixedNoiseGaussianLikelihood
from gpytorch.mlls import ExactMarginalLogLikelihood
from botorch.fit import fit_gpytorch_model
from botorch.acquisition.monte_carlo import qNoisyExpectedImprovement
from botorch.sampling.samplers import SobolQMCNormalSampler
from ax.core import SimpleExperiment
from ax.modelbridge import get_sobol
from ax.modelbridge.factory import get_botorch
from ax import ParameterType, SearchSpace, RangeParameter
from boiles.utils import append_to_csv, log
import numpy as np
import torch
import gpytorch
import random
import os
import csv

device = torch.device("cpu")
dtype = torch.double
SMOKE_TEST = os.environ.get("SMOKE_TEST")

os.system(
    "rm -rf runtime_data/case.* runtime_data/opt_history.csv runtime_data/model_state runtime_data/sod_runtime_samples.csv runtime_data/sorted_disper.csv log.txt discrete")


# define some global variables
# 365481
random_seed = 100
initial_samples = 5
opt_iterations = 100
bounds = ((4, 8), (1, 5), (0.2, 0.4))
#bounds = ((5, 7), (2, 4), (0.2, 0.25))
#bounds = ((5, 7), (2, 4), (0.3, 0.4))
#bounds = ((5, 7), (2, 4), (0.255, 0.3))
#bounds = ((5, 7), (2, 4), (0.225, 0.255))
#bounds = ((5, 7), (2, 4), (0.2, 0.225))
#bounds = ((5, 7), (2, 4), (0.2, 0.4))
#bounds = ((4, 6), (1, 10), (0.2, 0.4))
#bounds = ((1, 6), (1, 20), (0.2, 0.4))
#bounds = ((1, 6), (1, 20), (0.2, 0.4))  q max 6 is enough, C upper 20, basic 0.4 - upper limit
# the steps for each parameter
intervals = (1, 1, 0.001)


# 2 Define a GP surrogate model
class SimpleCustomGP(ExactGP, GPyTorchModel):
    _num_outputs = 1  # to inform GPyTorchModel API

    def __init__(self, train_X, train_Y):
        # squeeze output dim before passing train_Y to ExactGP
        noise = torch.ones(train_X.shape[0]) * 0.001
        super().__init__(train_X, train_Y.squeeze(-1), FixedNoiseGaussianLikelihood(noise))
        #super().__init__(train_X, train_Y.squeeze(-1), GaussianLikelihood())
        self.mean_module = ConstantMean()
        self.covar_module = ScaleKernel(
            base_kernel = MaternKernel(nu=2.5, ard_num_dims=train_X.shape[-1]),
        )
        self.to(train_X)  # make sure we're on the right device/dtype

    def forward(self, x):
        mean_x = self.mean_module(x)
        covar_x = self.covar_module(x)
        return MultivariateNormal(mean_x, covar_x)


# 3 Define a function to build GP and fit GP, return the fitted GP
def get_and_fit_simple_custom_gp(Xs, Ys, **kwargs):
    model = SimpleCustomGP(Xs[0], Ys[0])
    mll = ExactMarginalLogLikelihood(model.likelihood, model)
    fit_gpytorch_model(mll)
    #model.eval()
    #with torch.no_grad(), gpytorch.settings.fast_pred_var():
        #f_preds = model(Xs[0])
        #f_var = f_preds.variance
    #print(f_var)
    return model


# 4 Define the optimization problem
import xml.etree.ElementTree as ET
from boiles.objective.sodshocktube import Sod
from boiles.utils import append_to_csv, log
# import time

schemefile = "/home/roy_fricker/OwnPrograms/boiles/crosstest/02_sobo_teno5_sod_shock_tube_ctest/runtime_data/scheme.xml"
inputfile = "/home/roy_fricker/OwnPrograms/boiles/crosstest/02_sobo_teno5_sod_shock_tube_ctest/inputfiles/sod_60.xml"
alpaca = "/home/roy_fricker/OwnPrograms/Alpaca_all/WINTER_ALPACA_1D/build/ALPACA"

# we use a xml file to inform the ALPACA the parameters of TENO5. We can get rid of compiling in this way
def configure_scheme_xml(para):
    tree = ET.ElementTree(file=schemefile)
    root = tree.getroot()
    root[0].text = "0"
    q, cq, eta = para[0], para[1], para[2]
    ct = 1e-5
    for i, para in enumerate([q, cq, eta, ct]):
        root[i].text = str(para)
    tree.write(schemefile)

def sod_disper_sum_error(para, case_num):
    q, cq, eta = para
    configure_scheme_xml(para)
    os.system(f"cd runtime_data; mpiexec -n 4 {alpaca} {inputfile}")
    os.system(f"mv runtime_data/sod_60 runtime_data/case.{case_num}")

    sod = Sod(file=f"runtime_data/case.{case_num}/domain/data_0.200*.h5")
    sod_disper, sod_disper_raw, state_disper = sod.objective_sum_disper()

    log(f'Sod case.{case_num:<2} with q={int(q):<3} cq={int(cq):<4} eta={format(eta, ".3f"):<4}; Dispersion error: {format(sod_disper, ".4f")}')

    append_to_csv(f'runtime_data/sod_runtime_samples.csv',
                  [case_num, q, cq, eta, sod_disper, sod_disper_raw, state_disper])
    return sod_disper

class SodMetrics:

    def __init__(self):
        super().__init__()
        self.case_num = 1  # count the number of test case
        self.data_dict = {}  # use a dict to retrieve data

    def sod_error(self, x):
        q, cq, eta = x["q"], x["cq"], x["eta"]
        if self.data_dict.get(f"{str(q)}, {str(cq)}, {format(eta, '.3f')}", None) is None:
            sod_disper = -1 * sod_disper_sum_error([q, cq, eta], self.case_num)
            self.data_dict[f"{str(q)}, {str(cq)}, {format(eta, '.3f')}"] = sod_disper
            self.case_num += 1
            input_label = f"{format(q, '.1f')}, {format(cq, '.1f')}, {format(eta, '.3f')}"
            sampled_input[input_label] = (q, cq, eta)
        else:
            sod_disper = self.data_dict[f"{str(q)}, {str(cq)}, {format(eta, '.3f')}"]
            input_label = f"{format(q, '.1f')}, {format(cq, '.1f')}, {format(eta, '.3f')}"
            sampled_input[input_label] = (q, cq, eta)

        return sod_disper


sampled_input = {}
sod_metrics = SodMetrics()
name = "disper"


def sod_disper(x):
    return {name: (sod_metrics.sod_error(x), 0)}


# 5 Build an experiment and return
def build_exp(search_space, seed, minimize):
    exp = SimpleExperiment(
        name=name,
        search_space=search_space,
        evaluation_function=sod_disper,
        objective_name=name,
        minimize=minimize,
    )
    sobol = get_sobol(exp.search_space, seed=seed)
    exp.new_batch_trial(generator_run=sobol.gen(initial_samples))
    return exp


#6 Adapted acquisition strategy of optimization loop
# bounds is a global variable
axes = np.array([intervals[i] / (bounds[i][1] - bounds[i][0]) for i in range(len(bounds))])
valid_normalized_samples = np.array([np.arange(0, 1 + 1e-6, axes[i]) for i in range(len(bounds))])
constraints = []
nltrainx = []

#6.1 rounder
def rounder(values):
    def f(x):
        idx = np.argmin(np.abs(values - x))
        return values[idx]
    return np.frompyfunc(f, 1, 1)

#6.2 checks if candidate is already used
def candidate_existing(acq_cand):
    acq_cand_to_gridpoint = [rounder(valid_normalized_samples[i])(acq_cand[i]) for i in range(len(bounds))]
    acq_cand_to_funcbounds = [acq_cand_to_gridpoint[i] * (bounds[i][1] - bounds[i][0]) + bounds[i][0] for i in
                              range(len(bounds))]
    acq_cand_to_funcbounds_label = f"{format(acq_cand_to_funcbounds[0], '.1f')}, {format(acq_cand_to_funcbounds[1], '.1f')}, {format(acq_cand_to_funcbounds[2], '.3f')}"
    return acq_cand_to_funcbounds_label in sampled_input.keys()

#6.3 inner loop to get next candidate
# uses qNEI instead of UCB
# uses botorch-algorithm to evaluate next sample - no selfmade evaluation of acquisition function
def multistart_minimize2(qNEI, constraints, n_iter):
    n = 10
    num_res = 1
    q = n_iter
    x_array, acq_array = [], []
    for i in range(1, n + 1):
        # q determines the number of next samples from given training points
        # q starts with 1 and will be increased if necessary
        x_val, acq_val = optimize_acqf_and_get_observation_inner(qNEI, q, num_res)
        for a in range(0, q):
            new_x = [x_val[a][0].item(), x_val[a][1].item(), x_val[a][2].item()]
            new_acq = [acq_val.item()]
            # checks if all constraints are satisfied - uses trial and error approach
            # acquisition function is not influenced by constraints
            con_satis = 1
            for o in range(len(constraints)):
                con = ((new_x[0] - constraints[o][0]) / (axes[0] / 2)) ** 6 \
                      + ((new_x[1] - constraints[o][1]) / (axes[1] / 2)) ** 6 \
                      + ((new_x[2] - constraints[o][2]) / (axes[2] / 2)) ** 6 - 1.1
                if con < 0:
                    con_satis = 0
            if con_satis == 1:
                x_array.append(new_x)
                acq_array.append(new_acq)

    # gives list of possible next sample points
    res = np.array(x_array)
    return res

#6.4 specify optimize_acqf to determine next candidate
def optimize_acqf_and_get_observation_inner(acq_func, q, num_res):
    from botorch.optim import optimize_acqf
    # optimize
    bounds_for_opt = torch.tensor([[0.0, 0.0, 0.0], [1.0, 1.0, 1.0]], device=device, dtype=dtype)
    candidates, acq_value_list = optimize_acqf(
        acq_function=acq_func,
        bounds=bounds_for_opt,
        q=q,
        num_restarts=num_res,
        raw_samples=random_seed,  # used for intialization heuristic
    )
    # observe new values
    new_x = candidates.detach()
    acq_val = acq_value_list.detach()
    return new_x, acq_val

#6.5 gives closest value out of list for given point
def closest_val(res_gridpoint, acq_cand):
    x1, x2, x3 = res_gridpoint[:, 0], res_gridpoint[:, 1], res_gridpoint[:, 2]
    p1, p2, p3 = acq_cand[0], acq_cand[1], acq_cand[2]
    y = -((x1 - p1) ** 2 + (x2 - p2) ** 2 + (x3 - p3) ** 2)
    return y

#6.6 main algorithm - get next candidate
def get_constraint_optimized_cand(model, acq_cand):
    if not candidate_existing(acq_cand.squeeze().numpy()):
        return acq_cand

    print(acq_cand)
    print(sampled_input)

    existing_inputs = np.array([value for _, value in sampled_input.items()])
    normalized_existing_inputs = np.zeros_like(existing_inputs)
    for i in range(len(bounds)):
        normalized = (existing_inputs[:, i] - bounds[i][0]) / (bounds[i][1] - bounds[i][0])
        normalized_existing_inputs[:, i] = rounder(valid_normalized_samples[i])(normalized)

    # acq_cand is an value that has not been rounded to grid point
    acq_cand_to_gridpoint = [rounder(valid_normalized_samples[i])(acq_cand.numpy()[0][i]) for i in range(len(bounds))]
    constraints.append(acq_cand_to_gridpoint)


    # extracts training points out of sampled_input
    i = 0
    for _, x in sampled_input.items():
        adn = torch.tensor([[x[0], x[1], x[2]]], device=device, dtype=dtype)
        if i == 0:
            train_x = adn
        else:
            train_x = torch.cat([train_x, adn])
        i += 1

    ltrainx = len(train_x)
    nltrainx.append(ltrainx)

    qmc_sampler = SobolQMCNormalSampler(num_samples=random_seed)

    qNEI = qNoisyExpectedImprovement(
        model=model,
        X_baseline=train_x,
        sampler=qmc_sampler
    )

    n_iter = 0
    regenerate = False

    while n_iter < 5:
        #         if acq_x_int in raw_inputs.astype(int).tolist():
        regenerate = True
        n_iter += 1
        print(f"--- Constrained minimizing started! iteration {n_iter} ---")

        if len(bounds) == 3:
            res = multistart_minimize2(qNEI, constraints, n_iter)
            if len(res) == 0:
                continue
        else:
            raise Exception(f"Such dimension is invalid.")

        res_gridpoint = np.zeros_like(res)
        for i in range(len(bounds)):
            res_gridpoint[:, i] = rounder(valid_normalized_samples[i])(res[:, i])
        res_gridpoint = np.unique(res_gridpoint, axis=0)

        # sorts res_gridpoint - which value is closest to acq_cand_to_gridpoint?
        closest = closest_val(res_gridpoint, acq_cand_to_gridpoint)
        sort_index = np.array(closest).argsort()
        res_gridpoint = np.array(res_gridpoint)[sort_index]
        print(res_gridpoint)

        for generated_candidate in res_gridpoint:
            if not candidate_existing(generated_candidate):
                print("\tNon-repeated cand: ", torch.tensor(np.expand_dims(generated_candidate, 0)))
                return torch.tensor(np.expand_dims(generated_candidate, 0))
    print("\tRandom cand: ", torch.tensor(np.expand_dims(np.random.rand(len(bounds)), 0)))
    return torch.tensor(np.expand_dims(np.random.rand(len(bounds)), 0))


#6.7 a modification of the original 'scipy_optimizer' function.
def scipy_optimizer_discrete(
        acq_function,
        bounds,
        n: int,
        inequality_constraints=None,
        fixed_features=None,
        rounding_func=None,
        **kwargs,
):
    num_restarts: int = kwargs.get("num_restarts", 20)
    raw_samples: int = kwargs.get("num_raw_samples", 25 * num_restarts)

    if kwargs.get("joint_optimization", False):
        sequential = False
    else:
        sequential = True
        # use SLSQP by default for small problems since it yields faster wall times
        if "method" not in kwargs:
            kwargs["method"] = "SLSQP"

    from ax.models.torch.botorch_defaults import optimize_acqf

    X, expected_acquisition_value = optimize_acqf(
        acq_function=acq_function,
        bounds=bounds,
        q=n,
        num_restarts=num_restarts,
        raw_samples=raw_samples,
        options=kwargs,
        inequality_constraints=inequality_constraints,
        fixed_features=fixed_features,
        sequential=sequential,
        post_processing_func=rounding_func,
    )
    # here we call the algorithm
    X = get_constraint_optimized_cand(acq_function.model, X)
    return X, expected_acquisition_value



# 6 Define the optimization loop
from boiles.utils import save_single_model_state

def optimize(search_space, seed, minimize):
    exp = build_exp(search_space, seed, minimize)
    # building the initial GP
    model = get_botorch(
        experiment=exp,
        data=exp.eval(),
        search_space=exp.search_space,
        model_constructor=get_and_fit_simple_custom_gp,
    )

    save_single_model_state(model, 0, name)

    # start optimization loop
    for i in range(opt_iterations):
        generator_run = model.gen(1)
        exp.new_trial(generator_run=generator_run)
        model = get_botorch(
            experiment=exp,
            data=exp.eval(),
            search_space=exp.search_space,
            model_constructor=get_and_fit_simple_custom_gp,
            acqf_optimizer=scipy_optimizer_discrete
        )
        print(f"Running optimization batch {i + 1}/{opt_iterations}...")

        save_single_model_state(model, i + 1, name)

    print("Done!")
    return model, exp


# 7 Optimizing the Sods shock tube
from ax import ChoiceParameter
import warnings
warnings.filterwarnings("ignore")

parameter_space = SearchSpace(
    parameters=[
        ChoiceParameter(
            name="q",
            parameter_type=ParameterType.INT,
            values=torch.arange(bounds[0][0], bounds[0][1] + 1e-6, intervals[0]),
            is_ordered=True,
            sort_values=True
        ),
        ChoiceParameter(
            name="cq",
            parameter_type=ParameterType.INT,
            values=torch.arange(bounds[1][0], bounds[1][1] + 1e-6, intervals[1]),
            is_ordered=True,
            sort_values=True
        ),
        ChoiceParameter(
            name="eta",
            parameter_type=ParameterType.FLOAT,
            values=torch.arange(bounds[2][0], bounds[2][1] + 1e-6, intervals[2]),
            is_ordered=True,
            sort_values=True
        ),
    ]
)
model, exp = optimize(
    search_space=parameter_space,
    seed=random_seed,
    minimize=False
)

# 8 Find the optimum
from ax.service.utils.report_utils import exp_to_df

sorted_df = exp_to_df(exp).sort_values(by=[name], ascending=False).head(20)
opt_x = [sorted_df.head(1).q.values[0], sorted_df.head(1).cq.values[0], sorted_df.head(1).eta.values[0]]

best_x20 = np.zeros((20, 4))
for i in range(0, 20):
    best_x20[i] = [sorted_df.head(20).q.values[i], sorted_df.head(20).cq.values[i], sorted_df.head(20).eta.values[i], sorted_df.head(20).disper.values[i]]
append_to_csv(f'runtime_data/sorted_disper.csv', best_x20)


# 9 Plot a 3D volume using Plotly
from ax.models.torch.utils import predict_from_model
import plotly.graph_objects as go
import gpytorch

# get all samples
data = []
for obs in model.get_training_data():
    x = obs.features.parameters
    data.append([x["q"], x["cq"], x["eta"], obs.data.means[0]])
data = np.array(data)

n = 30
x = torch.linspace(0, 1, n).reshape(-1, 1)
x_test = gpytorch.utils.grid.create_data_from_grid(x.repeat(1, 3))
pred, var = predict_from_model(model.model.model, x_test)
pred = pred.numpy().reshape((n, n, n))

# don't doubt the order, it is correct
y, z, x = np.meshgrid(
    torch.linspace(bounds[1][0], bounds[1][1], n).reshape(-1, 1),
    torch.linspace(bounds[2][0], bounds[2][1], n).reshape(-1, 1),
    torch.linspace(bounds[0][0], bounds[0][1], n).reshape(-1, 1)
)

fig_list = []
volume = go.Volume(
    x=x.flatten(),
    y=y.flatten(),
    z=z.flatten(),
    value=pred.flatten(),
    isomin=pred.min(),
    isomax=abs(pred.min()) * 0.1 + pred.min(),
    opacity=0.3,  # needs to be small to see through all surfaces
    surface_count=20,  # needs to be a large number for good volume rendering
    colorscale="Viridis"
)
fig_list += [volume]
scatter = go.Scatter3d(
    x=data[:, 0].flatten(),
    y=data[:, 1].flatten(),
    z=data[:, 2].flatten(),
    mode="markers",
    marker=dict(color="red", symbol="x", size=1.5)
)
fig = go.Figure()
fig.add_trace(volume)
fig.add_trace(scatter)

fig.update_layout(
    autosize=True,
    scene=dict(
        xaxis=dict(range=[bounds[0][0], bounds[0][1]]),
        yaxis=dict(range=[bounds[1][0], bounds[1][1]]),
        zaxis=dict(range=[bounds[2][0], bounds[2][1]])
    ),
    width=600,
    height=600,
    title="GP",
    showlegend=False,
)
fig.show()

# 10 Plot the Sods shock tube with optimal and Lins parameters
import matplotlib.pyplot as plt

os.system("rm -rf runtime_data/sod*")

inputfile = "/home/roy_fricker/OwnPrograms/boiles/crosstest/02_sobo_teno5_sod_shock_tube_ctest/inputfiles/sod_60.xml"
alpaca = "/home/roy_fricker/OwnPrograms/Alpaca_all/WINTER_ALPACA_1D/build/ALPACA"

configure_scheme_xml([6, 1, 0.200])
os.system(f"cd runtime_data; mpiexec -n 4 {alpaca} {inputfile}")
os.system(f"mv runtime_data/sod_60 runtime_data/sod_lin")

configure_scheme_xml(opt_x)
os.system(f"cd runtime_data; mpiexec -n 4 {alpaca} {inputfile}")
os.system(f"mv runtime_data/sod_60 runtime_data/sod_opt")

sod_opt = Sod(file="runtime_data/sod_opt/domain/data_0.200000.h5")
sod_lin = Sod(file="runtime_data/sod_lin/domain/data_0.200000.h5")
states = ("density", "velocity", "pressure")
x = sod_lin.result["x_cell_center"]
plt.figure(figsize=(15, 4), dpi=100)
for i, state in enumerate(states):
    plt.subplot(1, 3, i + 1)
    plt.title(state)
    plt.plot(x, sod_lin.reference[state], c="black", linewidth=0.6)
    plt.scatter(x, sod_lin.result[state], label=r"$TENO5_{Lin}$", marker="s", facecolor='none', edgecolor="blue",
                linewidth=0.5, s=40)
    plt.scatter(x, sod_opt.result[state], label=r"$TENO5_{Opt}$", marker="s", facecolor='none', edgecolor="red",
                linewidth=0.5, s=40)
    plt.legend()

plt.show()

print("opt_x: ", opt_x)
print(best_x20)

