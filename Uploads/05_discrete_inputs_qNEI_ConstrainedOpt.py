# script for discrete input space
# uses qNoisyExpectedImprovement instead of UCB

#1 Import
from botorch.models.gpytorch import GPyTorchModel
from botorch.models import FixedNoiseGP, ModelListGP
from gpytorch.distributions import MultivariateNormal
from gpytorch.means import ConstantMean
from gpytorch.models import ExactGP
from gpytorch.kernels import RBFKernel, ScaleKernel
from gpytorch.likelihoods.gaussian_likelihood import GaussianLikelihood
from gpytorch.mlls import ExactMarginalLogLikelihood
from botorch.fit import fit_gpytorch_model
from ax.core import SimpleExperiment
from ax.modelbridge import get_sobol
from ax import ParameterType, SearchSpace, RangeParameter
import numpy as np
from ax.modelbridge.factory import get_botorch
import random
import torch
import os

device = torch.device("cpu")
dtype = torch.double
SMOKE_TEST = os.environ.get("SMOKE_TEST")

random_seed = 10
initial_samples = 10
opt_iterations = 20
bounds = ((-5, 10), (0, 15))
# the steps for each parameter
intervals = (0.6, 0.6)

cancel_point = (np.pi, 2.275)
cancel_rad = 2


#2 GP surrogate model
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


#3 fit GP
def get_and_fit_simple_custom_gp(Xs, Ys, **kwargs):
    model = SimpleCustomGP(Xs[0], Ys[0])
    mll_nei = ExactMarginalLogLikelihood(model.likelihood, model)
    #mll_nei, model_nei = initialize_model(train_x_nei, train_obj_nei, train_con_nei)
    fit_gpytorch_model(mll_nei)
    return model


#4 define optimization problem
name = "branin"
sampled_input = {}


def branin(X):
    x1, x2 = X["x1"], X["x2"]

    y = (x2 - 5.1 / (4 * np.pi ** 2) * x1 ** 2 + 5 / np.pi * x1 - 6) ** 2
    y += 10 * (1 - 1 / (8 * np.pi)) * np.cos(x1) + 10
    # let's add some synthetic observation noise
    y += random.normalvariate(0, 0.1)
    y = -y
    # using string to safely compare if the inputs have been generated
    try:
        input_label = f"{format(x1, '.3f')}, {format(x2, '.3f')}"
        sampled_input[input_label] = (x1, x2)
    except:
        pass
    return {name: (y, 0.0)}


def branin2(X):
    x1, x2 = X[:, 0], X[:, 1]

    y = (x2 - 5.1 / (4 * np.pi ** 2) * x1 ** 2 + 5 / np.pi * x1 - 6) ** 2
    y += 10 * (1 - 1 / (8 * np.pi)) * np.cos(x1) + 10
    # let's add some synthetic observation noise
    y += random.normalvariate(0, 0.1)
    y = -y
    return y


#9 change
#from ax.models.torch.botorch_defaults import optimize_acqf
from botorch.acquisition.monte_carlo import qNoisyExpectedImprovement

# bounds is a global variable
axes = np.array([intervals[i] / (bounds[i][1] - bounds[i][0]) for i in range(len(bounds))])
valid_normalized_samples = np.array([np.arange(0, 1 + 1e-6, axes[i]) for i in range(len(bounds))])
constraints = []
#bounds_for_opt = torch.tensor([[bounds[i][0] for i in range(len(bounds))], [bounds[i][1] for i in range(len(bounds))]], device=device, dtype=dtype)


def rounder(values):
    def f(x):
        idx = np.argmin(np.abs(values - x))
        return values[idx]

    return np.frompyfunc(f, 1, 1)


def candidate_existing(acq_cand):
    acq_cand_to_gridpoint = [rounder(valid_normalized_samples[i])(acq_cand[i]) for i in range(len(bounds))]
    acq_cand_to_funcbounds = [acq_cand_to_gridpoint[i] * (bounds[i][1] - bounds[i][0]) + bounds[i][0] for i in
                              range(len(bounds))]

    acq_cand_to_funcbounds_label = f"{format(acq_cand_to_funcbounds[0], '.3f')}, {format(acq_cand_to_funcbounds[1], '.3f')}"
    return acq_cand_to_funcbounds_label in sampled_input.keys()


def outcome_constraint(X, cancel_point):
    Con = 1.1 - (((X[:, 0] - cancel_point[0]) / (axes[0] / 2)) ** 6 + (
            (X[:, 1] - cancel_point[1]) / (axes[1] / 2)) ** 6)
    return Con


# Define a construct to extract the objective and constraint from the GP
from botorch.acquisition.objective import ConstrainedMCObjective

def obj_callable(Z):
    return Z[..., 0]

def constraint_callable(Z):
    return Z[..., 1]

# define a feasibility-weighted objective for optimization
constrained_obj = ConstrainedMCObjective(
    objective=obj_callable,
    constraints=[constraint_callable],
)


# main change
# uses qNEI instead of UCB
# uses botorch-algorithm to evaluate next sample - no selfmade evaluation of acquisition function
def multistart_minimize2(model, constraints, acq_cand_to_gridpoint, n_iter):
    from botorch.models import FixedNoiseGP, ModelListGP
    from gpytorch.mlls.sum_marginal_log_likelihood import SumMarginalLogLikelihood
    from botorch.acquisition.objective import ConstrainedMCObjective
    from botorch import fit_gpytorch_model
    from botorch.acquisition.monte_carlo import qExpectedImprovement, qNoisyExpectedImprovement
    from botorch.sampling.samplers import SobolQMCNormalSampler
    from botorch.exceptions import BadInitialCandidatesWarning

    #extracts training points out of sampled_input
    i = 0
    for _, x in sampled_input.items():
        adn = torch.tensor([[x[0], x[1]]], device=device, dtype=dtype)
        if i == 0:
            train_x = adn
        else:
            train_x = torch.cat([train_x, adn])
        i += 1

    #print(train_x)
    NOISE_SE = 0.1
    train_yvar = torch.tensor(NOISE_SE ** 2, device=device, dtype=dtype)
    exact_obj = branin2(train_x).unsqueeze(-1)
    exact_con = outcome_constraint(train_x, acq_cand_to_gridpoint).unsqueeze(-1)
    train_obj = exact_obj + NOISE_SE * torch.randn_like(exact_obj)
    train_con = exact_con + NOISE_SE * torch.randn_like(exact_con)
    #print(train_obj)
    print(train_con)
    model_obj = FixedNoiseGP(train_x, train_obj, train_yvar.expand_as(train_obj)).to(train_x)
    model_con = FixedNoiseGP(train_x, train_con, train_yvar.expand_as(train_con)).to(train_x)
    model_nei = ModelListGP(model_obj, model_con)
    mll_nei = SumMarginalLogLikelihood(model_nei.likelihood, model_nei)
    fit_gpytorch_model(mll_nei)

    qNEI = qNoisyExpectedImprovement(
        model=model_nei,
        X_baseline=train_x,
        objective=constrained_obj
    )

    n = 10
    num_res = 1
    q = n_iter
    x_array, acq_array = [], []
    for i in range(1, n + 1):
        # q determines the number of next samples from given training points
        # q starts with 1 and will be increased if necessary
        x_val = optimize_acqf_and_get_observation_inner(qNEI, q, num_res)
        for a in range(0, q):
            new_x = [x_val[a][0].item(), x_val[a][1].item()]
            #new_acq = [acq_val.item()]

            # checks if all constraints are satisfied - uses trial and error approach
            # acquisition function is not influenced by constraints
            print(constraints)
            con_satis = 1
            for o in range(len(constraints)):
                con = ((new_x[0] - constraints[o][0]) / (axes[0] / 2)) ** 6 + (
                            (new_x[1] - constraints[o][1]) / (axes[1] / 2)) ** 6 - 1.1
                if con < 0:
                    con_satis = 0
            if con_satis == 1:
                x_array.append(new_x)
                #acq_array.append(new_acq)

    # gives list of possible next sample points
    res = np.array(x_array)
    #res = res.squeeze()
    return res


def optimize_acqf_and_get_observation_inner(acq_func, q, num_res):
    from botorch.optim import optimize_acqf
    # optimize
    bounds_for_opt = torch.tensor([[0.0, 0.0], [1.0, 1.0]], device=device, dtype=dtype)
    candidates, _ = optimize_acqf(
        acq_function=acq_func,
        bounds=bounds_for_opt,
        q=q,
        num_restarts=num_res,
        raw_samples=128,  # used for intialization heuristic
    )
    # observe new values
    new_x = candidates.detach()
    #acq_val = acq_value_list.detach()
    return new_x


def closest_val(res_gridpoint, acq_cand):
    x1, x2 = res_gridpoint[:, 0], res_gridpoint[:, 1]
    p1, p2 = acq_cand[0], acq_cand[1]
    y = (x1 - p1) ** 2 + (x2 - p2) ** 2
    return y


# main algorithm
def get_constraint_optimized_cand(model, acq_cand):
    if not candidate_existing(acq_cand.squeeze().numpy()):
        return acq_cand

    existing_inputs = np.array([value for _, value in sampled_input.items()])
    normalized_existing_inputs = np.zeros_like(existing_inputs)
    for i in range(len(bounds)):
        normalized = (existing_inputs[:, i] - bounds[i][0]) / (bounds[i][1] - bounds[i][0])
        normalized_existing_inputs[:, i] = rounder(valid_normalized_samples[i])(normalized)

    # acq_cand is an value that has not been rounded to grid point
    acq_cand_to_gridpoint = [rounder(valid_normalized_samples[i])(acq_cand.numpy()[0][i]) for i in range(len(bounds))]
    #print(acq_cand_to_gridpoint)
    constraints.append(acq_cand_to_gridpoint)

    n_iter = 0
    regenerate = False

    while n_iter < 5:
        #         if acq_x_int in raw_inputs.astype(int).tolist():
        regenerate = True
        n_iter += 1
        print(f"--- Constrained minimizing started! iteration {n_iter} ---")

        if len(bounds) == 2:
            res = multistart_minimize2(model, constraints, acq_cand_to_gridpoint, n_iter)
            #print(res)
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


# a modification of the original 'scipy_optimizer' function.
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
    raw_samples: int = kwargs.get("num_raw_samples", 50 * num_restarts)

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



#5 build experiment and return
def build_exp(search_space, seed, minimize):
    exp = SimpleExperiment(
        name=name,
        search_space=search_space,
        evaluation_function=branin, # should be the same function name you defined in last step
        objective_name=name,
        minimize=minimize,
    )
    sobol = get_sobol(exp.search_space, seed=seed)
    exp.new_batch_trial(generator_run=sobol.gen(initial_samples))
    return exp


#6
def optimize(search_space, seed, minimize):
    exp = build_exp(search_space, seed, minimize)
    # buiding the initial GP
    model = get_botorch(
        experiment=exp,
        data=exp.eval(),
        search_space=exp.search_space,
        model_constructor=get_and_fit_simple_custom_gp,
    )
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
    print("Done!")
    return model


#7 optimizing Branin function
from ax import ChoiceParameter
import warnings
warnings.filterwarnings("ignore")

parameter_space = SearchSpace(
    parameters=[
        ChoiceParameter(
            name="x1",
            parameter_type=ParameterType.FLOAT,
            values=torch.arange(bounds[0][0], bounds[0][1]+1e-6, intervals[0]),
            is_ordered=True,
            sort_values=True
        ),
        ChoiceParameter(
            name="x2",
            parameter_type=ParameterType.FLOAT,
            values=torch.arange(bounds[1][0], bounds[1][1]+1e-6, intervals[1]),
            is_ordered=True,
            sort_values=True
        )
    ]
)


#8 Activation
model = optimize(
    search_space=parameter_space,
    seed=random_seed,
    minimize=False
)


#9 plot
import matplotlib.pyplot as plt
from ax.models.torch.utils import predict_from_model
import torch
import gpytorch

# GP takes normalized input between (0, 1)
n = 100
x1= torch.linspace(0, 1, n).reshape((-1, 1))
x2 = torch.linspace(0, 1, n).reshape((-1, 1))
x1_x2 = torch.cat((x1, x2), 1)
x1_x2 = gpytorch.utils.grid.create_data_from_grid(x1_x2)

pred, var = predict_from_model(model.model.model, x1_x2)
pred = pred.numpy().reshape(n, n)

# We plot GP using real range
x1 = torch.linspace(-5, 10, n).reshape((-1, 1))
x2 = torch.linspace(0, 15, n).reshape((-1, 1))
x1, x2 = np.meshgrid(x1, x2)

plt.figure(figsize=(8, 4), dpi=100)
# plot true branin
plt.subplot(1, 2, 1)
plt.contour(x1, x2, branin({"x1": x1, "x2": x2})[name][0], linewidths=0.6, levels=50)
plt.title("Branin")
# plot GP
# black crosses are initial samples, blues are opt samples and red triangles are real minimums
plt.subplot(1, 2, 2)
plt.contour(x1, x2, pred, linewidths=0.6, levels=50)
i = 0
for _, x in sampled_input.items():
    color = "black" if i < initial_samples else "blue"
    plt.scatter(x[0], x[1], c=color, marker="x")
    i += 1
# plot the analytical solutions
plt.scatter([-np.pi, np.pi, 9.42478], [12.275, 2.275, 2.475], c="red", marker="^", s=8)
plt.scatter(cancel_point[0], cancel_point[1], c="red", marker="x")
plt.title("GP")

plt.show()