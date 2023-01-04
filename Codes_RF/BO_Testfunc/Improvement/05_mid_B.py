# script for discrete input space
# 2/3 scripts to make a comparison between new developed algorithm and tradtional algorithm
# performs BO with new developed algorithm
# uses qNoisyExpectedImprovement instead of UCB

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

os.system("rm -rf runtime_data/initial_condition.csv runtime_data/final_condition_pre.csv runtime_data/pred_qNEI.npy runtime_data/pred_UCB.npy runtime_data/model_state_final_pre.pth runtime_data/model_state_init.pth")

device = torch.device("cpu")
dtype = torch.double
SMOKE_TEST = os.environ.get("SMOKE_TEST")

# Adapt parameters
random_seed = 107
initial_samples = 5
opt_iterations = 40
bounds = ((-5, 5), (-5, 5))
# the steps for each parameter
intervals = (0.25, 0.25)
noise_val = 0.001


def read_from_csv_list(file_name: str) -> dict:
    import re
    """
    a helper function to read data from csv file
    :param file_name: csv file name
    :return: list
    """
    with open(file_name, 'r') as csvfile:
        reader = csv.reader(csvfile)
        data_list = str([row for row in reader])

    data_list_clean = str(data_list)[2:-2]
    data_list_clean_6 = data_list_clean.replace(")", ");")
    data_list_clean_7 = data_list_clean_6.replace(";,", ";")
    data_list_clean_8 = data_list_clean_7.replace(":", "=")
    data_list_clean_9 = data_list_clean_8.replace("{", "{; ")
    data_list_clean_10 = re.sub(r"[\[{}\]]", "", data_list_clean_9)
    data_list_clean_11 = data_list_clean_10.split(";")
    dictionary = {}
    for subString in data_list_clean_11:
        if len(subString) > 5:
            subS = subString.split("=")
            input_label = subS[0]
            input_label = str(input_label)[2:-1]
            ssubS = subS[1]
            ssubS = str(ssubS)[2:-1]
            ssubSN = ssubS.split(",")
            x1 = float(ssubSN[0])
            x2 = float(ssubSN[1])
            dictionary[input_label] = (x1, x2)

    return dictionary


#2 GP surrogate model
class SimpleCustomGP(ExactGP, GPyTorchModel):
    _num_outputs = 1  # to inform GPyTorchModel API

    def __init__(self, train_X, train_Y):
        # squeeze output dim before passing train_Y to ExactGP
        noise = torch.ones(train_X.shape[0]) * noise_val
        #super().__init__(train_X, train_Y.squeeze(-1), FixedNoiseGaussianLikelihood(noise))
        super().__init__(train_X, train_Y.squeeze(-1), GaussianLikelihood())
        self.mean_module = ConstantMean()
        self.covar_module = ScaleKernel(
            base_kernel = MaternKernel(nu=2.5, ard_num_dims=train_X.shape[-1]),
        )
        self.to(train_X)  # make sure we're on the right device/dtype

    def forward(self, x):
        mean_x = self.mean_module(x)
        covar_x = self.covar_module(x)
        return MultivariateNormal(mean_x, covar_x)


#4 define optimization problem
name = "branin"
sampled_input = {}

def branin(X):
    x1, x2 = X["x1"], X["x2"]
    # Himmelblau
    y = (x1 ** 2 + x2 - 11) ** 2 + (x1 + x2 ** 2 - 7) ** 2
    # let's add some synthetic observation noise
    #y += random.normalvariate(0, 0.001)
    y = -y
    # using string to safely compare if the inputs have been generated
    try:
        input_label = f"{format(x1, '.3f')}, {format(x2, '.3f')}"
        sampled_input[input_label] = (x1, x2)
    except:
        pass
    return {name: (y, 0.0)}


#5 build experiment and return
def build_exp(search_space, seed, minimize):
    exp = SimpleExperiment(
        name=name,
        search_space=search_space,
        evaluation_function=branin,
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
    acq_cand_to_funcbounds_label = f"{format(acq_cand_to_funcbounds[0], '.3f')}, {format(acq_cand_to_funcbounds[1], '.3f')}"
    #print(sampled_input)
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
            new_x = [x_val[a][0].item(), x_val[a][1].item()]
            new_acq = [acq_val.item()]
            # checks if all constraints are satisfied - uses trial and error approach
            # acquisition function is not influenced by constraints
            con_satis = 1
            for o in range(len(constraints)):
                con = ((new_x[0] - constraints[o][0]) / (axes[0] / 2)) ** 6 + (
                            (new_x[1] - constraints[o][1]) / (axes[1] / 2)) ** 6 - 1.1
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
    bounds_for_opt = torch.tensor([[0.0, 0.0], [1.0, 1.0]], device=device, dtype=dtype)
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
    x1, x2 = res_gridpoint[:, 0], res_gridpoint[:, 1]
    p1, p2 = acq_cand[0], acq_cand[1]
    y = -((x1 - p1) ** 2 + (x2 - p2) ** 2)
    return y

#6.6 main algorithm - get next candidate
def get_constraint_optimized_cand(model, acq_cand):
    if not candidate_existing(acq_cand.squeeze().numpy()):
        return acq_cand

    print(acq_cand)

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
        adn = torch.tensor([[x[0], x[1]]], device=device, dtype=dtype)
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

        if len(bounds) == 2:
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


def get_and_fit_simple_custom_gp_init(Xs, Ys, **kwargs):
    model = SimpleCustomGP(Xs[0], Ys[0])
    mll_nei = ExactMarginalLogLikelihood(model.likelihood, model)
    fit_gpytorch_model(mll_nei)
    torch.save(model.state_dict(), 'runtime_data/model_state_init.pth')
    return model

def get_and_fit_simple_custom_gp_over(Xs, Ys, **kwargs):
    model = SimpleCustomGP(Xs[0], Ys[0])
    state_dict = torch.load('runtime_data/model_state_init.pth')
    model.load_state_dict(state_dict)
    return model

#3 fit GP
def get_and_fit_simple_custom_gp(Xs, Ys, **kwargs):
    model = SimpleCustomGP(Xs[0], Ys[0])
    mll_nei = ExactMarginalLogLikelihood(model.likelihood, model)
    fit_gpytorch_model(mll_nei)

    model.eval()
    with torch.no_grad(), gpytorch.settings.fast_pred_var():
        f_preds = model(Xs[0])
        f_var = f_preds.variance
    print(f_var)

    torch.save(model.state_dict(), 'runtime_data/model_state_final_pre.pth')
    return model


#7 outer #optimization loop
def optimize(search_space, seed, minimize):
    exp = build_exp(search_space, seed, minimize)
    # building the initial GP
    model = get_botorch(
        experiment=exp,
        data=exp.eval(),
        search_space=exp.search_space,
        model_constructor=get_and_fit_simple_custom_gp_init,
    )

    os.system(f"cd runtime_data")
    append_to_csv(f'runtime_data/initial_condition.csv', [sampled_input])

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


#8 Define search space
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


#9 Activation
model = optimize(
    search_space=parameter_space,
    seed=random_seed,
    minimize=False
)

os.system(f"cd runtime_data")
append_to_csv(f'runtime_data/final_condition_pre.csv', [sampled_input])

#10 plot
import matplotlib.pyplot as plt
from ax.models.torch.utils import predict_from_model

# GP takes normalized input between (0, 1)
n = 100
x1 = torch.linspace(0, 1, n).reshape((-1, 1))
x2 = torch.linspace(0, 1, n).reshape((-1, 1))
x1_x2 = torch.cat((x1, x2), 1)
x1_x2 = gpytorch.utils.grid.create_data_from_grid(x1_x2)

pred, var = predict_from_model(model.model.model, x1_x2)
pred = pred.numpy().reshape(n, n)

def saveList(myList,filename):
    # the filename should mention the extension 'npy'
    np.save(filename, myList)

saveList(pred,'runtime_data/pred_qNEI.npy')

# We plot GP using real range
x1 = torch.linspace(-5, 5, n).reshape((-1, 1))
x2 = torch.linspace(-5, 5, n).reshape((-1, 1))
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
lnlt=len(nltrainx)
i = 0
for _, x in sampled_input.items():
    color = "black" if i < initial_samples else "blue"
    for a in range(lnlt):
        if i == nltrainx[a]:
            color = "blue"
    #plt.scatter(x[0], x[1], c=color, marker="x")
    i += 1
# plot the analytical solutions
#plt.scatter([-np.pi, np.pi, 9.42478], [12.275, 2.275, 2.475], c="red", marker="^", s=8)
plt.xlim([bounds[0][0], bounds[0][1]])
plt.ylim([bounds[1][0], bounds[1][1]])
plt.title("GP")
#plt.show()

#Auswertung
n = 10
x1 = torch.linspace(0, 1, n).reshape((-1, 1))
x2 = torch.linspace(0, 1, n).reshape((-1, 1))
x1_x2 = torch.cat((x1, x2), 1)
x1_x2 = gpytorch.utils.grid.create_data_from_grid(x1_x2)

pred_aus, var = predict_from_model(model.model.model, x1_x2)
pred_aus = pred_aus.numpy().reshape(n, n)

saveList(pred_aus,'runtime_data/pred_aus_qNEI.npy')

# We plot GP using real range
x1 = torch.linspace(-5, 5, n).reshape((-1, 1))
x2 = torch.linspace(-5, 5, n).reshape((-1, 1))
x1, x2 = np.meshgrid(x1, x2)

print(pred_aus)