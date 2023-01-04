#discrete inputs own
# script for discrete input space
# 3/3 scripts to make a comparison between new developed algorithm and tradtional algorithm
# performs BO with traditional algorithm by using same initial points as 05_mid_B
# plots a comparison

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

# Adapt parameters
random_seed = 107
initial_samples = 5
opt_iterations = 40
bounds = ((-5, 5), (-5, 5))
# the steps for each parameter
intervals = (0.25, 0.25)
noise_val = 0.001

levels_n = np.arange(-1, 4, 0.25).tolist()

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
        evaluation_function=branin, # should be the same function name you defined in last step
        objective_name=name,
        minimize=minimize,
    )
    sobol = get_sobol(exp.search_space, seed=seed)
    exp.new_batch_trial(generator_run=sobol.gen(initial_samples))
    return exp


#3 fit GP
def get_and_fit_simple_custom_gp(Xs, Ys, **kwargs):
    model = SimpleCustomGP(Xs[0], Ys[0])
    mll = ExactMarginalLogLikelihood(model.likelihood, model)
    fit_gpytorch_model(mll)
    return model

def get_and_fit_simple_custom_gp_over(Xs, Ys, **kwargs):
    model = SimpleCustomGP(Xs[0], Ys[0])
    state_dict = torch.load('runtime_data/model_state_init.pth')
    model.load_state_dict(state_dict)
    return model


#6
def optimize(search_space, seed, minimize):
    exp = build_exp(search_space, seed, minimize)
    # building the initial GP
    model = get_botorch(
        experiment=exp,
        data=exp.eval(),
        search_space=exp.search_space,
        model_constructor=get_and_fit_simple_custom_gp_over,
    )

    sampled_input = read_from_csv_list(f'runtime_data/initial_condition.csv')

    # start optimization loop
    for i in range(opt_iterations):
        generator_run = model.gen(1)
        exp.new_trial(generator_run=generator_run)
        model = get_botorch(
            experiment=exp,
            data=exp.eval(),
            search_space=exp.search_space,
            model_constructor=get_and_fit_simple_custom_gp
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
            #lower=bounds[0][0],
            #upper=bounds[0][1]+1e-6
        ),
        ChoiceParameter(
            name="x2",
            parameter_type=ParameterType.FLOAT,
            values=torch.arange(bounds[1][0], bounds[1][1]+1e-6, intervals[1]),
            is_ordered=True,
            sort_values=True
            #lower=bounds[1][0],
            #upper=bounds[1][1]+1e-6
        )
    ]
)


#8 Activation
model = optimize(
    search_space=parameter_space,
    seed=random_seed,
    minimize=False
)


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
#print(pred)
pred = pred.numpy().reshape(n, n)
#c = 60
#pred = c * (pred - 0.94)

def saveList(myList,filename):
    # the filename should mention the extension 'npy'
    np.save(filename, myList)

saveList(pred,'runtime_data/pred_Trad.npy')


def loadList(filename):
    # the filename should mention the extension 'npy'
    tempNumpyArray=np.load(filename)
    return tempNumpyArray.tolist()


# Comparison
sampled_input_init = read_from_csv_list(f'runtime_data/initial_condition.csv')
sampled_input_Trad = sampled_input
sampled_input_qNEI = read_from_csv_list(f'runtime_data/final_condition_pre.csv')
sampled_input_Real = read_from_csv_list(f'runtime_data/Real_pre.csv')
#print(sampled_input_init)
#print(sampled_input_Trad)
#print(sampled_input_qNEI)

#pred_qNEI = pred
#pred_UCB = pred

# We plot GP using real range
x1 = torch.linspace(bounds[0][0], bounds[0][1], n).reshape((-1, 1))
x2 = torch.linspace(bounds[1][0], bounds[1][1], n).reshape((-1, 1))
x1, x2 = np.meshgrid(x1, x2)

pred_branin = branin({"x1": x1, "x2": x2})[name][0]
pred_qNEI = loadList('runtime_data/pred_qNEI.npy')
pred_Trad = loadList('runtime_data/pred_Trad.npy')
pred_Real = loadList('runtime_data/pred_Real.npy')
pred_branin = -1 * pred_branin
pred_qNEI = np.negative(pred_qNEI)
pred_Trad = np.negative(pred_Trad)
pred_Real = np.negative(pred_Real)

pred_aus_qNEI = loadList('runtime_data/pred_aus_qNEI.npy')
pred_aus_Real = loadList('runtime_data/pred_aus_Real.npy')
pred_aus_qNEI = np.negative(pred_aus_qNEI)
pred_aus_Real = np.negative(pred_aus_Real)

plt.figure(figsize=(12, 4), dpi=100)
# plot true branin
#plt.subplot(1, 3, 1)
#plt.contour(x1, x2, pred_branin, linewidths=0.6, levels=50)
#plt.xlim([bounds[0][0], bounds[0][1]])
#plt.ylim([bounds[1][0], bounds[1][1]])
#plt.title("Branin")

# plot GP
# black crosses are initial samples, blues are opt samples and red triangles are real minimums
plt.subplot(1, 3, 1)
plt.contour(x1, x2, pred_Real, linewidths=0.6, levels=levels_n)
#lnlt_UCB = len(nltrainx_UCB)
i = 0
for _, x in sampled_input_Real.items():
    color = "black" if i < initial_samples else "blue"
    #for a in range(lnlt_UCB):
        #if i == nltrainx_UCB[a]:
            #color = "green"
    #plt.scatter(x[0], x[1], c=color, marker="x")
    i += 1
# plot the analytical solutions
#plt.scatter([-np.pi, np.pi, 9.42478], [12.275, 2.275, 2.475], c="red", marker="^", s=8)
plt.xlim([bounds[0][0], bounds[0][1]])
plt.ylim([bounds[1][0], bounds[1][1]])
plt.title("Himmelblau")


# black crosses are initial samples, blues are opt samples and red triangles are real minimums
plt.subplot(1, 3, 2)
plt.contour(x1, x2, pred_Trad, linewidths=0.6, levels=levels_n)
#lnlt_UCB = len(nltrainx_UCB)
i = 0
for _, x in sampled_input_Trad.items():
    color = "black" if i < initial_samples else "blue"
    #for a in range(lnlt_UCB):
        #if i == nltrainx_UCB[a]:
            #color = "green"
    plt.scatter(x[0], x[1], c=color, marker="x")
    i += 1
# plot the analytical solutions
#plt.scatter([-np.pi, np.pi, 9.42478], [12.275, 2.275, 2.475], c="red", marker="^", s=8)
plt.xlim([bounds[0][0], bounds[0][1]])
plt.ylim([bounds[1][0], bounds[1][1]])
plt.title("Traditional")

plt.subplot(1, 3, 3)
plt.contour(x1, x2, pred_qNEI, linewidths=0.6, levels=levels_n)
#lnlt=len(nltrainx)
i = 0
for _, x in sampled_input_qNEI.items():
    color = "black" if i < initial_samples else "blue"
    #for a in range(lnlt):
        #if i == nltrainx[a]:
            #color = "green"
    plt.scatter(x[0], x[1], c=color, marker="x")
    i += 1
# plot the analytical solutions
#plt.scatter([-np.pi, np.pi, 9.42478], [12.275, 2.275, 2.475], c="red", marker="^", s=8)
plt.xlim([bounds[0][0], bounds[0][1]])
plt.ylim([bounds[1][0], bounds[1][1]])
plt.title("Our BO algorithm")

plt.show()

#print(pred_aus_qNEI)
#print(pred_aus_Real)
error_pred_aus=pred_aus_qNEI - pred_aus_Real
#print(error_pred_aus)

from numpy import linalg as LA
error_norm=LA.norm(error_pred_aus)
print(error_norm)