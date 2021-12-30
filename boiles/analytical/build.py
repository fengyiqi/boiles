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
from mytools.models.factory import get_botorch
from mytools.analytical.functions import TestFunction2D
from mytools.utils import *
from ax.models.torch.utils import (
    _to_inequality_constraints,
    predict_from_model,
)
from mytools.analytical.functions import solutions

import warnings
from gpytorch.utils.warnings import *
warnings.filterwarnings('ignore', category=UserWarning)
warnings.filterwarnings('ignore', category=NumericalWarning)
warnings.filterwarnings('ignore', category=GPInputWarning)
warnings.filterwarnings('ignore', category=OldVersionWarning)

display = False


class SimpleCustomGP(ExactGP, GPyTorchModel):
    _num_outputs = 1  # to inform GPyTorchModel API

    def __init__(self, train_X, train_Y):
        # squeeze output dim before passing train_Y to ExactGP
        super().__init__(train_X, train_Y.squeeze(-1), GaussianLikelihood())
        # super().__init__(train_X, train_Y.squeeze(-1), FixedNoiseGaussianLikelihood(noise=torch.tensor(np.ones(train_X.shape[0]).reshape(-1, 1)*1e-6)))
        self.mean_module = ConstantMean()
        self.covar_module = ScaleKernel(
            base_kernel=RBFKernel(ard_num_dims=train_X.shape[-1]),
        )
        self.to(train_X)  # make sure we're on the right device/dtype

    def forward(self, x):
        mean_x = self.mean_module(x)
        covar_x = self.covar_module(x)
        return MultivariateNormal(mean_x, covar_x)


def _get_and_fit_simple_custom_gp(Xs, Ys, **kwargs):
    model = SimpleCustomGP(Xs[0], Ys[0])
    mll = ExactMarginalLogLikelihood(model.likelihood, model)
    fit_gpytorch_model(mll)
    return model


def build_exp(search_space, test_function: TestFunction2D, initial_samples, seed=None, minimize=True):
    name = test_function.name
    exp = SimpleExperiment(
        name=name,
        search_space=search_space,
        evaluation_function=test_function.function,
        objective_name=name,
        minimize=minimize,
    )
    sobol = get_sobol(exp.search_space, seed=seed)
    exp.new_batch_trial(generator_run=sobol.gen(initial_samples))
    return exp


def optimize(search_space, test_function: TestFunction2D, initial_samples, opt_iteration, seed=None, minimize=True,
             count=None):

    exp = build_exp(search_space, test_function, initial_samples, seed, minimize)

    for i in range(opt_iteration):
        model = get_botorch(
            experiment=exp,
            data=exp.eval(),
            search_space=exp.search_space,
            model_constructor=_get_and_fit_simple_custom_gp,
        )

        generator_run = model.gen(1)
        #     print(generator_run.arms[0])
        #     generator_run.arms[0]
        #     print(generator_run.arms[0])
        batch = exp.new_trial(generator_run=generator_run)
        if display:
            log(f"Running optimization batch {i + 1}/{opt_iteration}...")
        if count is not None:
            OC.discrete["counter"] += 1

    log("Done!")

    return model


def plot_results(model, levels):

    def unit(x, bounds):
        return (x - bounds[0]) / (bounds[1] - bounds[0])

    x1_bounds, x2_bounds = TC.opt_bounds[0], TC.opt_bounds[1]

    bounds = [x1_bounds, x2_bounds]

    data = read_from_csv(f"{TC.function}.csv")

    data = np.array(data, dtype=float)
    inputs = data[:, :2]
    inputs[:, 0] = unit(inputs[:, 0], bounds[0])  # + random.random() * 1e-8
    inputs[:, 1] = unit(inputs[:, 1], bounds[1])  # + random.random() * 1e-8

    fig, ax = plt.subplots(figsize=(6, 5), dpi=150)
    x1, x2, pred = get_gpdata(model, n=100)

    label_fontsize = 18
    tick_fontsize = 16

    con = ax.contourf(x1, x2, pred, levels=levels)
    ax.scatter(inputs[:, 0], inputs[:, 1], s=40, marker="x", c='white', alpha=0.3)

    minima = solutions[TC.function].copy()
    minima[:, 0] = unit(minima[:, 0], TC.func_bounds[0])
    minima[:, 1] = unit(minima[:, 1], TC.func_bounds[1])
    ax.scatter(minima[:, 0], minima[:, 1], s=40, marker="^", c='orange', alpha=0.3)

    # acq_x1_float = cast_through_int(acq_cand[0, 0], bounds[0])
    # acq_x2_float = cast_through_int(acq_cand[0, 1], bounds[1])

    # ax.scatter(acq_x1_float, acq_x2_float, s=20, marker="x", c='red')

    ax.set_title(f"{TC.function.upper()}")
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.set_xlabel(r'$x1$', fontsize=label_fontsize)
    ax.set_ylabel(r'$x2$', fontsize=label_fontsize)
    ax.tick_params(labelsize=tick_fontsize)

    cbar = fig.colorbar(con, ax=ax)
    cbar.ax.tick_params(labelsize=tick_fontsize)
    cbar.formatter.set_powerlimits((-2, 2))
    cbar.update_ticks()
    fig.tight_layout()
    # plt.savefig("branin_with_tree.jpg")
    # plt.show()

import plotly.graph_objects as go

def plot_results3d(model, n=20):
    x = torch.linspace(0, 1, n).reshape((-1, 1))
    test_x = torch.cat((x, x, x), 1)
    test_x = gpytorch.utils.grid.create_data_from_grid(test_x)

    pred, var = predict_from_model(model, test_x)
    pred = pred.numpy().reshape((n, n, n))
    X, Y, Z = np.meshgrid(x, x, x)

    fig = go.Figure(data=go.Volume(
        x=X.flatten(),
        y=Y.flatten(),
        z=Z.flatten(),
        value=pred.flatten(),
        isomin=0.1,
        isomax=0.8,
        opacity=0.1,  # needs to be small to see through all surfaces
        surface_count=17,  # needs to be a large number for good volume rendering
        slices_z=dict(show=True, locations=[0.5]),
    ))
    fig.update_layout(
        autosize=False,
        # yaxis_range=[0, 1],
        # xaxis_range=[0, 1],
        # scene=dict(
        #     xaxis=dict(range=[0, 1], autorange=False),
        #     yaxis=dict(range=[0, 1], autorange=False),
        #     # zaxis=dict(range=[-1, 6], ),
        # ),
        width=800,
        height=800,
        # margin=dict(r=20, l=10, b=10, t=10),
        showlegend=False,
        # title=title,
        # xaxis_title="x1",
        # yaxis_title="x2"
    )
    # fig.update_yaxes(rangemode="tozero")
    fig.show()


def get_gpdata(model, n=100):

    x1_test = torch.linspace(0, 1, n).reshape((-1, 1))
    x2_test = torch.linspace(0, 1, n).reshape((-1, 1))

    x1_x2_test = torch.cat((x1_test, x2_test), 1)
    x1_x2_test = gpytorch.utils.grid.create_data_from_grid(x1_x2_test)

    pred, var = predict_from_model(model, x1_x2_test)
    pred = pred.numpy().reshape((n, n))
    x1, x2 = np.meshgrid(x1_test, x2_test)
    return x1, x2, pred