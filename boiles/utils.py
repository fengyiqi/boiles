#!/usr/bin/env python3

import csv
# from mytools.solvers.alpaca_builder import AlpacaBuilder
import json
from gpytorch.priors.torch_priors import GammaPrior, MultivariateNormalPrior
import torch
from gpytorch.mlls.marginal_log_likelihood import MarginalLogLikelihood
from .config.opt_config import *
import os
from .ax_utilities import get_datadic
import matplotlib.pyplot as plt
from matplotlib import cm
from ax.modelbridge import TorchModelBridge
import gpytorch
from .config import *
import numpy as np
# from .data.std_prior import *

def read_from_csv(file_name: str) -> np.array:
    """
    a helper function to read data from csv file
    :param file_name: csv file name
    :return: numpy array
    """
    with open(file_name, 'r') as csvfile:
        reader = csv.reader(csvfile)
        data_rows = np.array([row for row in reader])

    data = []
    for row in data_rows:
        if len(row) != 0:
            data.append(row)

    return np.array(data)


def write_to_csv(file_name: str, data: list):
    """
    a helper function to write data into csv file
    :param file_name: csv file name
    :param data: a n-d array that needs to write into csv file
    """
    with open(file_name, 'w') as csvfile:
        writer = csv.writer(csvfile)
        writer.writerows(data)


def append_to_csv(file_name: str, data: list):
    """
    a helper funtion to append data into csv file
    :param file_name: csv file name
    :param data: a n-d array that needs to append into csv file
    :return:
    """
    with open(file_name, 'a+') as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(data)


def read_lengthscales_prior(json_name: str, key: str, std_scale: float = None):
    r"""
    read lengthscales prior distribution data from json file
    :param json_name: json file name
    :param key: key word
    :return: MultivariateNormalPrior or GammaPrior
    """
    try:
        with open(json_name, 'r') as f:
            prior_lengthscale = json.load(f)
        lengthscale_0 = prior_lengthscale[key][0]
        lengthscale_1 = prior_lengthscale[key][1]

        log(f'Reading GP lengthscale prior for "{key}" as\n{lengthscale_0, lengthscale_1}')
        if std_scale is None:
            std_0 = (lengthscale_0 * lengthscale_std_scale[SC.Riemann][key])**2
            std_1 = (lengthscale_1 * lengthscale_std_scale[SC.Riemann][key])**2
        else:
            std_0 = (lengthscale_0 * std_scale) ** 2
            std_1 = (lengthscale_1 * std_scale) ** 2
        cov = np.diag([std_0, std_1])
        l_prior = MultivariateNormalPrior(torch.tensor([lengthscale_0, lengthscale_1]),
                                          covariance_matrix=torch.tensor(cov))
    except:
        log("No prior distribution, using GammaPrior.")
        l_prior = GammaPrior(3.0, 6.0)

    return l_prior


def save_lengthscales_prior(mll: MarginalLogLikelihood):
    r"""
    save lengthscales prior distribution into a json file
    :param mll: MarginalLogLilihood
    :return: None
    """
    data_dic = {}
    for i, case in enumerate(OP.test_cases):
        model_l = mll.model.models[i].covar_module.base_kernel.lengthscale.cpu().detach().numpy().squeeze().tolist()
        log(f'Lengthscales for "{case.name}":')
        log(f"{model_l}")
        if case.sensitivity_control == "prior_distribution":
            # model_l = mll.model.models[i].covar_module.base_kernel.lengthscale.cpu().detach().numpy().squeeze().tolist()
            log(f"Lengthscales prior distribution")
            data_dic[case.name] = model_l
        else:
            log(f"No lengthscales prior distribution, use general GammaPrior")
    if not os.path.exists('lengthscale_prior.json'):
        with open('lengthscale_prior.json', 'w') as f:
            json.dump(data_dic, f)
        log(f'Prior lengthscales:')
        for key, value in data_dic.items():
            log(f"{key}, {value}")



def plot_runtime_surface(name_list, ehvi_model, iteration):
    r"""
    Since there is no way to save the experiment, this function is used to generate GP surface online
    :param name_list: case name list
    :param ehvi_model: ehvi model
    :param iteration: number of iteration
    :return: None
    """
    for name in name_list:

        save_path = f'{OC.case_folder}/figures'
        if not os.path.exists(save_path):
            os.makedirs(save_path)

        data1 = get_datadic(model=ehvi_model, param_x="q", param_y="cq", metric_name=name)

        fig = plt.figure(figsize=(15, 15))
        ax = fig.gca(projection='3d')
        X, Y, Z = data1['mean']['x'], data1['mean']['y'], data1['mean']['z']
        X, Y = np.meshgrid(X, Y)

        surf = ax.plot_surface(X, Y, Z, rstride=1, cstride=1, linewidth=2, cmap=cm.viridis, alpha=1.0)
        fig.colorbar(surf, shrink=0.5, aspect=10)

        ax.view_init(20, -70)
        plt.xlabel('q', fontsize=18)
        plt.ylabel('cq', fontsize=18)
        ax.set_zlabel('', fontsize=18)
        plot_filename = os.path.join(save_path, f'{name}_{str(iteration)}_surface.png')
        plt.savefig(plot_filename)
        # plt.close()


def plot_runtime_contour(name_list, ehvi_model, iteration):
    r"""
    Since there is no way to save the experiment, this function is used to generate GP contour online
    :param name_list: case name list
    :param ehvi_model: ehvi model
    :param iteration: number of iteration
    :return: None
    """
    for name in name_list:

        save_path = f'{OC.case_folder}/figures'
        if not os.path.exists(save_path):
            os.makedirs(save_path)

        data = get_datadic(model=ehvi_model, param_x="q", param_y="cq", metric_name=name)
        json_filename = os.path.join(save_path, f'{name}_{str(iteration)}.json')
        with open(json_filename, 'w') as f:
            json.dump(data, f)

        X = np.array(data['mean']['x'])*OC.intervals['q']
        Y = np.array(data['mean']['y'])*OC.intervals['cq']
        Z = data['mean']['z']
        X, Y = np.meshgrid(X, Y)

        fig, ax = plt.subplots(dpi=150)
        ax_ = ax.contourf(X, Y, Z, levels=20, cmap=cm.viridis)
        fig.colorbar(ax_)
        ax.contour(X, Y, Z, levels=10, colors='k', alpha=1.0, linestyles='--', linewidths=1.0)

        ax.set_title('Mean', )
        ax.set_xlabel('q', )
        ax.set_ylabel('cq', )
        plot_filename = os.path.join(save_path, f'{name}_{str(iteration)}.png')
        fig.savefig(plot_filename)
        # plt.close()


def apply_exponential_weights(data):
    data = data.squeeze()
    num = len(data)
    x = np.linspace(0, 1, num)
    y = np.exp(x)
    weights = y / y.sum()
    log(weights)
    return np.dot(weights, data)


def apply_linear_weights(data):
    data = data.squeeze()
    num = len(data)
    x = np.linspace(0, 1, num)
    weights = x / x.sum()
    log(weights)
    return np.dot(weights, data)


def save_model_state(ehvi_model: TorchModelBridge, iteration):

    save_path = f'{OC.case_folder}/model_state'
    if not os.path.exists(save_path):
        os.makedirs(save_path)
    for model in ehvi_model.model.model.models:
        model_dict = model.state_dict()
        torch.save(model_dict, f'{save_path}/{model.name}_model_state_{iteration}.pth')

def save_single_model_state(ei_model: TorchModelBridge, iteration):

    save_path = f'{OC.case_folder}/model_state'
    if not os.path.exists(save_path):
        os.makedirs(save_path)

    model_dict = ei_model.model.model.state_dict()
    torch.save(model_dict, f'{save_path}/{ei_model.model.model.name}_model_state_{iteration}.pth')

# def initialize_alpaca(ic: int,  # a value that can be divided by 4
#                       dim: int,
#                       stencil: str = 'WENOCU6M1',
#                       cq: int = 1000,
#                       q: int = 4,
#                       min_time_step: str = "std::numeric_limits<double>::epsilon()"):
#     r"""
#     Initialize ALPACA, including set parameters, cmake ALPACA according to the given dimension
#     and make ALPACA
#     :param stencil:
#     :param cq: cq in WENO5-CU6-M1
#     :param q: q in WENO5-CU6-M1
#     :param ic: internal cells per block
#     :param dim: case dimension
#     :param min_time_step: minimum time step size.
#     :return: None
#     """
#     alpaca = AlpacaBuilder()
#     alpaca.set_reconstruction_stencil(stencil)
#     alpaca.set_limit_end_time("true")
#     alpaca.set_ic(ic)
#     alpaca.set_minimum_time_step_size(min_time_step)
#     if stencil == 'WENOCU6M1':
#         alpaca.set_m1_scheme(cq, q)
#
#     alpaca.cmake_alpaca(dimension=dim)
#     alpaca.compile_alpaca()
#
#     return alpaca


def unit_x(value):
    cq_raw = value[:, 0]
    q_raw = value[:, 1]
    cq = cq_raw / OC.intervals['cq']
    q = q_raw / OC.intervals['q']
    cq = (cq - OC.bounds[0][0]) / (OC.bounds[0][1] - OC.bounds[0][0])
    q = (q - OC.bounds[1][0]) / (OC.bounds[1][1] - OC.bounds[1][0])
    return np.vstack((cq, q)).transpose()


def de_unit_x(value):
    cq_unit = value[:, 0]
    q_unit = value[:, 1]
    cq_raw = cq_unit * (OC.bounds[0][1] - OC.bounds[0][0]) + OC.bounds[0][0]
    q_raw = q_unit * (OC.bounds[1][1] - OC.bounds[1][0]) + OC.bounds[1][0]
    cq_raw *= OC.intervals['cq']
    q_raw *= OC.intervals['q']
    return np.vstack((cq_raw, q_raw)).transpose()


def de_unit_single_x(value, key: str):
    if key == "cq":
        x = value * (OC.bounds[0][1] - OC.bounds[0][0]) + OC.bounds[0][0]
        x *= OC.intervals['cq']
    if key == "q":
        x = value * (OC.bounds[1][1] - OC.bounds[1][0]) + OC.bounds[1][0]
        x *= OC.intervals['q']
    return x.reshape(-1, 1)


def standaardize_y(value):
    mean = value.mean()
    std = value.std()
    return (value - mean) / std


def de_standaardize_y(value, ref):
    mean = np.array(ref.mean())
    std = np.array(ref.std())
    return value * std + mean


def plot_contour(model,
                 title=None,
                 raw_data=False,
                 ref=None,
                 levels=30,
                 label_fontsize=10,
                 tick_fontsize=10,
                 save=True,
                 save_name="gp.jpg"):
    n = 100
    cq_test = torch.linspace(0, 1, n).reshape((-1, 1))
    q_test = torch.linspace(0, 1, n).reshape((-1, 1))
    cq_q_test = torch.cat((cq_test, q_test), 1)
    cq_q_test = gpytorch.utils.grid.create_data_from_grid(cq_q_test)

    model.eval()
    with torch.no_grad():
        # compute posterior
        posterior_test = model.posterior(cq_q_test)
        # Get upper and lower confidence bounds (2 standard deviations from the mean)
        lower, upper = posterior_test.mvn.confidence_region()

    if raw_data:
        q_test = de_unit_single_x(q_test, 'q')
        cq_test = de_unit_single_x(cq_test, 'cq')
    x1, x2 = np.meshgrid(q_test, cq_test)
    pred_test = posterior_test.mean.cpu().numpy()
    if raw_data:
        pred_test = de_standaardize_y(pred_test, ref=ref)
    pred_test = pred_test.reshape((n, n)).transpose()

    fig, ax = plt.subplots(figsize=(6, 5), dpi=100)
    con = ax.contourf(x1, x2, pred_test, levels=levels)
    ax.set_xlabel(r'$q$', fontsize=label_fontsize)
    ax.set_ylabel(r'$C_q$', fontsize=label_fontsize)
    ax.tick_params(labelsize=tick_fontsize)
    if title is not None:
        ax.set_title(title)
    if raw_data:
        ax.yaxis.get_major_formatter().set_powerlimits((3, 3))
    cbar = fig.colorbar(con, ax=ax)
    #     cbar.set_label(colorbar_label, size=18)
    cbar.ax.tick_params(labelsize=tick_fontsize)
    cbar.formatter.set_powerlimits((-2, 2))

    cbar.update_ticks()
    fig.tight_layout()
    if save:
        plt.savefig(save_name)


def cross_validation(model, cq_q, accu, title=None, raw_data=False):

    model.eval()
    with torch.no_grad():
        # compute posterior
        posterior = model.posterior(cq_q)

        # Get upper and lower confidence bounds (2 standard deviations from the mean)
        lower, upper = posterior.mvn.confidence_region()

    pred = posterior.mean.cpu().numpy()
    if raw_data:
        ref = accu
        pred = de_standaardize_y(pred, ref=ref)
    diag = np.linspace(accu.min(), accu.max(), 50)

    fig, ax = plt.subplots(figsize=(5, 5), dpi=100)
    ax.scatter(accu, pred, c='none', edgecolor='blue')
    ax.set_xlabel('Solution')
    ax.set_ylabel('Prediction')

    if title is not None:
        ax.set_title(title)
    ax.grid()
    ax.plot(diag, diag, c='black', linestyle='--')


def log(string: str, print_to_terminal=True):
    with open("log.txt", "a+") as file:
        file.write(string)
        file.write("\n")
    if print_to_terminal:
        print(string)