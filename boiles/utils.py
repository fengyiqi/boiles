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


def save_model_state(ehvi_model: TorchModelBridge, iteration):

    save_path = f'{OC.case_folder}/model_state'
    if not os.path.exists(save_path):
        os.makedirs(save_path)
    for model in ehvi_model.model.model.models:
        model_dict = model.state_dict()
        torch.save(model_dict, f'{save_path}/{model.name}_model_state_{iteration}.pth')


def save_single_model_state(ei_model: TorchModelBridge, iteration, name):

    save_path = f'{OC.case_folder}/model_state'
    if not os.path.exists(save_path):
        os.makedirs(save_path)

    model_dict = ei_model.model.model.state_dict()
    torch.save(model_dict, f'{save_path}/{name}_model_state_{iteration}.pth')


def log(string: str, print_to_terminal=True):
    with open("log.txt", "a+") as file:
        file.write(string)
        file.write("\n")
    if print_to_terminal:
        print(string)