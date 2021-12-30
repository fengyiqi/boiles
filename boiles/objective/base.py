#!/usr/bin/env python3

import numpy as np
from fnmatch import fnmatchcase as match
import matplotlib.pyplot as plt
import os
# from mytools.opt_config import *


def smoothness_indicator(x: list,
                         value: list,
                         threshold: float,
                         plot: bool = False,
                         plot_savepath: str = None) -> list:
    r""" WENO5 smoothness indicator
    @param plot: if plot the smoothness indicator
    @param x: x coordinates
    @param value: value of interest
    @param threshold: indicate jump level
    @param plot_savepath: the path where the plot should be saved
    @return: index list where there is a jump
    """
    f = []

    for i in range(len(x) - 2):
        # WENO5 smoothness indicator
        beta = 4 * value[i] ** 2 - 13 * value[i] * value[i + 1] + 13 * value[i + 1] ** 2 + 5 * value[i] * \
               value[i + 2] - 13 * value[i + 1] * value[i + 2] + 4 * value[i + 2] ** 2
        f.append(beta)

    f = np.array(f).reshape(-1, 1)
    index = np.where(f > threshold)
    if plot:
        x_f = np.array(x[1:-1]).reshape(-1, 1)
        fig, ax = plt.subplots(dpi=150)

        ax.plot(x_f, f, 'k-')
        ax.hlines(threshold, x_f[0], x_f[-1], colors="r", linestyles="dashed")
        ax.set_title('Smoothness indicator', fontsize=18)
        fig.savefig(plot_savepath)
        plt.close(fig)

    return index[0] + 1


def check_results_exist(folder: str,
                        file_name: str):
    r"""
    If the simulation is divergent, there isn't a result file.
    This function is used to check if the simulation is successful.
    :return: if exist, return full file_name and True, else return [] and False
    """
    name = os.listdir(folder)
    h5_name = [_name for _name in name if match(_name, file_name)]
    if h5_name:
        return h5_name[-1], True
    else:
        return [], False


def get_data_items(
        data_type: str,
        data_name: str,
        git: bool
):
    pass




class ObjectiveFunction(object):

    def __init__(self,
                 results_folder: str,
                 result_filename: str,
                 git: bool = False,
                 ):
        self.results_folder = results_folder
        # self.result_filename = result_filename

        self.result_filename, self.result_exit = check_results_exist(results_folder, result_filename)

        if self.result_exit:
            self.result_path = os.path.join(self.results_folder, self.result_filename)

        self.reference: dict



