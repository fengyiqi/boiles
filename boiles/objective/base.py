#!/usr/bin/env python3

import numpy as np
import matplotlib.pyplot as plt
import os
import h5py
from abc import abstractmethod
import glob


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

def do_get_data(h5file, state: str, dimension: int):
    vel_keys = ["velocity_x", "velocity_y", "velocity_z"]
    vel_dict = {}
    if state == "velocity":
        for i in range(dimension):
            vel_dict[vel_keys[i]] = h5file["cell_data"][state][:, i, 0]
        if dimension == 1:
            # for 1D, we only return an array of x velocity
            return vel_dict["velocity_x"]
        else:
            return vel_dict
    else:
        return h5file["cell_data"][state][:, 0, 0]


def try_get_data(file, output: str, dimension: int):
    with h5py.File(file, "r") as f:
        if output in f["cell_data"].keys():
            return do_get_data(f, output, dimension)
        else:
            return None


def get_coords_and_order(cell_vertices, vertex_coordinates, dimension):
    ordered_vertex_coordinates = vertex_coordinates[cell_vertices]
    coords = np.mean(ordered_vertex_coordinates, axis=1)
    x_order = coords[:, 0].argsort(kind="stable")
    coords = coords[x_order]
    order = x_order
    if dimension == 1:
        return coords, order
    y_order = coords[:, 1].argsort(kind="stable")
    coords = coords[y_order]
    order = x_order[y_order]
    if dimension == 2:
        return coords, order
    z_order = coords[:, 2].argsort(kind="stable")
    coords = coords[z_order]
    order = x_order[y_order[z_order]]
    return coords, order


class ObjectiveFunction(object):

    def __init__(self,
                 file: str,
                 ):
        self.file_list = glob.glob(file)
        assert not len(self.file_list) == 0, "No such .h5 file"
        assert not len(self.file_list) > 1, "Found multiple .h5 file"
        self.result_exit = True if len(self.file_list) ==1 else False
        self.file = self.file_list[0]


    @abstractmethod
    def get_results(self, file):
        r"""
            get all valid results from .h5 data
        """
        pass

    @abstractmethod
    def get_ordered_data(self, file, state: str, order):
        r"""
            order the data for 1D, 2D and 3D
        """
        pass
