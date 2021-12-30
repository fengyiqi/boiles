#!/usr/bin/env python3

from .base import ObjectiveFunction, smoothness_indicator
from more_itertools import chunked
import more_itertools as mit
import matplotlib.pyplot as plt
from ..config.opt_problems import *
import h5py
import numpy as np
from ..test_cases import *

si_threshold = ShuBase200.si_threshold


class ShuOsher(ObjectiveFunction):

    def __init__(self,
                 results_folder: str,
                 result_filename: str = 'data_1.800*.h5',
                 git: bool = False,
                 plot: bool = True
                 ):

        super(ShuOsher, self).__init__(results_folder, result_filename, git=git)
        self.plot = plot
        self.plot_savepath = results_folder
        self.disper_name = OP.test_cases[0].name
        self.shock_name = OP.test_cases[1].name
        if self.result_exit:
            self.result = self.get_1d_h5data(self.result_path, git)
            self.solution_filename = ShuBase200.ref_data
            self.reference_raw = self.get_1d_h5data(self.solution_filename, git=False)
            self.reference = self.get_fvm_reference()

    @staticmethod
    def get_1d_h5data(file, git=False):
        with h5py.File(file, "r") as data:
            if git:
                cell_vertices = data["domain"]["cell_vertices"][:, :]
                vertex_coordinates = data["domain"]["vertex_coordinates"][:, :]
                density = data["simulation"]["density"][:]
                pressure = data["simulation"]["pressure"][:]
                velocity = data["simulation"]["velocityX"][:]
            else:
                cell_vertices = data["mesh_topology"]["cell_vertex_IDs"][:, :]
                vertex_coordinates = data["mesh_topology"]["cell_vertex_coordinates"][:, :]
                density = data["cell_data"]["density"][:, 0, 0]
                pressure = data["cell_data"]["pressure"][:, 0, 0]
                velocity = data["cell_data"]["velocity"][:, 0, 0]

        ordered_vertex_coordinates = vertex_coordinates[cell_vertices]
        cell_centers = np.mean(ordered_vertex_coordinates, axis=1)
        longest_axis = np.argmax(np.argmax(cell_centers, axis=0))
        x_cell_center = cell_centers[:, longest_axis]
        min_cell_coordinates = np.min(ordered_vertex_coordinates, axis=1)
        max_cell_coordinates = np.max(ordered_vertex_coordinates, axis=1)
        delta_xyz = max_cell_coordinates - min_cell_coordinates
        # volume = np.prod( delta_xyz, axis = 1 )

        e_i = pressure / (density * (1.4 - 1))
        e_kin = 0.5 * velocity ** 2
        energy = density * (e_i + e_kin)
        entropy = np.log(pressure / (density ** 1.4))
        enthalpy = (energy + pressure) / density
        # eva = (energy + pressure) * velocity

        data_dict = {'x_cell_center': x_cell_center,
                     'density': density,
                     'pressure': pressure,
                     'velocity': velocity,
                     'internal_energy': e_i,
                     'kinetic_energy': e_kin,
                     'total_energy': energy,
                     'entropy': entropy,
                     'enthalpy': enthalpy,
                     }

        return data_dict

    def get_fvm_reference(self):
        r"""
        Analytical solution of Shu-Osher problem doesn't exist, a grid-convergent solution with 12800
        cells along x-direction is used as the reference data. To make it comparable with under-resolved
        simulation, a fvm average is needed.
        :return: fvm averaged reference
        """
        interval = len(self.reference_raw['x_cell_center']) / len(self.result['x_cell_center'])
        fvm_reference = {}
        for key in self.reference_raw.keys():
            fvm_reference[key] = np.array([sum(x) / len(x) for x in chunked(self.reference_raw[key], int(interval))])
        return fvm_reference

    def _shock_index(self, value=None, plot=None, threshold=si_threshold):
        r"""
        Get the index of shock.
        :param value: value list that is used in smoothness indicator
        :param threshold: a threshold below which the index is shock index
        :param plot: if plot a figure and save in case folder
        :return: index group, list[list[], list[], ...]
        """
        if plot is None:
            plot = self.plot
        if value is None:
            value = self.reference['density']
        plot_filename = self.plot_savepath + '/shu_si.jpg'
        total_index = smoothness_indicator(x=list(self.reference['x_cell_center']),
                                           value=value,
                                           threshold=threshold,
                                           plot=plot,
                                           plot_savepath=plot_filename)
        index_group = [list(group) for group in mit.consecutive_groups(total_index)]
        return index_group

    def _get_pointwise_gradients(self, key):
        x_center = self.result['x_cell_center']
        dx = x_center[1] - x_center[0]
        result = self.result[key]
        reference = self.reference[key]

        pointwise_grads_ref = abs(reference[:-1] - reference[1:]) / dx
        pointwise_grads_sim = abs(result[:-1] - result[1:]) / dx
        return pointwise_grads_ref, pointwise_grads_sim

    def _shock_error(self, key, threshold=si_threshold):
        r"""
        Calculate the sum of inverse gradient at shock.
        :return: sum of inverse gradient at every shock
        """
        _, grad_sim = self._get_pointwise_gradients(key)
        shock_index_group = self._shock_index(self.reference[key], threshold=threshold)
        shock_grad = 0
        for index in shock_index_group:
            for i in index[:-1]:
                shock_grad += abs(grad_sim[i])
        objective_shock = 1 / shock_grad

        return objective_shock

    def objective_shock(self, key='density', threshold=si_threshold):
        r"""
        Return a clamped error
        :return: error and simulation state.
        """
        upper_bound = OP.test_cases[1].highest_error_from_initial
        if self.result_exit:
            real_error = self._shock_error(key, threshold=threshold)
            clipped_error = real_error if real_error <= upper_bound else upper_bound
            return clipped_error, real_error, 'completed'
        else:
            return upper_bound, 10000, 'divergent'

    def _disper_error(self, key, l2norm=True, ord=2, threshold=si_threshold):
        r"""
        Compute gradient-difference error in smooth region. At first all the gradients and respective
        difference is computed. Then the value of shock index will be removed from the array
        :param l2norm: if output l2 norm or average of all the difference
        :return: average or l2-norm of all the difference
        """
        grad_ref, grad_sim = self._get_pointwise_gradients(key)
        error_list_total = abs(grad_ref - grad_sim)
        shock_index_group = self._shock_index(self.reference[key], threshold=threshold)
        remove_index = []
        for index in shock_index_group:
            for i in index[:-1]:
                remove_index.append(i)
        error_list_total = np.delete(error_list_total, remove_index)
        error = np.linalg.norm(error_list_total, ord=ord) if l2norm else np.average(error_list_total)
        return error

    def _valua_error(self, key, l2norm=True, ord=2, threshold=si_threshold):
        shock_index_group = self._shock_index(self.reference[key], threshold=threshold)
        remove_index = []
        for index in shock_index_group:
            for i in index[:-1]:
                remove_index.append(i)
        error_list_total = np.delete(abs(self.reference[key] - self.result[key]), remove_index)
        error = np.linalg.norm(error_list_total, ord=ord) if l2norm else np.average(error_list_total)
        return error

    def objective_disper(self, key='density', ord=2, threshold=si_threshold, gradient_diff=True):
        r"""
        Return a clamped error
        :return: error and simulation state.
        """
        upper_bound = OP.test_cases[0].highest_error_from_initial
        if self.result_exit:
            if gradient_diff:
                real_error = self._disper_error(key, ord=ord, threshold=threshold)
            else:
                real_error = self._valua_error(key, ord=ord, threshold=threshold)
            clipped_error = real_error if real_error <= upper_bound else upper_bound
            return clipped_error, real_error, 'completed'

        else:
            return upper_bound, 10000, 'divergent'
