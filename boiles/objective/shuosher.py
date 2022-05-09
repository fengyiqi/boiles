#!/usr/bin/env python3

from .base import smoothness_indicator
from more_itertools import chunked
import more_itertools as mit
import numpy as np
from ..test_cases import *
from .simulation1d import Simulation1D
import json
import importlib.resources

si_threshold = ShuBase200.si_threshold


def read_shuosher_ref():
    with importlib.resources.open_text("boiles", "shuosher_ref.json") as file:
        data = json.load(file)
    for key, value in data.items():
        data[key] = np.array(value)
    data['entropy'] = np.log(data["pressure"] / (data["density"] ** 1.4))
    return data


class ShuOsher(Simulation1D):

    def __init__(
            self,
            file: str
    ):

        super(ShuOsher, self).__init__(file=file)
        self.dimension = 1
        self.disper_name = "disper"
        self.shock_name = "shock"
        self.plot = False
        self.plot_savepath = "_"
        if self.result_exit:
            self.reference_raw = read_shuosher_ref()
            self.reference = self.get_fvm_reference()

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
        upper_bound = ShuShock200.highest_error_from_initial
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
        upper_bound = ShuDisper200.highest_error_from_initial
        if self.result_exit:
            if gradient_diff:
                real_error = self._disper_error(key, ord=ord, threshold=threshold)
            else:
                real_error = self._valua_error(key, ord=ord, threshold=threshold)
            clipped_error = real_error if real_error <= upper_bound else upper_bound
            return clipped_error, real_error, 'completed'

        else:
            return upper_bound, 10000, 'divergent'
