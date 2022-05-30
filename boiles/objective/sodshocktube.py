#!/usr/bin/env python3

from ..config.opt_problems import OP
from ..test_cases.sod_60.sod_disper_60 import SodDisper60
from ..test_cases.sod_60.sod_shock_60 import SodShock60
from .simulation1d import Simulation1D
import numpy as np

index_group = {
    40: {
        "density": [26, 27, 33, 34],
        "pressure": [33, 34],
        "velocity": [33, 34]
    },
    60: {
        "density": [40, 41, 50, 51],
        "pressure": [50, 51],
        "velocity": [50, 51]
    }
}


class Sod(Simulation1D):

    def __init__(
            self,
            file: str
    ):
        super(Sod, self).__init__(file=file)
        if self.result_exit:
            self.cells = self.result['density'].shape[0]
            self.dx = 1 / self.cells
            self.reference = self.get_sod_reference_solution()

    def get_sod_reference_solution(self, cells=None) -> dict:
        if cells is None:
            cells = self.cells
        x = np.linspace(self.dx / 2, 1 - self.dx / 2, cells)
        pressure = np.hstack((np.ones(x[x < 0.263357].shape[0]),
                              (-0.704295 * x[(0.263357 <= x) & (x < 0.485946)] + 1.18548) ** 7,
                              np.ones(x[(0.485946 <= x) & (x < 0.850431)].shape[0]) * 0.303130,
                              np.ones(x[0.850431 <= x].shape[0]) * 0.1))
        density = np.hstack((np.ones(x[x < 0.263357].shape[0]),
                             (-0.704295 * x[(0.263357 <= x) & (x < 0.485946)] + 1.18548) ** 5,
                             np.ones(x[(0.485946 <= x) & (x < 0.685491)].shape[0]) * 0.426319,
                             np.ones(x[(0.685491 <= x) & (x < 0.850431)].shape[0]) * 0.265574,
                             np.ones(x[0.850431 <= x].shape[0]) * 0.125))
        x_velocity = np.hstack((np.zeros(x[x < 0.263357].shape[0]),
                                4.16667 * x[(0.263357 <= x) & (x < 0.485946)] - 1.09372,
                                np.ones(x[(0.485946 <= x) & (x < 0.850431)].shape[0]) * 0.927453,
                                np.zeros(x[0.850431 <= x].shape[0])))

        return {
            'x_cell_center': x,
            'pressure': pressure,
            'density': density,
            'velocity': x_velocity
        }

    def get_all_gradients_difference(self, key='density', ord=2):

        # ref = self.get_sod_reference_solution()[key]
        results = self.result[key]
        # ref_g = np.gradient(ref, self.dx)
        results_g = np.gradient(results, self.dx)
        # g_diff_all = abs(results_g - ref_g)
        shock_grad = 0
        # if self.cells == 100:
        #     g_diff = np.delete(g_diff_all, [68, 84])
        #     for i in [68, 84]:
        #         shock_grad += results_g[i]
        #     self.shock = 1 / shock_grad
        # elif self.cells == 80:
        #     g_diff = np.delete(g_diff_all, [54, 67])
        #     for i in [54, 67]:
        #         shock_grad += results_g[i]
        #     self.shock = 1 / shock_grad
        if self.cells == 60:
            # g_diff = np.delete(g_diff_all, index_group[self.cells][key])
            for i in index_group[self.cells][key]:
                shock_grad += abs(results_g[i])
            return 1 / shock_grad
        else:
            raise Exception(f"No configuration for {self.cells} grid")

        # self.disper = np.linalg.norm(g_diff, ord=ord)

    def get_all_value_difference(self, key='density', exclude_shocks=True):

        if exclude_shocks:
            ref = np.delete(self.get_sod_reference_solution(self.cells)[key], index_group[self.cells][key])
            results = np.delete(self.result[key], index_group[self.cells][key])
        else:
            ref = self.get_sod_reference_solution(self.cells)[key]
            results = self.result[key]
        diff = abs(results - ref)
        return np.linalg.norm(diff, ord=2)

    def objective_disper(self, prop="density"):

        if OP.test_cases is not None and OP.test_cases[0] is SodDisper60:
            upper_bound = OP.test_cases[0].highest_error_from_initial
        else:
            upper_bound = np.inf
        if self.result_exit:

            temp = self.get_all_value_difference(key=prop)
            error = temp.clip(-np.inf, upper_bound)
            if OP.test_cases is None:
                return error
            else:
                return error, temp, 'completed',
        else:
            return upper_bound, 1000, 'divergent'

    def objective_sum_disper(self, props=None):
        r"""
        Return a clamped error
        :return: error and simulation state.
        """
        if props is None:
            props = ["density", "pressure", "velocity"]
        upper_bound = SodDisper60.highest_error_from_initial
        if self.result_exit:
            errors = np.array([self.get_all_value_difference(key=prop) for prop in props])
            sum_error = 0
            for i, error in enumerate(errors):
                sum_error += SodDisper60.normalize(error, props[i])

            sum_error_clipped = np.clip(sum_error, -np.inf, upper_bound)
            if OP.test_cases is None:
                return sum_error_clipped
            else:
                return sum_error_clipped, sum_error, 'completed',
        else:
            return upper_bound, np.nan, 'divergent'

    def objective_shock(self, prop="density"):
        r"""
        Return a clamped error
        :return: error and simulation state.
        """

        if OP.test_cases is not None and OP.test_cases[1] is SodShock60:
            upper_bound = OP.test_cases[1].highest_error_from_initial
        else:
            upper_bound = np.inf
        if self.result_exit:

            temp = self.get_all_gradients_difference(prop)
            error = temp.clip(-np.inf, upper_bound)
            if OP.test_cases is None:
                return error
            else:
                return error, temp, 'completed'
        else:
            return upper_bound, 1000, 'divergent'
