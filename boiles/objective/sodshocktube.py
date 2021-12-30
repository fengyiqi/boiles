#!/usr/bin/env python3

from .base import ObjectiveFunction
from ..config.opt_problems import OP
from ..test_cases.sod_60.sod_disper_60 import SodDisper60
from ..test_cases.sod_60.sod_shock_60 import SodShock60
import h5py
import numpy as np

index_group = {
    60: {
        "density": [40, 41, 50, 51],
        "pressure": [50, 51],
        "x_velocity": [50, 51]
    }
}


class SodShockTube(ObjectiveFunction):

    def __init__(self,
                 results_folder: str,
                 result_filename: str = 'data_0.200*.h5',
                 git: bool = False,
                 plot: bool = True,
                 ):

        super(SodShockTube, self).__init__(results_folder, result_filename, git=git)
        self.plot = plot
        self.plot_savepath = results_folder
        if self.result_exit:
            self.result = self.get_shu_h5data(self.result_path, git)
            self.cells = self.result['density'].shape[0]
            self.dx = 1 / self.cells
            self.reference = self.get_sod_reference_solution()
            # self.get_all_gradients_difference()

    # @staticmethod
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
            'x_velocity': x_velocity
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

    def get_all_value_difference(self, key='density'):

        ref = np.delete(self.get_sod_reference_solution()[key], index_group[self.cells][key])
        results = np.delete(self.result[key], index_group[self.cells][key])

        diff = abs(results - ref)

        return np.linalg.norm(diff, ord=2)

    @staticmethod
    def get_shu_h5data(file, git=False):

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
                try:
                    effective_diss_rate = np.array(data["cell_data"]["effective_dissipation_rate"][:, 0, 0])
                    numerical_diss_rate = np.array(data["cell_data"]["numerical_dissipation_rate"][:, 0, 0])
                except:
                    pass
                cell_vertices = np.array(data["mesh_topology"]["cell_vertex_IDs"])
                vertex_coordinates = np.array(data["mesh_topology"]["cell_vertex_coordinates"])

        ordered_vertex_coordinates = vertex_coordinates[cell_vertices]
        coords = np.mean(ordered_vertex_coordinates, axis=1)
        first_trafo = coords[:, 0].argsort(kind='stable')

        cell_vertices = cell_vertices[first_trafo]
        vertex_coordinates = vertex_coordinates[first_trafo]
        density = density[first_trafo]
        pressure = pressure[first_trafo]
        velocity = velocity[first_trafo]
        effective_diss_rate = effective_diss_rate[first_trafo]
        numerical_diss_rate = numerical_diss_rate[first_trafo]

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
                     'x_velocity': velocity,
                     'internal_energy': e_i,
                     'kinetic_energy': e_kin,
                     'total_energy': energy,
                     'entropy': entropy,
                     'enthalpy': enthalpy,
                     'numerical_dissipation_rate': numerical_diss_rate,
                     'effective_dissipation_rate': effective_diss_rate,
                     }

        return data_dict

    def objective_disper(self, prop="density"):
        r"""
        Return a clamped error
        :return: error and simulation state.
        """

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
            props = ["density", "pressure", "x_velocity"]
        upper_bound = SodDisper60.highest_error_from_initial
        if self.result_exit:
            if OP.test_cases is not None and OP.test_cases[0] is SodDisper60:
                upper_bound = OP.test_cases[0].highest_error_from_initial
                errors = np.array([self.objective_disper(prop=prop)[0] for prop in props])
            else:
                errors = np.array([self.objective_disper(prop=prop) for prop in props])
            temp = 0
            for j, error in enumerate(errors):
                temp += SodDisper60.normalize(error, props[j])

            e = np.clip(temp, -np.inf, upper_bound)
            if OP.test_cases is None:
                return e
            else:
                return e, temp, 'completed',
        else:
            return upper_bound, 1000, 'divergent'

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
