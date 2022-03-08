#!/usr/bin/env python3

import matplotlib.pyplot as plt
import h5py
import sympy
from .base import ObjectiveFunction, try_get_data, get_coords_and_order
import numpy as np
from boiles.postprocessing.smoothness import do_weno5_si, symmetry, symmetry_x_fixed_y


class Simulation1D(ObjectiveFunction):

    def __init__(self,
                 results_folder: str,
                 result_filename: str = 'data_1.00*.h5',
                 git: bool = False,
                 # shape: tuple = None
                 ):
        self.dimension = 1
        super(Simulation1D, self).__init__(results_folder, result_filename, git=git)
        # self.shape = shape
        self.smoothness_threshold = 0.33
        if self.result_exit:
            self.result = self.get_results(self.result_path)

    def get_ordered_data(self, file, state: str, order):
        data = try_get_data(file, state, self.dimension)
        if data is not None:
            # if state == "velocity":
            #     data["velocity_x"] = np.array(data)
            # else:
            data = np.array(data[order])

            return data
        else:
            return None

    def get_results(self, file):

        with h5py.File(file, "r") as data:
            cell_vertices = np.array(data["mesh_topology"]["cell_vertex_IDs"])
            vertex_coordinates = np.array(data["mesh_topology"]["cell_vertex_coordinates"])

        coords, order = get_coords_and_order(cell_vertices, vertex_coordinates, self.dimension)
        # edge_cells_number: the cell number along each dimension
        # edge_cells_number, is_integer = sympy.integer_nthroot(coords.shape[0], self.dimension)
        # if self.shape is None:
        #     self.shape = (edge_cells_number, edge_cells_number)
        density = self.get_ordered_data(file, "density", order)
        pressure = self.get_ordered_data(file, "pressure", order)
        velocity = self.get_ordered_data(file, "velocity", order)
        kinetic_energy = 0.5 * density * velocity**2
        effective_dissipation_rate = self.get_ordered_data(file, "effective_dissipation_rate", order)
        numerical_dissipation_rate = self.get_ordered_data(file, "numerical_dissipation_rate", order)
        ducros = self.get_ordered_data(file, "ducros", order)
        schlieren = self.get_ordered_data(file, "schlieren", order)
        temperature = self.get_ordered_data(file, "temperature", order)
        thermal_conductivity = self.get_ordered_data(file, "thermal_conductivity", order)

        data_dict = {
            'density': density,
            'pressure': pressure,
            'velocity': velocity,
            'coords': coords,
            'effective_dissipation_rate': effective_dissipation_rate,
            'numerical_dissipation_rate': numerical_dissipation_rate,
            'ducros': ducros,
            'kinetic_energy': kinetic_energy,
            'schlieren': schlieren,
            'temperature': temperature,
            'thermal_conductivity': thermal_conductivity
        }

        return data_dict

    def smoothness(self, state="numerical_dissipation_rate", threshold=None):
        if threshold is None:
            threshold = self.smoothness_threshold
        return internal_smoothness(self.result[state], threshold=threshold)

    def truncation_errors(self):
        r"""
            return: dissipation, dispersion, true_error, abs_error
        """
        num_rate = self.result["numerical_dissipation_rate"]
        dissipation = np.where(num_rate >= 0, num_rate, 0).sum()
        dispersion = np.where(num_rate <= 0, num_rate, 0).sum()
        true_error = num_rate.sum()
        abs_error = np.abs(num_rate).sum()
        return dissipation, dispersion, true_error, abs_error


def internal_smoothness(value, threshold=0.333):
    # compute internal smoothness indicator (internal means we don't construct boundary, e.g. 64*64 -> 60*60)
    x_size = len(value[2:-2])
    si_buffer = np.zeros(x_size)
    stencil = np.zeros(5)
    for x in np.arange(x_size):
        stencil[0] = value[x]
        stencil[1] = value[x + 1]
        stencil[2] = value[x + 2]
        stencil[3] = value[x + 3]
        stencil[4] = value[x + 4]
        _, a2_weno5_x, a3_weno5_x = do_weno5_si(stencil)

        si_buffer[x] = min([a2_weno5_x, a3_weno5_x])
    si_regular = np.where(si_buffer > threshold, 1, 0)
    score = si_regular.sum() / x_size
    return si_regular, score
