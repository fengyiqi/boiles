#!/usr/bin/env python3

import matplotlib.pyplot as plt
import time
import h5py

import sympy
from .base import ObjectiveFunction, try_get_data, get_coords_and_order
# from mytools.config.opt_config import *
import numpy as np


class RTI(ObjectiveFunction):

    def __init__(self,
                 results_folder: str,
                 result_filename: str = 'data_2.00*.h5',
                 git: bool = False,
                 shape: int = (128, 32)
                 ):
        super(RTI, self).__init__(results_folder, result_filename, git=git)
        self.dimension = 2
        self.plot_savepath = results_folder
        if self.result_exit:
            self.shape = shape
            self.result = self.get_results(self.result_path)

            # self.spectrum_data = self._create_spectrum()
            # self.reference = self._calculate_reference()
            # self.plot_tke()

    def get_ordered_data(self, file, state: str, order):
        data = try_get_data(file, state, self.dimension)
        if data is not None:
            if state == "velocity":
                data["velocity_x"] = np.array(data["velocity_x"])[order].reshape(self.shape)
                data["velocity_y"] = np.array(data["velocity_y"])[order].reshape(self.shape)
            else:
                data = np.array(data[order])
                data = data.reshape(self.shape)
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
        density = self.get_ordered_data(file, "density", order)
        pressure = self.get_ordered_data(file, "pressure", order)
        velocity = self.get_ordered_data(file, "velocity", order)
        kinetic_energy = 0.5 * density * (velocity["velocity_x"]**2 + velocity["velocity_y"]**2)
        effective_dissipation_rate = self.get_ordered_data(file, "effective_dissipation_rate", order)
        numerical_dissipation_rate = self.get_ordered_data(file, "numerical_dissipation_rate", order)
        vorticity = self.get_ordered_data(file, "vorticity", order)
        ducros = self.get_ordered_data(file, "ducros", order)
        schlieren = self.get_ordered_data(file, "schlieren", order)

        data_dict = {
            'density': density,
            'pressure': pressure,
            'velocity': velocity,
            'vorticity': vorticity,
            'coords': coords,
            'effective_dissipation_rate': effective_dissipation_rate,
            'numerical_dissipation_rate': numerical_dissipation_rate,
            'ducros': ducros,
            'kinetic_energy': kinetic_energy,
            'schlieren': schlieren
        }

        return data_dict

    def get_data(self, file, git):
        with h5py.File(file, "r") as data:
            if git:
                density = np.array(data["simulation"]["density"])
                velocity_x = np.array(data["simulation"]["velocityX"])
                velocity_y = np.array(data["simulation"]["velocityY"])
                pressure = np.array(data["simulation"]["pressure"])
                cell_vertices = np.array(data["domain"]["cell_vertices"])
                vertex_coordinates = np.array(data["domain"]["vertex_coordinates"])
            else:
                density = np.array(data["cell_data"]["density"][:, 0, 0])
                velocity_x = np.array(data["cell_data"]["velocity"][:, 0, 0])
                velocity_y = np.array(data["cell_data"]["velocity"][:, 1, 0])
                vorticity = np.array(data["cell_data"]["vorticity"][:, 0, 0])
                pressure = np.array(data["cell_data"]["pressure"][:, 0, 0])
                cell_vertices = np.array(data["mesh_topology"]["cell_vertex_IDs"])
                vertex_coordinates = np.array(data["mesh_topology"]["cell_vertex_coordinates"])

        nc, is_integer = sympy.integer_nthroot(density.shape[0], 2)
        ordered_vertex_coordinates = vertex_coordinates[cell_vertices]
        coords = np.mean(ordered_vertex_coordinates, axis=1)

        first_trafo = coords[:, 0].argsort(kind='stable')
        coords = coords[first_trafo]
        second_trafo = coords[:, 1].argsort(kind='stable')
        coords = coords[second_trafo]

        trafo = first_trafo[second_trafo]

        density = density[trafo]
        density = density.reshape(self.shape)
        pressure = pressure[trafo]
        pressure = pressure.reshape(self.shape)
        velocity_x = velocity_x[trafo]
        velocity_x = velocity_x.reshape(self.shape)
        velocity_y = velocity_y[trafo]
        velocity_y = velocity_y.reshape(self.shape)
        vorticity = vorticity[trafo]
        vorticity = vorticity.reshape(self.shape)

        velocity = {'velocity_x': velocity_x,
                    'velocity_y': velocity_y
                    }
        data_dict = {'x_cell_center': None,
                     'density': density,
                     'pressure': pressure,
                     'velocity': velocity,
                     'vorticity': vorticity,
                     'internal_energy': None,
                     'kinetic_energy': None,
                     'total_energy': None,
                     'entropy': None,
                     'enthalpy': None,
                     'nc': nc,
                     'cell_vertices': cell_vertices,
                     'vertex_coordinates': vertex_coordinates,
                     'ordered_vertex_coordinates': ordered_vertex_coordinates,
                     'coords': coords

                     }

        return data_dict
