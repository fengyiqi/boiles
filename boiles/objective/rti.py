#!/usr/bin/env python3

import matplotlib.pyplot as plt
import time
import h5py

import sympy
from .base import ObjectiveFunction
# from mytools.config.opt_config import *
import numpy as np


class RTI(ObjectiveFunction):

    def __init__(self,
                 results_folder: str,
                 result_filename: str = 'data_10.00*.h5',
                 git: bool = False,
                 shape: int = (256, 64)
                 ):
        super(RTI, self).__init__(results_folder, result_filename, git=git)
        self.plot_savepath = results_folder
        if self.result_exit:
            self.shape = shape
            self.result = self.get_data(self.result_path, git)

            # self.spectrum_data = self._create_spectrum()
            # self.reference = self._calculate_reference()
            # self.plot_tke()

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
