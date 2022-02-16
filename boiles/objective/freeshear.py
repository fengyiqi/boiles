#!/usr/bin/env python3

import matplotlib.pyplot as plt
import time
import h5py

import sympy
from .base import ObjectiveFunction, try_get_data, get_coords_and_order
# from mytools.config.opt_config import *
import numpy as np


class FreeShearFlow(ObjectiveFunction):

    def __init__(self,
                 results_folder: str,
                 result_filename: str = 'data_1.00*.h5',
                 git: bool = False,
                 ):
        self.dimension = 2
        super(FreeShearFlow, self).__init__(results_folder, result_filename, git=git)
        self.plot_savepath = results_folder
        self.spectrum_data = np.array([])
        self.reference = np.array([])
        if self.result_exit:
            self.result = self.get_results(self.result_path)
            # self.spectrum_data = self._create_spectrum()
            # self.reference = self._calculate_reference()
            # self.plot_tke()

    def get_ordered_data(self, file, state: str, order, edge_cells):
        data = try_get_data(file, state, self.dimension)
        if data is not None:
            if state == "velocity":
                data["velocity_x"] = np.array(data["velocity_x"])[order].reshape(edge_cells, edge_cells)
                data["velocity_y"] = np.array(data["velocity_y"])[order].reshape(edge_cells, edge_cells)
            else:
                data = np.array(data[order])
                data = data.reshape(edge_cells, edge_cells)
            return data
        else:
            return None

    def get_results(self, file):

        with h5py.File(file, "r") as data:
            cell_vertices = np.array(data["mesh_topology"]["cell_vertex_IDs"])
            vertex_coordinates = np.array(data["mesh_topology"]["cell_vertex_coordinates"])

        coords, order = get_coords_and_order(cell_vertices, vertex_coordinates, self.dimension)
        # edge_cells_number: the cell number along each dimension
        edge_cells_number, is_integer = sympy.integer_nthroot(coords.shape[0], self.dimension)
        density = self.get_ordered_data(file, "density", order, edge_cells_number)
        pressure = self.get_ordered_data(file, "pressure", order, edge_cells_number)
        velocity = self.get_ordered_data(file, "velocity", order, edge_cells_number)
        kinetic_energy = 0.5 * density * (velocity["velocity_x"]**2 + velocity["velocity_y"]**2)
        effective_dissipation_rate = self.get_ordered_data(file, "effective_dissipation_rate", order, edge_cells_number)
        numerical_dissipation_rate = self.get_ordered_data(file, "numerical_dissipation_rate", order, edge_cells_number)
        vorticity = self.get_ordered_data(file, "vorticity", order, edge_cells_number)
        ducros = self.get_ordered_data(file, "ducros", order, edge_cells_number)
        schlieren = self.get_ordered_data(file, "schlieren", order, edge_cells_number)

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

    def _create_spectrum(self):
        velocity_x = self.result['velocity']['velocity_x']
        velocity_y = self.result['velocity']['velocity_y']
        N = self.result['density'].shape[0]
        # U0 = 33.13148
        U0 = 1.0
        Figs_Path = self.results_folder

        localtime = time.asctime(time.localtime(time.time()))

        eps = 1e-50  # to void log(0)

        U = velocity_x / U0
        V = velocity_y / U0

        amplsU = abs(np.fft.fftn(U) / U.size)
        amplsV = abs(np.fft.fftn(V) / V.size)

        EK_U = amplsU ** 2
        EK_V = amplsV ** 2

        EK_U = np.fft.fftshift(EK_U)
        EK_V = np.fft.fftshift(EK_V)
        box_sidex = np.shape(EK_U)[0]
        box_sidey = np.shape(EK_U)[1]

        box_radius = int(np.ceil((np.sqrt((box_sidex) ** 2 + (box_sidey) ** 2 )) / 2.) + 1)

        centerx = int(box_sidex / 2)
        centery = int(box_sidey / 2)
        self.center = centerx

        EK_U_avsphr = np.zeros(box_radius, ) + eps  ## size of the radius
        EK_V_avsphr = np.zeros(box_radius, ) + eps  ## size of the radius

        for i in range(box_sidex):
            for j in range(box_sidey):
                wn = int(np.round(np.sqrt((i - centerx) ** 2 + (j - centery) ** 2)))
                EK_U_avsphr[wn] = EK_U_avsphr[wn] + EK_U[i, j]
                EK_V_avsphr[wn] = EK_V_avsphr[wn] + EK_V[i, j]

        EK_avsphr = 0.5 * (EK_U_avsphr + EK_V_avsphr)
        self.realsize = len(np.fft.rfft(U[:, 0]))

        localtime = time.asctime(time.localtime(time.time()))

        dataout = np.zeros((box_radius, 2))
        dataout[:, 0] = np.arange(0, len(dataout))
        dataout[:, 1] = EK_avsphr[0:len(dataout)]

        np.savetxt(Figs_Path + self.result_filename[:-8] + '.csv', dataout, delimiter=",")

        return dataout

    def _calculate_A(self):

        effective_wn = slice(7, self.realsize + 1)

        wn_for_interpolation = self.spectrum_data[effective_wn, 0].reshape(-1, 1)
        ke_for_interpolation = self.spectrum_data[effective_wn, 1].reshape(-1, 1)
        p = np.power(wn_for_interpolation, -5 / 3).reshape(-1, 1)
        A = (np.linalg.inv(p.T.dot(p))).dot(p.T).dot(ke_for_interpolation)
        return A

    def _calculate_reference(self):
        A = self._calculate_A()
        reference = A * np.power(self.spectrum_data[:, 0], -5 / 3).reshape(-1, 1)
        return reference

    def plot_tke(self):

        fig, ax = plt.subplots(dpi=150)
        ax.set_title(f"Kinetic Energy Spectrum at t={self.result_filename[5:-8]}")
        ax.set_xlabel(r"k (wavenumber)")
        ax.set_ylabel(r"TKE of the k$^{th}$ wavenumber")

        ax.loglog(np.arange(0, self.realsize),
                  self.spectrum_data[0:self.realsize, 1],
                  'k',
                  label=f'simulation (k<{self.realsize})')
        ax.loglog(np.arange(self.realsize, self.spectrum_data.shape[0]),
                  self.spectrum_data[self.realsize:, 1],
                  'k--',
                  label=f'simulation (k>={self.realsize})')
        ax.loglog(self.spectrum_data[:, 0].squeeze(), self.reference, 'r', label='reference')
        ax.legend(loc='lower left')
        ax.set_ylim(10 ** -15, 1)
        ax.grid(which='both')
        plot_filename = self.plot_savepath + self.result_filename[:-8] + 'tke.png'
        fig.savefig(plot_filename)
        plt.close(fig)