#!/usr/bin/env python3

import matplotlib.pyplot as plt
import h5py
import sympy
from .base import ObjectiveFunction, try_get_data, get_coords_and_order
import numpy as np
from boiles.postprocessing.smoothness import do_weno5_si, symmetry, symmetry_x_fixed_y


class Simulation3D(ObjectiveFunction):

    def __init__(self,
                 results_folder: str,
                 result_filename: str = 'data_20.00*.h5',
                 git: bool = False,
                 shape: tuple = None
                 ):
        self.dimension = 3
        super(Simulation3D, self).__init__(results_folder, result_filename, git=git)
        self.shape = shape
        self.smoothness_threshold = 0.33
        if self.result_exit:
            self.result, self.is_square = self.get_results(self.result_path)

    def get_ordered_data(self, file, state: str, order):
        data = try_get_data(file, state, self.dimension)
        if data is not None:
            if state == "velocity":
                data["velocity_x"] = np.array(data["velocity_x"])[order].reshape(self.shape)
                data["velocity_y"] = np.array(data["velocity_y"])[order].reshape(self.shape)
                data["velocity_z"] = np.array(data["velocity_z"])[order].reshape(self.shape)
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
        edge_cells_number, is_integer = sympy.integer_nthroot(coords.shape[0], self.dimension)
        if self.shape is None:
            self.shape = (edge_cells_number, edge_cells_number, edge_cells_number)
        density = self.get_ordered_data(file, "density", order)
        pressure = self.get_ordered_data(file, "pressure", order)
        velocity = self.get_ordered_data(file, "velocity", order)
        kinetic_energy = 0.5 * density * (velocity["velocity_x"]**2 + velocity["velocity_y"]**2 + velocity["velocity_z"]**2)
        effective_dissipation_rate = self.get_ordered_data(file, "effective_dissipation_rate", order)
        numerical_dissipation_rate = self.get_ordered_data(file, "numerical_dissipation_rate", order)
        vorticity = self.get_ordered_data(file, "vorticity", order)
        ducros = self.get_ordered_data(file, "ducros", order)
        schlieren = self.get_ordered_data(file, "schlieren", order)
        temperature = self.get_ordered_data(file, "temperature", order)
        thermal_conductivity = self.get_ordered_data(file, "thermal_conductivity", order)

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
            'schlieren': schlieren,
            'temperature': temperature,
            'thermal_conductivity': thermal_conductivity
        }

        return data_dict, is_integer

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

    def _create_spectrum(self):
        if not self.is_square:
            raise RuntimeError("For non-square domain, no spectrum can be computed!")
        velocity_x = self.result['velocity']['velocity_x']
        velocity_y = self.result['velocity']['velocity_y']
        velocity_z = self.result['velocity']['velocity_z']
        N = self.shape[0]
        # U0 = 33.13148
        U0 = 1.0

        eps = 1e-50  # to void log(0)

        U = velocity_x / U0
        V = velocity_y / U0
        W = velocity_z / U0

        amplsU = abs(np.fft.fftn(U) / U.size)
        amplsV = abs(np.fft.fftn(V) / V.size)
        amplsW = abs(np.fft.fftn(W) / W.size)

        EK_U = amplsU ** 2
        EK_V = amplsV ** 2
        EK_W = amplsW ** 2

        EK_U = np.fft.fftshift(EK_U)
        EK_V = np.fft.fftshift(EK_V)
        EK_W = np.fft.fftshift(EK_W)

        box_sidex = np.shape(EK_U)[0]
        box_sidey = np.shape(EK_U)[1]
        box_sidez = np.shape(EK_U)[2]

        box_radius = int(np.ceil((np.sqrt((box_sidex) ** 2 + (box_sidey) ** 2 + (box_sidez) ** 2)) / 2.) + 1)

        centerx = int(box_sidex / 2)
        centery = int(box_sidey / 2)
        centerz = int(box_sidez / 2)
        self.center = centerx

        EK_U_avsphr = np.zeros(box_radius, ) + eps  ## size of the radius
        EK_V_avsphr = np.zeros(box_radius, ) + eps  ## size of the radius
        EK_W_avsphr = np.zeros(box_radius, ) + eps  ## size of the radius

        for i in range(box_sidex):
            for j in range(box_sidey):
                for k in range(box_sidez):
                    wn = int(np.round(np.sqrt((i - centerx) ** 2 + (j - centery) ** 2 + (k - centerz) ** 2)))
                    EK_U_avsphr[wn] = EK_U_avsphr[wn] + EK_U[i, j, k]
                    EK_V_avsphr[wn] = EK_V_avsphr[wn] + EK_V[i, j, k]
                    EK_W_avsphr[wn] = EK_W_avsphr[wn] + EK_W[i, j, k]

        EK_avsphr = 0.5 * (EK_U_avsphr + EK_V_avsphr + EK_W_avsphr)
        self.realsize = len(np.fft.rfft(U[:, 0, 0]))

        dataout = np.zeros((box_radius, 2))
        dataout[:, 0] = np.arange(0, len(dataout))
        dataout[:, 1] = EK_avsphr[0:len(dataout)]

        np.savetxt(self.results_folder + self.result_filename[:-8] + '.csv', dataout, delimiter=",")

        return dataout

    def _calculate_A(self, spectrum):

        effective_wn = slice(7, self.realsize + 1)

        wn_for_interpolation = spectrum[effective_wn, 0]
        ke_for_interpolation = spectrum[effective_wn, 1].reshape(-1, 1)
        p = np.power(wn_for_interpolation, -5 / 3).reshape(-1, 1)
        A = (np.linalg.inv(p.T.dot(p))).dot(p.T).dot(ke_for_interpolation)
        return A.squeeze()

    def _calculate_reference(self, spectrum):
        A = self._calculate_A(spectrum)
        reference = A * np.power(spectrum[:, 0], -5 / 3)
        return reference

    def plot_tke(self, save_path: str = None):
        spectrum_data = self._create_spectrum()
        spectrum_ref  = self._calculate_reference(spectrum_data)
        fig, ax = plt.subplots(dpi=150)
        ax.set_title(f"Kinetic Energy Spectrum at t={self.result_filename[5:-8]}")
        ax.set_xlabel(r"k (wavenumber)")
        ax.set_ylabel(r"TKE of the k$^{th}$ wavenumber")

        ax.loglog(np.arange(0, self.realsize),
                  spectrum_data[0:self.realsize, 1],
                  'k',
                  linewidth=0.8,
                  label=f'$k<{self.realsize}$')
        ax.loglog(np.arange(self.realsize, spectrum_data.shape[0]),
                  spectrum_data[self.realsize:, 1],
                  'k--',
                  linewidth=0.8,
                  label=f'$k\geq{self.realsize}$')
        ax.loglog(spectrum_data[:, 0], spectrum_ref, 'r', linewidth=0.8, label='$ref$')
        ax.legend(loc='lower left')
        ax.set_ylim(10 ** -15, 1)
        ax.grid(which='both')
        if save_path is not None:
            fig.savefig(save_path)
        plt.show()
        plt.close(fig)

