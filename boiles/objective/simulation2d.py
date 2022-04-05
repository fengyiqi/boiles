#!/usr/bin/env python3

import matplotlib.pyplot as plt
import h5py
import sympy
from .base import ObjectiveFunction, try_get_data, get_coords_and_order
import numpy as np
from boiles.postprocessing.smoothness import do_weno5_si, symmetry, symmetry_x_fixed_y


class Simulation2D(ObjectiveFunction):

    def __init__(
            self,
            file: str,
            shape: tuple = None
        ):
        self.dimension = 2
        super(Simulation2D, self).__init__(file=file)
        self.shape = shape
        self.smoothness_threshold = 0.33
        self.center = 0
        self.realize = 0
        if self.result_exit:
            self.result, self.is_square = self.get_results(self.file)

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
        edge_cells_number, is_integer = sympy.integer_nthroot(coords.shape[0], self.dimension)
        if self.shape is None:
            self.shape = (edge_cells_number, edge_cells_number)
        density = self.get_ordered_data(file, "density", order)
        pressure = self.get_ordered_data(file, "pressure", order)
        velocity = self.get_ordered_data(file, "velocity", order)
        kinetic_energy = 0.5 * density * (velocity["velocity_x"]**2 + velocity["velocity_y"]**2)
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

    def smoothness(self, threshold=None, property="numerical_dissipation_rate"):
        if threshold is None:
            threshold = self.smoothness_threshold
        return internal_smoothness(self.result[property], threshold=threshold)

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

    def plot(self, prop: str):
        plt.figure(figsize=(5, 4), dpi=100)
        plt.imshow(self.result[prop], origin="lower")
        plt.title(prop)
        plt.colorbar()
        plt.tight_layout()
        plt.show()


    def _create_spectrum(self):
        if not self.is_square:
            raise RuntimeError("For non-square domain, no spectrum can be computed!")
        velocity_x = self.result['velocity']['velocity_x']
        velocity_y = self.result['velocity']['velocity_y']
        N = self.shape[0]
        # U0 = 33.13148
        U0 = 1.0
        # Figs_Path = self.results_folder

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

        dataout = np.zeros((box_radius, 2))
        dataout[:, 0] = np.arange(0, len(dataout))
        dataout[:, 1] = EK_avsphr[0:len(dataout)]

        # np.savetxt(Figs_Path + self.result_filename[:-8] + '.csv', dataout, delimiter=",")

        return dataout

    def _calculate_A(self, spectrum):

        effective_wn = slice(7, self.realsize + 1)

        wn_for_interpolation = spectrum[effective_wn, 0].reshape(-1, 1)
        ke_for_interpolation = spectrum[effective_wn, 1].reshape(-1, 1)
        p = np.power(wn_for_interpolation, -5 / 3).reshape(-1, 1)
        A = (np.linalg.inv(p.T.dot(p))).dot(p.T).dot(ke_for_interpolation)
        return A

    def _calculate_reference(self, spectrum):
        A = self._calculate_A(spectrum)
        reference = A * np.power(spectrum[:, 0], -5 / 3).reshape(-1, 1)
        return reference

    def plot_tke(self, save_path: str = None):
        spectrum_data = self._create_spectrum()
        spectrum_ref  = self._calculate_reference(spectrum_data)
        fig, ax = plt.subplots(dpi=150)
        ax.set_title(f"Kinetic Energy Spectrum")
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
        ax.loglog(spectrum_data[:, 0].squeeze(), spectrum_ref, 'r', linewidth=0.8, label='$ref$')
        ax.legend(loc='lower left')
        ax.set_ylim(10 ** -15, 1)
        ax.grid(which='both')
        if save_path is not None:
            fig.savefig(save_path)
        plt.show()
        plt.close(fig)


def internal_smoothness(value, threshold=0.333):
    # compute internal smoothness indicator (internal means we don't construct boundary, e.g. 64*64 -> 60*60)
    x_size, y_size = value[2:-2, 2:-2].shape
    si_buffer = np.zeros((x_size, y_size))
    stencil = np.zeros(5)
    for y in np.arange(y_size):
        for x in np.arange(x_size):
            stencil[0] = value[x, y + 2]
            stencil[1] = value[x + 1, y + 2]
            stencil[2] = value[x + 2, y + 2]
            stencil[3] = value[x + 3, y + 2]
            stencil[4] = value[x + 4, y + 2]
            _, a2_weno5_x, a3_weno5_x = do_weno5_si(stencil)
            stencil[0] = value[x + 2, y]
            stencil[1] = value[x + 2, y + 1]
            stencil[2] = value[x + 2, y + 2]
            stencil[3] = value[x + 2, y + 3]
            stencil[4] = value[x + 2, y + 4]
            _, a2_weno5_y, a3_weno5_y = do_weno5_si(stencil)
            si_buffer[x, y] = min([a2_weno5_x, a3_weno5_x, a2_weno5_y, a3_weno5_y])
    si_regular = np.where(si_buffer > threshold, 1, 0)
    score = si_regular.sum() / (x_size * y_size)
    return si_regular, score
