#!/usr/bin/env python3

import matplotlib.pyplot as plt
import h5py
import sympy
from .base import ObjectiveFunction, try_get_data, get_coords_and_order, alpaca_available_quantities, jaxfluids_available_quantities
import numpy as np
from boiles.postprocessing.smoothness import do_weno5_si, symmetry, symmetry_x_fixed_y


class Simulation3D(ObjectiveFunction):

    def __init__(
            self,
            file: str,
            shape: tuple = None,
            quantities: list = ["density", "velocity"],
            solver: str = "ALPACA",
        ):
        self.dimension = 3
        super(Simulation3D, self).__init__(file=file)
        self.shape = shape
        self.smoothness_threshold = 0.33
        self.is_square = False
        for quantity in quantities:
            if solver.lower() == "alpaca":
                assert quantity in alpaca_available_quantities, f"Invalid quantity: {quantity}. Valid quantities are {alpaca_available_quantities}"
                self.result = self.get_alpaca_results(self.file, quantities)
            elif solver.lower() == "jaxfluids":
                assert quantity in jaxfluids_available_quantities["primes"], f"Invalid quantity: {quantity}. Valid quantities are {jaxfluids_available_quantities['primes']}"
                self.result = self.get_jaxfluids_results(self.file, quantities)

    def get_ordered_data(self, file, quantity: str, order):
        data = try_get_data(file, quantity, self.dimension)
        if quantity == "velocity":
            data["velocity_x"] = np.array(data["velocity_x"])[order].reshape(self.shape)
            data["velocity_y"] = np.array(data["velocity_y"])[order].reshape(self.shape)
            data["velocity_z"] = np.array(data["velocity_z"])[order].reshape(self.shape)
        else:
            data = np.array(data[order]).reshape(self.shape)
        return data

    def get_alpaca_results(self, file, quantities):
        data_dict = {}
        with h5py.File(file, "r") as data:
            cell_vertices = np.array(data["mesh_topology"]["cell_vertex_IDs"])
            vertex_coordinates = np.array(data["mesh_topology"]["cell_vertex_coordinates"])

        coords, order = get_coords_and_order(cell_vertices, vertex_coordinates, self.dimension)
        # edge_cells_number: the cell number along each dimension
        edge_cells_number, self.is_square = sympy.integer_nthroot(coords.shape[0], self.dimension)
        if self.shape is None:
            self.shape = (edge_cells_number, edge_cells_number, edge_cells_number)
        for quantity in quantities:
            data_dict[quantity] = self.get_ordered_data(file, quantity, order)
        if "density" in quantities and "velocity" in quantities:
            data_dict["kinetic_energy"] = 0.5 * data_dict["density"] * (data_dict["velocity"]["velocity_x"]**2 + data_dict["velocity"]["velocity_y"]**2 + data_dict["velocity"]["velocity_z"]**2)
        
        return data_dict
    
    def get_jaxfluids_results(self, file, quantities):
        data_dict = {}
        with h5py.File(file, "r") as h5file:
            for quantity in quantities:
                assert quantity in jaxfluids_available_quantities["primes"], f"Invalid quantity: {quantity}. Valid primes quantities are {jaxfluids_available_quantities['primes']}"
                if quantity == "velocity":
                    data_dict[quantity] = {}
                    data_dict[quantity]["velocity_x"] = h5file["primes/velocity"][..., 0]
                    data_dict[quantity]["velocity_y"] = h5file["primes/velocity"][..., 1]
                    data_dict[quantity]["velocity_z"] = h5file["primes/velocity"][..., 2]
                else:
                    data_dict[quantity] = h5file[f"primes/{quantity}"][:]
        _, self.is_square = sympy.integer_nthroot(data_dict["density"].flatten().size, self.dimension)
        self.shape = data_dict["density"].shape
        return data_dict

    def truncation_errors(self):
        r"""
            return: dissipation, dispersion, true_error, abs_error
        """
        assert "numerical_dissipation_rate" in self.result.keys(), "numerical_dissipation_rate not found in result"
        num_rate = self.result["numerical_dissipation_rate"]
        dissipation = np.where(num_rate >= 0, num_rate, 0).sum()
        dispersion = np.where(num_rate <= 0, num_rate, 0).sum()
        true_error = num_rate.sum()
        abs_error = np.abs(num_rate).sum()
        return dissipation, dispersion, true_error, abs_error

    def _create_spectrum(self):
        assert "velocity" in self.result.keys(), "velocity not found in result"
        if not self.is_square:
            raise RuntimeError("For non-square domain, no spectrum can be computed!")
        velocity_x = self.result["velocity"]['velocity_x']
        velocity_y = self.result["velocity"]['velocity_y']
        velocity_z = self.result["velocity"]['velocity_z']
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

        # np.savetxt(self.results_folder + self.result_filename[:-8] + '.csv', dataout, delimiter=",")

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

    def plot_tke(self, plot_reference = False, y_lower: float = 1e-15, y_upper = 1, save_path: str = None, grid=False, show=False):
        spectrum_data = self._create_spectrum()
        spectrum_ref  = self._calculate_reference(spectrum_data)
        fig, ax = plt.subplots(dpi=150)
        # ax.set_title(f"Kinetic Energy Spectrum")
        ax.set_xlabel(r"$k$")
        ax.set_ylabel(r"$E(k)$")

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
        if plot_reference:
            ax.loglog(spectrum_data[:, 0], spectrum_ref, 'r', linewidth=0.8, label='$ref$')
        ax.legend(loc='lower left')
        ax.set_ylim(y_lower, y_upper)
        if grid:
            ax.grid(which='both')
        if save_path is not None:
            fig.savefig(save_path)
        plt.tight_layout()
        if show:
            plt.show()
        # plt.show()
        # plt.close(fig)

