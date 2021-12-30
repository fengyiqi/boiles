#!/usr/bin/env python3

import matplotlib.pyplot as plt
import time
import h5py

import sympy
from .base import ObjectiveFunction
# from mytools.config.opt_config import *
import numpy as np


class FreeShearFlow(ObjectiveFunction):

    def __init__(self,
                 results_folder: str,
                 result_filename: str = 'data_10.00*.h5',
                 git: bool = False,
                 cells: int = 64
                 ):
        super(FreeShearFlow, self).__init__(results_folder, result_filename, git=git)
        self.plot_savepath = results_folder
        if self.result_exit:
            self.cells = cells
            self.result = self.get_data(self.result_path, git);
            self.spectrum_data = self._create_spectrum()
            self.reference = self._calculate_reference()
            self.plot_tke()

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

                pressure = np.array(data["cell_data"]["pressure"][:, 0, 0])
                try:
                    effective_diss_rate = np.array(data["cell_data"]["effective_dissipation_rate"][:, 0, 0])

                    numerical_diss_rate = np.array(data["cell_data"]["numerical_dissipation_rate"][:, 0, 0])
                    vorticity = np.array(data["cell_data"]["vorticity"][:, 0, 0])
                except:
                    pass
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
        density = density.reshape(self.cells, self.cells)
        pressure = pressure[trafo]
        pressure = pressure.reshape(self.cells, self.cells)
        velocity_x = velocity_x[trafo]
        velocity_x = velocity_x.reshape(self.cells, self.cells)
        velocity_y = velocity_y[trafo]
        velocity_y = velocity_y.reshape(self.cells, self.cells)
        vorticity = None
        # effective_diss_rate = None
        # numerical_diss_rate = None
        try:
            effective_diss_rate = effective_diss_rate[trafo]
            effective_diss_rate = effective_diss_rate.reshape(self.cells, self.cells)
            numerical_diss_rate = numerical_diss_rate[trafo]
            numerical_diss_rate = numerical_diss_rate.reshape(self.cells, self.cells)
            # vorticity = vorticity[trafo]
            # vorticity = vorticity.reshape(self.cells, self.cells)
        except:
            pass

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
                     'coords': coords,
                     'effective_dissipation_rate': effective_diss_rate,
                     'numerical_dissipation_rate': numerical_diss_rate
                     }

        return data_dict

    def _create_spectrum(self):
        velocity_x = self.result['velocity']['velocity_x']
        velocity_y = self.result['velocity']['velocity_y']
        N = self.result['nc']
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