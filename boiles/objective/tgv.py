#!/usr/bin/env python3

import matplotlib.pyplot as plt
import time
import h5py

import sympy
from .base import ObjectiveFunction
from ..config.opt_config import OC
from ..config.opt_problems import OP
from ..test_cases.tgv import TGV
from ..test_cases.tgv import TGV
import numpy as np

start_wn = 3


class TaylorGreenVortex(ObjectiveFunction):

    def __init__(self,
                 results_folder: str,
                 result_filename: str = 'data_20.00*.h5',
                 git: bool = False,
                 plot: bool = True,
                 ):
        super(TaylorGreenVortex, self).__init__(results_folder, result_filename, git=git)
        self.plot = plot
        self.plot_savepath = results_folder
        self.center: int = 0
        # self.realize = 0
        if self.result_exit:
            self.result = self.get_tgv_h5data(self.result_path, git)
            self.spectrum_data = self._create_spectrum()
            self.reference = self._calculate_reference()
            if plot:
                self.plot_tke()

    # @staticmethod
    def get_tgv_h5data(self, file, git=False):

        with h5py.File(file, "r") as data:
            if git:
                density = np.array(data["simulation"]["density"])
                velocity_x = np.array(data["simulation"]["velocityX"])
                velocity_y = np.array(data["simulation"]["velocityY"])
                velocity_z = np.array(data["simulation"]["velocityZ"])
                pressure = np.array(data["simulation"]["pressure"])
                cell_vertices = np.array(data["domain"]["cell_vertices"])
                vertex_coordinates = np.array(data["domain"]["vertex_coordinates"])
            else:
                density = np.array(data["cell_data"]["density"][:, 0, 0])
                velocity_x = np.array(data["cell_data"]["velocity"][:, 0, 0])
                velocity_y = np.array(data["cell_data"]["velocity"][:, 1, 0])
                velocity_z = np.array(data["cell_data"]["velocity"][:, 2, 0])
                pressure = np.array(data["cell_data"]["pressure"][:, 0, 0])
                try:
                    effective_diss_rate = np.array(data["cell_data"]["effective_dissipation_rate"][:, 0, 0])
                    numerical_diss_rate = np.array(data["cell_data"]["numerical_dissipation_rate"][:, 0, 0])
                except:
                    pass
                cell_vertices = np.array(data["mesh_topology"]["cell_vertex_IDs"])
                vertex_coordinates = np.array(data["mesh_topology"]["cell_vertex_coordinates"])

        # Number of cells per dimension:
        # Only meaningful for cubic setups, otherwise full symmetry cannot be expected
        self.nc, is_integer = sympy.integer_nthroot(density.shape[0], 3)
        nc = self.nc
        if not is_integer:
            log = ['ERROR: Domain has no cubic size']
            return 1

        ordered_vertex_coordinates = vertex_coordinates[cell_vertices]
        coords = np.mean(ordered_vertex_coordinates, axis=1)

        first_trafo = coords[:, 0].argsort(kind='stable')
        coords = coords[first_trafo]
        second_trafo = coords[:, 1].argsort(kind='stable')
        coords = coords[second_trafo]
        third_trafo = coords[:, 2].argsort(kind='stable')
        coords = coords[third_trafo]

        trafo = first_trafo[second_trafo[third_trafo]]
        # Corner points
        # log = ['---Test Symmetry at 8 Corner Points---']

        density = density[trafo]
        density = density.reshape(nc, nc, nc)
        pressure = pressure[trafo]
        pressure = pressure.reshape(nc, nc, nc)
        velocity_x = velocity_x[trafo]
        velocity_x = velocity_x.reshape(nc, nc, nc)
        velocity_y = velocity_y[trafo]
        velocity_y = velocity_y.reshape(nc, nc, nc)
        velocity_z = velocity_z[trafo]
        velocity_z = velocity_z.reshape(nc, nc, nc)
        effective_diss_rate = None
        numerical_diss_rate = None
        try:
            effective_diss_rate = effective_diss_rate[trafo]
            effective_diss_rate = effective_diss_rate.reshape(nc, nc, nc)
            numerical_diss_rate = numerical_diss_rate[trafo]
            numerical_diss_rate = numerical_diss_rate.reshape(nc, nc, nc)
        except:
            pass

        velocity = {'velocity_x': velocity_x,
                    'velocity_y': velocity_y,
                    'velocity_z': velocity_z
                    }
        data_dict = {'x_cell_center': None,
                     'density': density,
                     'pressure': pressure,
                     'velocity': velocity,
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
        velocity_z = self.result['velocity']['velocity_z']
        N = self.result['nc']
        # U0 = 33.13148
        U0 = 1.0
        Figs_Path = self.results_folder

        localtime = time.asctime(time.localtime(time.time()))
        # print("Computing spectrum... ", localtime)

        # print("N =", N)
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

        # print("box sidex     =", box_sidex)
        # print("box sidey     =", box_sidey)
        # print("box sidez     =", box_sidez)
        # print("sphere radius =", box_radius)
        # print("centerbox     =", centerx)
        # print("centerboy     =", centery)
        # print("centerboz     =", centerz, "\n")

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
        #
        # print("Real      Kmax    = ", self.realsize)
        # print("Spherical Kmax    = ", len(EK_avsphr))

        TKEofmean_discrete = 0.5 * (np.sum(U / U.size) ** 2 + np.sum(V / V.size) ** 2 + np.sum(W / W.size) ** 2)
        TKEofmean_sphere = EK_avsphr[0]

        total_TKE_discrete = np.sum(0.5 * (U ** 2 + V ** 2 + W ** 2)) / (N * 1.0) ** 3
        total_TKE_sphere = np.sum(EK_avsphr)

        # print("the KE  of the mean velocity discrete  = ", TKEofmean_discrete)
        # print("the KE  of the mean velocity sphere    = ", TKEofmean_sphere)
        # print("the mean KE discrete  = ", total_TKE_discrete)
        # print("the mean KE sphere    = ", total_TKE_sphere)

        localtime = time.asctime(time.localtime(time.time()))
        # print("Computing spectrum... ", localtime, "- END \n")

        dataout = np.zeros((box_radius, 2))
        dataout[:, 0] = np.arange(0, len(dataout))
        dataout[:, 1] = EK_avsphr[0:len(dataout)]

        np.savetxt(Figs_Path + self.result_filename[:-8] + '.csv', dataout, delimiter=",")

        return dataout

    def _calculate_A(self):

        effective_wn = slice(start_wn, self.realsize + 1)

        wn_for_interpolation = self.spectrum_data[effective_wn, 0].reshape(-1, 1)
        ke_for_interpolation = self.spectrum_data[effective_wn, 1].reshape(-1, 1)
        p = np.power(wn_for_interpolation, -5 / 3).reshape(-1, 1)
        A = (np.linalg.inv(p.T.dot(p))).dot(p.T).dot(ke_for_interpolation)
        return A

    def _calculate_reference(self):
        A = self._calculate_A()
        reference = A * np.power(self.spectrum_data[:, 0], -5 / 3).reshape(-1, 1)
        return reference

    def _log_difference_norm(self, norm_ord=2):

        effective_wn = slice(start_wn, self.realsize + 1)
        reference = self._calculate_reference()
        diff_by_wn = np.log(self.spectrum_data[effective_wn, 1].reshape(-1, 1)) - np.log(reference[effective_wn])
        return np.linalg.norm(diff_by_wn, ord=norm_ord)

    def objective_spectrum(self):
        r"""
        Return a clamped error
        :return: error and simulation state.
        """
        if OP.test_cases is not None:
            upper_bound = OP.test_cases[-1].highest_error_from_initial
        else:
            upper_bound = np.inf
        if self.result_exit:
            real_error = self._log_difference_norm()
            clipped_error = np.clip(real_error, -np.inf, upper_bound)
            # print(f'true error: {error}')
            if OP.test_cases is None:
                return clipped_error
            else:
                return clipped_error, 'completed', real_error
        else:
            return upper_bound, 'divergent', 10000

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
        ax.set_ylim(10 ** -15, 10 ** -1)
        ax.grid(which='both', axis='x')
        plot_filename = self.plot_savepath + self.result_filename[:-8] + 'tgv.png'
        fig.savefig(plot_filename)
        plt.close(fig)

    def total_ke(self):
        # vel = self.result['velocity_array']
        # pho = self.result['density']
        # return 0.5 * ((vel ** 2).sum() * pho).sum() * ((2 * np.pi / 64) ** 3)
        if self.result_exit:
            velocity_x = self.result['velocity']['velocity_x']
            velocity_y = self.result['velocity']['velocity_y']
            velocity_z = self.result['velocity']['velocity_z']
            density = self.result['density']
            dx = 2 * np.pi / self.nc
            ke = 0.5 * density * (velocity_x ** 2 + velocity_y ** 2 + velocity_z ** 2) * (dx ** 3)

            return ke.sum()
        else:
            return -1
