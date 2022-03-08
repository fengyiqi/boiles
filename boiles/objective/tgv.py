#!/usr/bin/env python3

import matplotlib.pyplot as plt
import time
import h5py

import sympy
from ..config.opt_config import OC
from ..config.opt_problems import OP
from ..test_cases.tgv import TGV
from ..test_cases.tgv import TGV
from .simulation3d import Simulation3D
import numpy as np

start_wn = 3


class TaylorGreenVortex(Simulation3D):

    def __init__(self,
                 results_folder: str,
                 result_filename: str = 'data_20.00*.h5',
                 git: bool = False,
                 plot: bool = False,
                 ):
        super(TaylorGreenVortex, self).__init__(results_folder, result_filename, git=git)
        self.center = 0
        self.realize = 0

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
            effective_wn = slice(start_wn, self.realsize + 1)

            spectrum_data = self._create_spectrum()
            spectrum_ref = self._calculate_reference(spectrum_data)
            wn_wise_error = np.log(spectrum_data[effective_wn, 1]) - np.log(spectrum_ref[effective_wn])
            error = np.linalg.norm(wn_wise_error, ord=2)
            if OP.test_cases is None:
                return error
            else:
                clipped_error = np.clip(error, -np.inf, upper_bound)
                return clipped_error, error, 'completed'
        else:
            return upper_bound, np.nan, 'divergent'

