#!/usr/bin/env python3

import matplotlib.pyplot as plt
import time
import h5py

import sympy
from ..config.opt_config import OC
from ..config.opt_problems import OP
from ..test_cases.tgv import TGV
from ..test_cases.tgv_teno5 import TGVTENO5
from .simulation3d import Simulation3D
import numpy as np

start_wn = 3


class TaylorGreenVortex(Simulation3D):

    def __init__(
            self,
            file: str,
            shape: tuple = None,
            quantities: list = ["density", "velocity"],
            solver: str = "ALPACA",
    ):
        super(TaylorGreenVortex, self).__init__(file=file, shape=shape, quantities=quantities, solver=solver)
        self.center = 0
        self.realsize = 0

    def objective_spectrum(self, mse: bool = False):
        r"""
        Return a clamped error
        :return: error and simulation state.
        """

        if OP.test_cases is not None:
            upper_bound = TGVTENO5.highest_error_from_initial
        else:
            upper_bound = np.inf
        if self.result_exit:
            spectrum_data = self._create_spectrum()
            spectrum_ref = self._calculate_reference(spectrum_data)
            effective_wn = slice(start_wn, self.realsize + 1)
            wn_wise_error = np.log(spectrum_data[effective_wn, 1]) - np.log(spectrum_ref[effective_wn])
            error = np.linalg.norm(wn_wise_error, ord=2)
            if mse:
                error = error ** 2 / (self.realsize + 1 - start_wn)
            if OP.test_cases is None:
                return error
            else:
                clipped_error = np.clip(error, -np.inf, upper_bound)
                return clipped_error, error, 'completed'
        else:
            return upper_bound, np.nan, 'divergent'

