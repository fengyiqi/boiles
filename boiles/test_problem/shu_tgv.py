#!/usr/bin/env python3

import os
from botorch.test_functions.base import MultiObjectiveTestProblem
from typing import Optional
from ..objective.shuosher import ShuOsher
from ..objective.tgv import TaylorGreenVortex
from ..utils import read_from_csv, append_to_csv, log
from ..solvers.alpaca_builder import AlpacaBuilder
from ..config.opt_config import OC
from ..config.opt_problems import OP
from ..solvers.solver_config import SC
from ..test_cases.shuosher_200 import ShuBase200
from ..test_cases.tgv import TGV
import numpy as np

alpaca_loc = SC.solver_folder


class ShuOsherTGV(MultiObjectiveTestProblem):
    r"""Two objective problem composed of the Shu-Osher problem and Taylor-Green Vortex.

    ShuOsher:

        maximize gradient near jump but minimize gradient difference along continuous region

    TGV:

        log-scale difference between kinetic energy ans -5/3 principle

    """
    dim = OC.dim_inputs
    num_objectives = OC.dim_outputs
    _bounds = OC.opt_bounds
    _ref_point = OP.ref_point

    def __init__(self, case_num, noise_std: Optional[float] = None, negate: bool = False) -> None:
        r"""Constructor for ShuOsher-TGV.

        Arguments:
            noise_std: Standard deviation of the observation noise.
            negate: If True, negate the objectives.
        """
        self.shu_case_num = case_num  # count the number of simulated shu case
        self.tgv_case_num = case_num
        """
        When calculating the hypervolume the objective will be evaluated repeatedly, which is
        impossible for CFD solver. Three dictionaries are defined here to store the obtained objectives indexed
        by 'cq, q' string.
        """
        self.tgv_dict = {}
        self.shu_disper_dic = {}
        self.shu_shock_dic = {}

        self.disper = self.shock = self.disper_raw = self.shock_raw = None
        self.state_disper = self.state_shock = None

        self.turb = self.turb_raw = None
        self.state_turb = None

        super().__init__(noise_std=noise_std, negate=negate)

    def evaluate_true(self, X):
        r"""
        An abstract method of the base class
        """
        return NotImplemented

    def shu_error(self, x):
        r"""
        Get Shu-Osher problem error.
        :param x: Tensor([cq, q])
        :return: Shu-Osher dispersion error and shock capturing indicator
        """
        cq, q = self.scheme_parameters(x)

        # find if the result based on current cq and q has been evaluated.
        if self.shu_shock_dic.get(f'{str(cq)}, {str(q)}', None) is None:
            # when debugging, uses a simple function, else, uses the Shu-Osher problem.
            if OC.debug:
                self.disper, self.shock = np.cos(np.pi * (cq + q) / 2000) * 10, np.sin(np.pi * (cq + q) / 2000) * 30
            else:
                self.calculate_shu_error(cq, q, self.shu_case_num)

            self.shu_disper_dic[f'{str(cq)}, {str(q)}'] = self.disper
            self.shu_shock_dic[f'{str(cq)}, {str(q)}'] = self.shock
            self.shu_case_num += 1
        else:
            self.disper = self.shu_disper_dic[f'{str(cq)}, {str(q)}']
            self.shock = self.shu_shock_dic[f'{str(cq)}, {str(q)}']
        return self.disper, self.shock

    def tgv_error(self, x):
        r"""
        Get TGV error.
        :param x: Tensor([cq, q])
        :return: tgv error
        """
        cq, q = self.scheme_parameters(x)

        # find if the result based on current cq and q has been evaluated.
        if self.tgv_dict.get(f'{str(cq)}, {str(q)}', None) is None:
            # when debugging, uses an easy to evaluate function, else, uses the tgv
            if OC.debug:
                self.turb = np.cos(np.pi * (cq + q) / 2000) * 20
            else:
                self.calculate_tgv_error(cq, q, self.tgv_case_num)
            self.tgv_dict[f'{str(cq)}, {str(q)}'] = self.turb
            self.tgv_case_num += 1
        else:
            self.turb = self.tgv_dict[f'{str(cq)}, {str(q)}']
        return self.turb


    
    @staticmethod
    def scheme_parameters(x):
        return int(x[0] * OC.increment[0]), int(x[1] * OC.increment[1])


    def initialize_alpaca(self,
                        cq: int,
                        q: int,
                        ic: int,  # a value that can be divided by 4
                        dim: int,
                        min_time_step: str = "std::numeric_limits<double>::epsilon()"):
        r"""
        Initialize ALPACA, including set parameters, cmake ALPACA according to the given dimension
        and make ALPACA
        :param cq: cq in WENO5-CU6-M1
        :param q: q in WENO5-CU6-M1
        :param ic: internal cells per block
        :param dim: case dimension
        :param min_time_step: minimum time step size.
        :return: None
        """
        alpaca = AlpacaBuilder()
        alpaca.set_reconstruction_stencil("WENOCU6M1")
        alpaca.set_limit_end_time("true")
        alpaca.set_ic(ic)
        alpaca.set_minimum_time_step_size(min_time_step)
        alpaca.set_m1_scheme(cq, q)

        alpaca.cmake_alpaca(dimension=dim)
        alpaca.compile_alpaca()

        return alpaca

    def find_shuosher_data(self, cq, q, case_num):
        for file in ShuBase200.training_data:
            index_shu = None
            data_sheet = read_from_csv(file)
            index_shu = np.array(np.where((data_sheet[:, 1] == str(cq)) & (data_sheet[:, 2] == str(q))))
            if index_shu.size != 0:
                index_shu = index_shu[0, 0]
                # shu_disper = rows_shu[index_shu, 3].astype(np.float)
                self.disper = data_sheet[index_shu, 3].astype(float)
                self.disper_raw = data_sheet[index_shu, 4].astype(float)
                self.state_disper = data_sheet[index_shu, 5]

                self.shock = data_sheet[index_shu, 6].astype(float)
                self.shock_raw = data_sheet[index_shu, 7].astype(float)
                self.state_shock = data_sheet[index_shu, 8]

                alpaca = AlpacaBuilder()
                alpaca.build_workfolder(case_num)
                log("Found in database!")

                return True

        return False

    def calculate_shu_error(self, cq, q, case_num):
        r"""
        Run Shu-Osher problem simulation and get dispersion error or shock capturing indicator.

        Two objectives can be directly obtained from a csv data file which is obtained from
        previously simulated results by fixing the random seed, which could save a lot of time.

        :param cq: cq in WENO5-CU6-M1
        :param q: q in WENO5-CU6-M1
        :param case_num: case number.
        :return: dispersion error and shock capturing indicator of Shu-Osher problem.
        """
        log(f'Start Shu-Osher case.{case_num} with cq={cq}, q={q}...')
        if not self.find_shuosher_data(cq, q, case_num):

            alpaca = self.initialize_alpaca(cq, q, ic=40, dim=1, min_time_step='1e-8')
            alpaca.build_workfolder(case_num)
            alpaca.run_alpaca(case_num, cpu_num=ShuBase200.cpu_num, inputfile_name=ShuBase200.inputfile)

            results_folder = os.path.join(OC.case_folder, f"case.{str(case_num)}/{ShuBase200.inputfile[:-4]}/domain")
            shu = ShuOsher(
                results_folder=results_folder, 
                result_filename='data_1.800*.h5', 
                git=False, 
                plot=True
            )
            self.disper, self.disper_raw, self.state_disper = shu.objective_disper(
                key='density', 
                ord=2, 
                gradient_diff=ShuBase200.gradient_diff
            )
            self.shock, self.shock_raw, self.state_shock = shu.objective_shock(key='density')

        log(f'Shu-Osher problem dispersion error: {self.disper} with cq={cq} and q={q}')
        log(f'Shu-Osher problem raw jump error: {self.shock} with cq={cq} and q={q}')

        append_to_csv(f'{OC.case_folder}/shuosher_runtime_samples.csv',
                    [case_num, cq, q, self.disper, self.disper_raw, self.state_disper, self.shock, self.shock_raw, self.state_shock])


    def find_tgv_data(self, cq, q, case_num):
        for file in TGV.training_data:
            index_tgv = None
            data_sheet = read_from_csv(file)
            index_tgv = np.array(np.where((data_sheet[:, 1] == str(cq)) & (data_sheet[:, 2] == str(q))))
            if index_tgv.size != 0:
                index_tgv = index_tgv[0, 0]
                self.turb = data_sheet[index_tgv, 3].astype(float)
                self.turb_raw = data_sheet[index_tgv, 4].astype(float)
                self.state_tgv = data_sheet[index_tgv, 5]
                alpaca = AlpacaBuilder()
                alpaca.build_workfolder(case_num)
                log("Found in database!")
                # time.sleep(0.5)
                return True
        return False

    def calculate_tgv_error(self, cq, q, case_num):
        r"""
        Run TGV simulation and get the error.

        The objective can be directly obtained from a csv data file which is obtained from previously
        simulated results by fixing the random seed, which could save a lot of time.

        :param cq: cq in WENO5-CU6-M1
        :param q: q in WENO5-CU6-M1
        :param case_num: case number.
        :return: TGV error
        """
        log(f'Start TGV case.{case_num} with cq={cq}, q={q}...')
        if not self.find_tgv_data(cq, q, case_num):
            alpaca = self.initialize_alpaca(cq, q, ic=32, dim=3)
            alpaca.build_workfolder(case_num)
            alpaca.run_alpaca(case_num, cpu_num=TGV.cpu_num, inputfile_name=TGV.inputfile)

            results_folder = os.path.join(OC.case_folder, f"case.{str(case_num)}/{TGV.inputfile[:-4]}/domain")
            tke = TaylorGreenVortex(results_folder=results_folder,
                                    result_filename='data_20.00*.h5',
                                    git=False,
                                    plot=True,)

            self.turb, self.state_tgv, self.turb_raw = tke.objective_spectrum()

        log(f'TGV original error: {self.turb} with cq={cq} and q={q}')

        append_to_csv(f'{OC.case_folder}/tgv_runtime_samples.csv',
                    [case_num, cq, q, self.turb, self.turb_raw, self.state_tgv])

