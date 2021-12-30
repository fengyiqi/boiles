#!/usr/bin/env python3

from mytools.objective.sodshocktube import SodShockTube
from mytools.objective.tgv import TaylorGreenVortex
from botorch.test_functions.base import MultiObjectiveTestProblem
from typing import Optional
from mytools.utils import *
from mytools.config.opt_config import *
from mytools.config.opt_problems import OP
import time


class SodShockTube(MultiObjectiveTestProblem):
    r"""Two objective problem composed of the Shu-Osher problem and Taylor-Green Vortex.

    ShuOsher:

        maximize gradient near jump but minimize gradient difference along continuous region

    TGV:

        log-scale difference between kinetic energy ans -5/3 principle

    """
    dim = OC.num_inputs
    num_objectives = OC.num_outputs
    _bounds = OC.bounds
    _ref_point = OP.ref_point

    def __init__(self, case_num, noise_std: Optional[float] = None, negate: bool = False) -> None:
        r"""Constructor for ShuOsher-TGV.

        Arguments:
            noise_std: Standard deviation of the observation noise.
            negate: If True, negate the objectives.
        """
        self.sod_case_num = case_num  # count the number of simulated shu case
        """
        When calculating the hypervolume the objective will be evaluated repeatedly, which is
        impossible for CFD solver. Three dictionaries are defined here to store the obtained objectives indexed
        by 'cq, q' string.
        """
        self.sod_dissipation_dic = {}

        super().__init__(noise_std=noise_std, negate=negate)

    def evaluate_true(self, X):
        r"""
        An abstract method of the base class
        """
        return NotImplemented

    def sod_error(self, x):
        r"""
        Get Shu-Osher problem error.
        :param x: Tensor([cq, q])
        :return: Shu-Osher dispersion error and shock capturing indicator
        """
        cq = int(x[0] * OC.intervals['cq'])
        q = int(x[1] * OC.intervals['q'])

        # find if the result based on current cq and q has been evaluated.
        if self.sod_dissipation_dic.get(f'{str(cq)}, {str(q)}', None) is None:
            # when debugging, uses an easy to evaluate function, else, uses the Shu-Osher problem.
            if OC.debug:
                sod_dissipation = np.cos(np.pi * (cq + q) / 2000) * 10
            else:
                sod_dissipation = sod_error(cq, q, self.sod_case_num)

            self.sod_dissipation_dic[f'{str(cq)}, {str(q)}'] = sod_dissipation

            self.sod_case_num += 1
        else:
            sod_dissipation = self.sod_dissipation_dic[f'{str(cq)}, {str(q)}']

        return sod_dissipation


def initialize_alpaca(cq: int,
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


def sod_error(cq, q, case_num):
    r"""
    Run Shu-Osher problem simulation and get dispersion error or shock capturing indicator.

    Two objectives can be directly obtained from a csv data file which is obtained from
    previously simulated results by fixing the random seed, which could save a lot of time.

    :param cq: cq in WENO5-CU6-M1
    :param q: q in WENO5-CU6-M1
    :param case_num: case number.
    :return: dispersion error and shock capturing indicator of Shu-Osher problem.
    """
    log(f'start Sod case.{case_num} with cq={cq}, q={q}...')
    for file in CC.presaved_data['sod']:
        rows_sod = read_from_csv(file)
        index_sod = np.array(np.where((rows_sod[:, 1] == str(cq)) & (rows_sod[:, 2] == str(q))))
        if index_sod.size != 0:
            index_sod = index_sod[0, 0]
            sod_dissipation_raw = rows_sod[index_sod, 4].astype(np.float)
            sod_dissipation = sod_dissipation_raw if sod_dissipation_raw <= CC.error_upper_bound['dissipation'] else \
            CC.error_upper_bound['dissipation']
            state_sod = rows_sod[index_sod, 5]

            alpaca = AlpacaBuilder()
            alpaca.build_workfolder(case_num)
            log("Found in database!")
            time.sleep(1)
            break

    if index_sod.size == 0:
        alpaca = initialize_alpaca(cq, q, ic=100, dim=1)
        alpaca.build_workfolder(case_num)
        alpaca.run_alpaca(case_num, cpu_num=OC.processors['sod'], inputfile_name=sod_inputfile)

        results_folder = os.path.join(CC.case_folder, f"case.{str(case_num)}/{CC.inputfiles['sod'][:-4]}/domain")
        sod = SodShockTube(results_folder=results_folder,
                           result_filename='data_0.200*.h5',
                           git=False,
                           plot=True, )
        sod_dissipation, state_sod, sod_dissipation_raw = sod.objective_dissipation()

    log(f'Shu-Osher problem dispersion error: {sod_dissipation} with cq={cq} and q={q}')

    append_to_csv(f'{CC.case_folder}/sod_runtime_samples.csv',
                  [case_num, cq, q, sod_dissipation, sod_dissipation_raw, state_sod])
    return sod_dissipation


def tgv_error(cq, q, case_num):
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
    for file in CC.presaved_data['tgv']:

        rows_tgv = read_from_csv(file)
        index_tgv = np.array(np.where((rows_tgv[:, 1] == str(cq)) & (rows_tgv[:, 2] == str(q))))
        if index_tgv.size != 0:
            index_tgv = index_tgv[0, 0]
            # error_tgv = rows_tgv[index_tgv, 3].astype(np.float)
            error_tgv_raw = rows_tgv[index_tgv, 4].astype(np.float)
            error_tgv = error_tgv_raw if error_tgv_raw <= CC.error_upper_bound['tgv'] else CC.error_upper_bound['tgv']
            state_tgv = rows_tgv[index_tgv, 5]
            alpaca = AlpacaBuilder()
            alpaca.build_workfolder(case_num)
            log("Found in database!")
            time.sleep(2)
            break
    if index_tgv.size == 0:
        alpaca = initialize_alpaca(cq, q, ic=16, dim=3)
        alpaca.build_workfolder(case_num)
        alpaca.run_alpaca(case_num, cpu_num=CC.processors['tgv'], inputfile_name=tgv_inputfile)

        results_folder = os.path.join(CC.case_folder, f"case.{str(case_num)}/{CC.inputfiles['tgv'][:-4]}/domain")
        tke = TaylorGreenVortex(results_folder=results_folder,
                                result_filename='data_20.00*.h5',
                                git=False,
                                plot=True, )

        error_tgv, state_tgv, error_tgv_raw = tke.objective_spectrum()

    log(f'TGV original error: {error_tgv} with cq={cq} and q={q}')

    append_to_csv(f'{CC.case_folder}/tgv_runtime_samples.csv',
                  [case_num, cq, q, error_tgv, error_tgv_raw, state_tgv])
    return error_tgv
