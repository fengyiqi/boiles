#!/usr/bin/env python3

from ..objective.sodshocktube import SodShockTube
from ..objective.tgv import TaylorGreenVortex
from botorch.test_functions.base import MultiObjectiveTestProblem
from typing import Optional
from ..utils import *
from ..config.opt_config import *
# from ..test_cases import *
import time
from ..solvers.alpaca_teno5sensor import AlpacaTeno5Sensor
from ..solvers.solver_config import SC

alpaca_loc = SC.solver_folder


class SodTGVSensor(MultiObjectiveTestProblem):
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
        self.sod_case_num = case_num  # count the number of simulated shu case
        self.tgv_case_num = case_num
        """
        When calculating the hypervolume the objective will be evaluated repeatedly, which is
        impossible for CFD solver. Three dictionaries are defined here to store the obtained objectives indexed
        by 'cq, q' string.
        """
        self.tgv_dict = {}
        self.sod_disper_dic = {}
        self.sod_shock_dic = {}

        super().__init__(noise_std=noise_std, negate=negate)

    def evaluate_true(self, X):
        r"""
        An abstract method of the base class
        """
        return NotImplemented

    def tgv_error(self, x):
        r"""
        Get TGV error.
        :param x: Tensor([cq, q])
        :return: tgv error
        """
        st = x[0]
        # print(f"-----------\n{self.tgv_dict}\n-----------------")
        # find if the result based on current cq and q has been evaluated.
        if self.tgv_dict.get(f'{format(st, ".9f")}', None) is None:
            # when debugging, uses an easy to evaluate function, else, uses the tgv

            if OC.debug:
                error_tgv = np.cos(np.pi * st * 2000) * 20
            else:
                error_tgv = tgv_error(st, self.tgv_case_num)
                # append_to_csv(f"plotly/{TC.function}.csv", [st, error_tgv])
            self.tgv_dict[f'{format(st, ".9f")}'] = error_tgv
            self.tgv_case_num += 1
        else:
            log("tgv constructing pareto from dic")
            error_tgv = self.tgv_dict[f'{format(st, ".9f")}']

        return error_tgv

    def sod_error(self, x):
        r"""
        Get Shu-Osher problem error.
        :param x: Tensor([cq, q])
        :return: Shu-Osher dispersion error and shock capturing indicator
        """
        st = x[0]
        # print(f"-----------\n{self.sod_disper_dic}\n-----------------")
        # find if the result based on current cq and q has been evaluated.
        if self.sod_disper_dic.get(f'{format(st, ".9f")}', None) is None:
            # when debugging, uses an easy to evaluate function, else, uses the Shu-Osher problem.
            if OC.debug:
                sod_disper, sod_shock = np.cos(np.pi * st * 2000) * 10, np.sin(np.pi * st * 200) * 30
            else:
                sod_disper, sod_shock = sod_error(st, self.sod_case_num)

            self.sod_disper_dic[f'{format(st, ".9f")}'] = sod_disper
            self.sod_shock_dic[f'{format(st, ".9f")}'] = sod_shock
            self.sod_case_num += 1
        else:
            log("sod constructing pareto from dic")
            sod_disper = self.sod_disper_dic[f'{format(st, ".9f")}']
            sod_shock = self.sod_shock_dic[f'{format(st, ".9f")}']

        return sod_disper, sod_shock


def initialize_alpacateno5sensor(parameters: torch.tensor,
                                 ic: int,
                                 dim: int,
                                 min_time_step: str = "std::numeric_limits<double>::epsilon()"):
    r"""
    Initialize ALPACA, including set parameters, cmake ALPACA according to the given dimension
    and make ALPACA
    """
    # parameters = parameters.tolist()
    alpaca = AlpacaTeno5Sensor()
    alpaca.set_reconstruction_stencil("TENO5SENSOR")
    alpaca.set_limit_end_time("true")
    alpaca.set_ic(ic)
    alpaca.set_minimum_time_step_size(min_time_step)
    alpaca.set_teno5sensor_parameters(parameters)
    alpaca.cmake_alpaca(dimension=dim)
    alpaca.compile_alpaca()

    return alpaca


def sod_error(para, case_num):
    r"""
    Run Shu-Osher problem simulation and get dispersion error or shock capturing indicator.

    Two objectives can be directly obtained from a csv data file which is obtained from
    previously simulated results by fixing the random seed, which could save a lot of time.

    :param cq: cq in WENO5-CU6-M1
    :param q: q in WENO5-CU6-M1
    :param case_num: case number.
    :return: dispersion error and shock capturing indicator of Shu-Osher problem.
    """

    st = para
    index_sod = np.array([])
    sod_inputfile = OP.test_cases[0].inputfile
    log(f'Start Sod case.{case_num} with st={st}')
    for file in OP.test_cases[0].training_data:
        rows_sod = read_from_csv(file)
        formatted_st = np.array([format(v, ".9f") for v in rows_sod[:, 1].astype(float)])
        index_sod = np.array(np.where(formatted_st == str(format(st, ".9f"))))
        if index_sod.size != 0:
            index_sod = index_sod[0, 0]
            sod_disper = rows_sod[index_sod, 2].astype(float)
            sod_disper_raw = rows_sod[index_sod, 3].astype(float)
            state_disper = rows_sod[index_sod, 4]

            # sod_shock = rows_sod[index_sod, 5].astype(float)
            # sod_shock_raw = rows_sod[index_sod, 6].astype(float)
            # state_shock = rows_sod[index_sod, 7]
            # alpaca = AlpacaTeno5Sensor()
            # alpaca.build_workfolder(case_num)
            log("Found in database!")
            # time.sleep(0.5)
            break

    if index_sod.size == 0:
        alpaca = initialize_alpacateno5sensor(para, ic=60, dim=1)
        alpaca.build_workfolder(case_num)
        alpaca.run_alpaca(case_num, cpu_num=OP.test_cases[0].cpu_num, inputfile_name=sod_inputfile)

        results_folder = os.path.join(OC.case_folder, f"case.{str(case_num)}/{sod_inputfile[:-4]}/domain")
        sod = SodShockTube(results_folder=results_folder,
                           git=False,
                           plot=True, )
        sod_disper, sod_disper_raw, state_disper = sod.objective_sum_disper()
        # sod_shock, sod_shock_raw, state_shock = sod.objective_shock()

    log(f'SodShockTube problem dispersion error: {sod_disper}')
    # log(f'SodShockTube problem shock error: {sod_shock}')

    append_to_csv(f'{OC.case_folder}/sod_runtime_samples.csv',
                  [case_num, st, sod_disper, sod_disper_raw, state_disper])
    # print(type(shu_disper), type(shu_shock))
    # print(shu_disper, shu_shock)
    return sod_disper, None


def tgv_error(para, case_num):
    r"""
    Run TGV simulation and get the error.

    The objective can be directly obtained from a csv data file which is obtained from previously
    simulated results by fixing the random seed, which could save a lot of time.

    :param cq: cq in WENO5-CU6-M1
    :param q: q in WENO5-CU6-M1
    :param case_num: case number.
    :return: TGV error
    """
    st = para
    index_tgv = np.array([])
    log(f'Start TGV case.{case_num} with st={st}')
    for file in OP.test_cases[-1].training_data:
        rows_tgv = read_from_csv(file)
        formatted_st = np.array([format(v, ".9f") for v in rows_tgv[:, 1].astype(float)])
        index_tgv = np.array(np.where(formatted_st == str(format(st, ".9f"))))
        if index_tgv.size != 0:
            index_tgv = index_tgv[0, 0]
            error_tgv = rows_tgv[index_tgv, 2].astype(float)
            error_tgv_raw = rows_tgv[index_tgv, 3].astype(float)
            state_tgv = rows_tgv[index_tgv, 4]
            alpaca = AlpacaTeno5Sensor()
            alpaca.build_workfolder(case_num)
            log("Found in database!")
            break
    if index_tgv.size == 0:
        alpaca = initialize_alpacateno5sensor(para, ic=16, dim=3)
        alpaca.build_workfolder(case_num)
        alpaca.run_alpaca(case_num, cpu_num=OP.test_cases[-1].cpu_num, inputfile_name=OP.test_cases[-1].inputfile)

        results_folder = os.path.join(OC.case_folder, f"case.{str(case_num)}/{OP.test_cases[-1].inputfile[:-4]}/domain")
        tke = TaylorGreenVortex(results_folder=results_folder,
                                result_filename='data_20.00*.h5',
                                git=False,
                                plot=True,)

        error_tgv, state_tgv, error_tgv_raw = tke.objective_spectrum()

    log(f'TGV original error: {error_tgv}')

    append_to_csv(f'{OC.case_folder}/tgv_runtime_samples.csv',
                  [case_num, st, error_tgv, error_tgv_raw, state_tgv])
    return error_tgv
