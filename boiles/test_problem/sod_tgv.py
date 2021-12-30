#!/usr/bin/env python3

from mytools.objective.sodshocktube import SodShockTube
from mytools.objective.tgv import TaylorGreenVortex
from botorch.test_functions.base import MultiObjectiveTestProblem
from typing import Optional
from mytools.utils import *
from mytools.config.opt_config import *
from mytools.test_cases import *
import time
from mytools.solvers.alpaca_teno5 import AlpacaTeno5
from mytools.solvers.solver_config import SC


alpaca_loc = SC.solver_folder


class SodTGV(MultiObjectiveTestProblem):
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
        self.sod_disper_sum_dic = {}
        self.sod_disper_density = {}
        self.sod_disper_pressure = {}
        self.sod_disper_velocity = {}

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
        # print(x)
        # x_list = x.tolist()
        # d1 = x['x1'] * TC.step[0] + TC.func_bounds[0][0]
        # Cq = x['x2'] * TC.step[1] + TC.func_bounds[1][0]
        # q = x['x3'] * TC.step[2] + TC.func_bounds[2][0]
        d1 = x['x1'] / 100
        Cq = x['x2'] * 1000
        q = x['x3']
        # print(f"-----------\n{self.tgv_dict}\n-----------------")
        # find if the result based on current cq and q has been evaluated.
        if self.tgv_dict.get(f'{str(d1)}, {str(Cq)}, {str(q)}', None) is None:
            # when debugging, uses an easy to evaluate function, else, uses the tgv
            if OC.debug:
                error_tgv = np.cos(np.pi * (d1 + Cq + q) / 2000) * 20
            else:
                error_tgv = tgv_error([d1, Cq, q], self.tgv_case_num)
                append_to_csv(f"discrete/data.csv", [x["x1"], x["x2"], x["x3"], error_tgv])
            self.tgv_dict[f'{str(d1)}, {str(Cq)}, {str(q)}'] = error_tgv
            self.tgv_case_num += 1
        else:
            error_tgv = self.tgv_dict[f'{str(d1)}, {str(Cq)}, {str(q)}']

        return error_tgv

    def sod_error(self, x):
        r"""
        Get Shu-Osher problem error.
        :param x: Tensor([cq, q])
        :return: Shu-Osher dispersion error and shock capturing indicator
        """
        x_list = x.tolist()
        d1 = x_list[0]/100
        Cq = x_list[1]
        q = x_list[2]
        # print(f"-----------\n{self.sod_disper_dic}\n-----------------")
        # find if the result based on current cq and q has been evaluated.
        if self.sod_shock_dic.get(f'{str(d1)}, {str(Cq)}, {str(q)}', None) is None:
            # when debugging, uses an easy to evaluate function, else, uses the Shu-Osher problem.
            if OC.debug:
                sod_disper, sod_shock = np.cos(np.pi * (d1 + Cq + q) / 200) * 10, np.sin(np.pi * (d1 * Cq + q) / 200) * 30
            else:
                sod_disper, sod_shock = sod_error([d1, Cq, q], self.sod_case_num)

            self.sod_disper_dic[f'{str(d1)}, {str(Cq)}, {str(q)}'] = sod_disper
            self.sod_shock_dic[f'{str(d1)}, {str(Cq)}, {str(q)}'] = sod_shock
            self.sod_case_num += 1
        else:
            sod_disper = self.sod_disper_dic[f'{str(d1)}, {str(Cq)}, {str(q)}']
            sod_shock = self.sod_shock_dic[f'{str(d1)}, {str(Cq)}, {str(q)}']

        return sod_disper, sod_shock

    def sod_disper_sum_error(self, x, props=None):
        r"""
        Get Shu-Osher problem error.
        :param x: Tensor([cq, q])
        :param props: ...
        :return: Shu-Osher dispersion error and shock capturing indicator
        """
        # x_list = x.tolist()
        if props is None:
            props = ["density", "pressure", "x_velocity"]
        d1 = x['x1']/100
        Cq = x['x2']
        q = x['x3']
        # print(f"-----------\n{self.sod_disper_dic}\n-----------------")
        # find if the result based on current cq and q has been evaluated.
        if self.sod_disper_sum_dic.get(f'{str(d1)}, {str(Cq)}, {str(q)}', None) is None:
            # when debugging, uses an easy to evaluate function, else, uses the Shu-Osher problem.
            if OC.debug:
                sod_disper, sod_shock = np.cos(np.pi * (d1 + Cq + q) / 200) * 10, np.sin(np.pi * (d1 * Cq + q) / 200) * 30
            else:
                sod_disper = sod_disper_sum_error([d1, Cq, q], self.sod_case_num, props=props)
                if OC.discrete["activate"]:
                    append_to_csv(f"discrete/data.csv", [x["x1"], x["x2"], x["x3"], sod_disper])
            self.sod_disper_sum_dic[f'{str(d1)}, {str(Cq)}, {str(q)}'] = sod_disper
            self.sod_case_num += 1
        else:
            sod_disper = self.sod_disper_sum_dic[f'{str(d1)}, {str(Cq)}, {str(q)}']

        return sod_disper

    def sod_disper_mul(self, x):
        r"""
        Get Shu-Osher problem error.
        :param x: Tensor([cq, q])
        :return: Shu-Osher dispersion error and shock capturing indicator
        """
        x_list = x.tolist()
        d1 = x_list[0]/100
        Cq = x_list[1]
        q = x_list[2]
        # print(f"-----------\n{self.sod_disper_dic}\n-----------------")
        # find if the result based on current cq and q has been evaluated.
        if self.sod_disper_density.get(f'{str(d1)}, {str(Cq)}, {str(q)}', None) is None:
            # when debugging, uses an easy to evaluate function, else, uses the Shu-Osher problem.
            if OC.debug:
                errors = (1, 1, 1)
            else:
                errors = sod_disper_mul_error([d1, Cq, q], self.sod_case_num)
            disper_density = errors[0]
            disper_pressure = errors[1]
            disper_velocity = errors[2]

            self.sod_disper_density[f'{str(d1)}, {str(Cq)}, {str(q)}'] = disper_density
            self.sod_disper_pressure[f'{str(d1)}, {str(Cq)}, {str(q)}'] = disper_pressure
            self.sod_disper_velocity[f'{str(d1)}, {str(Cq)}, {str(q)}'] = disper_velocity

            self.sod_case_num += 1
        else:
            disper_density = self.sod_disper_density[f'{str(d1)}, {str(Cq)}, {str(q)}']
            disper_pressure = self.sod_disper_pressure[f'{str(d1)}, {str(Cq)}, {str(q)}']
            disper_velocity = self.sod_disper_velocity[f'{str(d1)}, {str(Cq)}, {str(q)}']
        return disper_density, disper_pressure, disper_velocity


def initialize_alpacateno5(parameters: torch.tensor,
                           ic: int,
                           dim: int,
                           case: str,
                           min_time_step: str = "std::numeric_limits<double>::epsilon()"):
    r"""
    Initialize ALPACA, including set parameters, cmake ALPACA according to the given dimension
    and make ALPACA
    """
    # parameters = parameters.tolist()
    alpaca = AlpacaTeno5()
    alpaca.set_reconstruction_stencil("TENO5OPT")
    alpaca.set_limit_end_time("true")
    alpaca.set_ic(ic)
    alpaca.set_minimum_time_step_size(min_time_step)
    if case == "tgv":
        alpaca.set_teno5_parameters(d0=0.95-parameters[0],
                                    d1=parameters[0],
                                    Ct=1e-5,
                                    C=parameters[1],
                                    q=parameters[2])
    if case == "sod":
        alpaca.set_teno5_parameters(d0=0.55,
                                    d1=parameters[0],
                                    Ct=1e-5,
                                    C=parameters[1],
                                    q=parameters[2])
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

    d1 = para[0]
    Cq = para[1]
    q = para[2]
    index_sod = np.array([])
    log(f'Start Sod case.{case_num} with d1={d1}, Cq={Cq} and q={q}')
    for file in OP.test_cases[0].training_data:
        rows_sod = read_from_csv(file)
        # print(rows_sod[:, 1], str(d1), (rows_sod[:, 1] == str(d1)))
        # print(rows_sod[:, 2], str(Cq), (rows_sod[:, 2] == str(Cq)))
        # print(rows_sod[:, 3], str(q), (rows_sod[:, 3] == str(q)))
        index_sod = np.array(np.where((rows_sod[:, 1] == str(d1)) & (rows_sod[:, 2] == str(Cq)) & (rows_sod[:, 3] == str(q))))
        if index_sod.size != 0:
            index_sod = index_sod[0, 0]
            # shu_disper = rows_shu[index_shu, 3].astype(np.float)
            sod_disper = rows_sod[index_sod, 4].astype(float)
            sod_disper_raw = rows_sod[index_sod, 5].astype(float)
            state_disper = rows_sod[index_sod, 6]

            sod_shock = rows_sod[index_sod, 7].astype(float)
            sod_shock_raw = rows_sod[index_sod, 8].astype(float)
            state_shock = rows_sod[index_sod, 9]
            alpaca = AlpacaTeno5()
            alpaca.build_workfolder(case_num)
            log("Found in database!")
            # time.sleep(0.5)
            break

    if index_sod.size == 0:
        alpaca = initialize_alpacateno5(para, ic=60, dim=1, case="sod")
        alpaca.build_workfolder(case_num)
        alpaca.run_alpaca(case_num, cpu_num=OP.test_cases[0].cpu_num, inputfile_name=OP.test_cases[0].inputfile)

        results_folder = os.path.join(OC.case_folder, f"case.{str(case_num)}/{OP.test_cases[0].inputfile[:-4]}/domain")
        sod = SodShockTube(results_folder=results_folder,
                           git=False,
                           plot=True, )
        sod_disper, sod_disper_raw, state_disper = sod.objective_disper()
        sod_shock, sod_shock_raw, state_shock = sod.objective_shock()

    log(f'SodShockTube problem dispersion error: {sod_disper}')
    log(f'SodShockTube problem shock error: {sod_shock}')

    append_to_csv(f'{OC.case_folder}/sod_runtime_samples.csv',
                  [case_num, d1, Cq, q, sod_disper, sod_disper_raw, state_disper, sod_shock, sod_shock_raw, state_shock])
    # print(type(shu_disper), type(shu_shock))
    # print(shu_disper, shu_shock)
    return sod_disper, sod_shock


def sod_disper_sum_error(para, case_num, props):
    r"""
    Run Shu-Osher problem simulation and get dispersion error or shock capturing indicator.

    Two objectives can be directly obtained from a csv data file which is obtained from
    previously simulated results by fixing the random seed, which could save a lot of time.

    :param cq: cq in WENO5-CU6-M1
    :param q: q in WENO5-CU6-M1
    :param case_num: case number.
    :return: dispersion error and shock capturing indicator of Shu-Osher problem.
    """

    d1 = para[0]
    Cq = para[1]
    q = para[2]
    index_sod = np.array([])
    log(f'Start Sod case.{case_num} with d1={d1}, Cq={Cq} and q={q}')
    for file in OP.test_cases[0].training_data:
        rows_sod = read_from_csv(file)
        # print(rows_sod[:, 1], str(d1), (rows_sod[:, 1] == str(d1)))
        # print(rows_sod[:, 2], str(Cq), (rows_sod[:, 2] == str(Cq)))
        # print(rows_sod[:, 3], str(q), (rows_sod[:, 3] == str(q)))
        index_sod = np.array(np.where((rows_sod[:, 1] == str(d1)) & (rows_sod[:, 2] == str(Cq)) & (rows_sod[:, 3] == str(q))))
        if index_sod.size != 0:
            index_sod = index_sod[0, 0]
            # shu_disper = rows_shu[index_shu, 3].astype(np.float)
            sod_disper = rows_sod[index_sod, 4].astype(float)
            sod_disper_raw = rows_sod[index_sod, 5].astype(float)
            state_disper = rows_sod[index_sod, 6]

            alpaca = AlpacaTeno5()
            alpaca.build_workfolder(case_num)
            log("Found in database!")
            # time.sleep(0.5)
            break

    if index_sod.size == 0:
        alpaca = initialize_alpacateno5(para, ic=60, dim=1, case="sod")
        alpaca.build_workfolder(case_num)
        alpaca.run_alpaca(case_num, cpu_num=OP.test_cases[0].cpu_num, inputfile_name=OP.test_cases[0].inputfile)

        results_folder = os.path.join(OC.case_folder, f"case.{str(case_num)}/{OP.test_cases[0].inputfile[:-4]}/domain")
        sod = SodShockTube(results_folder=results_folder,
                           git=False,
                           plot=True, )
        sod_disper, sod_disper_raw, state_disper = sod.objective_sum_disper(props=props)

    log(f'SodShockTube problem dispersion error: {sod_disper}')

    append_to_csv(f'{OC.case_folder}/sod_runtime_samples.csv',
                  [case_num, d1, Cq, q, sod_disper, sod_disper_raw, state_disper])
    # print(type(shu_disper), type(shu_shock))
    # print(shu_disper, shu_shock)
    return sod_disper


def sod_disper_mul_error(para, case_num):
    r"""
    Run Shu-Osher problem simulation and get dispersion error or shock capturing indicator.

    Two objectives can be directly obtained from a csv data file which is obtained from
    previously simulated results by fixing the random seed, which could save a lot of time.

    :param para:
    :param case_num: case number.
    :param props:
    :return: dispersion error and shock capturing indicator of Shu-Osher problem.
    """

    d1 = para[0]
    Cq = para[1]
    q = para[2]
    index_sod = np.array([])
    log(f'Start Sod case.{case_num} with d1={d1}, Cq={Cq} and q={q}')
    for file in OP.test_cases[0].training_data:
        rows_sod = read_from_csv(file)
        # print(rows_sod[:, 1], str(d1), (rows_sod[:, 1] == str(d1)))
        # print(rows_sod[:, 2], str(Cq), (rows_sod[:, 2] == str(Cq)))
        # print(rows_sod[:, 3], str(q), (rows_sod[:, 3] == str(q)))
        index_sod = np.array(np.where((rows_sod[:, 1] == str(d1)) & (rows_sod[:, 2] == str(Cq)) & (rows_sod[:, 3] == str(q))))
        if index_sod.size != 0:
            index_sod = index_sod[0, 0]
            # shu_disper = rows_shu[index_shu, 3].astype(np.float)
            sod_disper_density = rows_sod[index_sod, 4].astype(float)
            sod_disper_density_raw = rows_sod[index_sod, 5].astype(float)
            state_disper_density = rows_sod[index_sod, 6]
            sod_disper_pressure = rows_sod[index_sod, 7].astype(float)
            sod_disper_pressure_raw = rows_sod[index_sod, 8].astype(float)
            state_disper_pressure = rows_sod[index_sod, 9]
            sod_disper_velocity = rows_sod[index_sod, 10].astype(float)
            sod_disper_velocity_raw = rows_sod[index_sod, 11].astype(float)
            state_disper_velocity = rows_sod[index_sod, 12]

            alpaca = AlpacaTeno5()
            alpaca.build_workfolder(case_num)
            log("Found in database!")
            # time.sleep(0.5)
            break

    if index_sod.size == 0:
        alpaca = initialize_alpacateno5(para, ic=60, dim=1, case="sod")
        alpaca.build_workfolder(case_num)
        alpaca.run_alpaca(case_num, cpu_num=OP.test_cases[0].cpu_num, inputfile_name=OP.test_cases[0].inputfile)

        results_folder = os.path.join(OC.case_folder, f"case.{str(case_num)}/{OP.test_cases[0].inputfile[:-4]}/domain")
        sod = SodShockTube(results_folder=results_folder,
                           git=False,
                           plot=True, )
        sod_disper_density, sod_disper_density_raw, state_disper_density = sod.objective_sum_disper(
            props=["density"]
        )
        sod_disper_pressure, sod_disper_pressure_raw, state_disper_pressure = sod.objective_sum_disper(
            props=["pressure"]
        )
        sod_disper_velocity, sod_disper_velocity_raw, state_disper_velocity = sod.objective_sum_disper(
            props=["x_velocity"]
        )
    log(f'SodShockTube problem density dispersion error: {sod_disper_density}')
    log(f'SodShockTube problem pressure dispersion error: {sod_disper_pressure}')
    log(f'SodShockTube problem x_velocity dispersion error: {sod_disper_velocity}')

    append_to_csv(f'{OC.case_folder}/sod_runtime_samples.csv',
                  [
                      case_num,
                      d1,
                      Cq,
                      q,
                      sod_disper_density,
                      sod_disper_density_raw,
                      state_disper_density,
                      sod_disper_pressure,
                      sod_disper_pressure_raw,
                      state_disper_pressure,
                      sod_disper_velocity,
                      sod_disper_velocity_raw,
                      state_disper_velocity
                  ])
    # print(type(shu_disper), type(shu_shock))
    # print(shu_disper, shu_shock)
    return sod_disper_density, sod_disper_pressure, sod_disper_velocity


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
    d1 = para[0]
    Cq = para[1]
    q = para[2]
    index_tgv = np.array([])
    log(f'Start TGV case.{case_num} with d1={d1}, Cq={Cq} and q={q}...')
    for file in OP.test_cases[-1].training_data:

        rows_tgv = read_from_csv(file)
        # print(rows_tgv[:, 3], str(q))
        index_tgv = np.array(np.where((rows_tgv[:, 1] == str(d1)) & (rows_tgv[:, 2] == str(Cq)) & (rows_tgv[:, 3] == str(q))))
        if index_tgv.size != 0:
            index_tgv = index_tgv[0, 0]
            error_tgv = rows_tgv[index_tgv, 4].astype(float)
            error_tgv_raw = rows_tgv[index_tgv, 5].astype(float)
            # error_tgv = error_tgv_raw if error_tgv_raw <= CC.error_upper_bound['tgv'] else CC.error_upper_bound['tgv']
            state_tgv = rows_tgv[index_tgv, 6]
            alpaca = AlpacaTeno5()
            alpaca.build_workfolder(case_num)
            log("Found in database!")
            # time.sleep(1)
            break
    if index_tgv.size == 0:
        alpaca = initialize_alpacateno5(para, ic=16, dim=3, case="tgv")
        alpaca.build_workfolder(case_num)
        alpaca.run_alpaca(case_num, cpu_num=TGV.cpu_num, inputfile_name=OP.test_cases[-1].inputfile)

        results_folder = os.path.join(OC.case_folder, f"case.{str(case_num)}/{OP.test_cases[-1].inputfile[:-4]}/domain")
        tke = TaylorGreenVortex(results_folder=results_folder,
                                result_filename='data_20.00*.h5',
                                git=False,
                                plot=True,)

        error_tgv, state_tgv, error_tgv_raw = tke.objective_spectrum()

    log(f'TGV original error: {error_tgv}')

    append_to_csv(f'{OC.case_folder}/tgv_runtime_samples.csv',
                  [case_num, d1, Cq, q, error_tgv, error_tgv_raw, state_tgv])
    return error_tgv
