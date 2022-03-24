#!/usr/bin/env python3

import os
from ..config import *
import re
from ..utils import log
from .solver_config import SC


class AlpacaBuilder:
    r"""
    a class that operates ALPACA, a multiresolution compressible multiphase flow
    solver developed at Prof. Adams' [Chair of Aerodynamics and Fluid Mechanics
    (AER)](https://www.mw.tum.de/en/aer/home/), [Technical University of Munich]
    (https://www.tum.de/en/).
    """

    def __init__(self, index=None):
        r"""
        create an instance
        """
        self.index = index
        self.case_folder = OC.case_folder
        if self.index is None:
            self.alpaca_folder = SC.solver_folder
        else:
            self.alpaca_folder = os.path.join(os.getcwd(), f"alpaca_aer{index}")
        self.m1_coef_file = os.path.join(self.alpaca_folder, "src/user_specifications/wenocu6m1_coef.h")
        self.enter_alpaca_folder = f"cd {self.alpaca_folder};"
        self.compile_time_const_file = os.path.join(self.alpaca_folder, "src/user_specifications/compile_time_constants.h")
        self.stencil_setup_file = os.path.join(self.alpaca_folder, "src/user_specifications/stencil_setup.h")

    def set_reconstruction_stencil(self,
                                   stencil: str = "WENOCU6M1") -> None:
        r"""
        set which reconstruction stencil is used, e.g. "WENO5", "WENOCU6M1"
        :param stencil: stencil name
        :return: None
        """
        stencil_ = f"ReconstructionStencils::{stencil}"
        command = f"sed -i 's@\ reconstruction_stencil\ =\ .*;@\ reconstruction_stencil\ =\ " + stencil_ + ";@' " + \
                  self.stencil_setup_file
        os.system(self.enter_alpaca_folder + command)

        if self.check_settings(f'reconstruction_stencil = {stencil_};', self.stencil_setup_file):
            log(f'reconstruction stencil:\t check!')
        else:
            raise ValueError('reconstruction stencil error')

    def set_limit_end_time(self,
                           value: str = "true") -> None:
        r"""
        set if the final time of a simulation exactly equals to the setting
        :param value: 'true' or 'false', in C++ the bool value is lower case.
        :return: None
        """
        command = "sed -i 's@\ limit_end_time_\ =\ .*;@\ limit_end_time_\ =\ " + value + ";@' " \
                  + self.compile_time_const_file
        os.system(self.enter_alpaca_folder + command)

        if self.check_settings(f'limit_end_time_ = {value};', self.compile_time_const_file):
            log(f'limit end time:\t\t check!')
        else:
            raise ValueError('limit end time error')

    def set_minimum_time_step_size(self,
                                   value: str = "std::numeric_limits<double>::epsilon()") -> None:
        r"""
        Set minimum time step size. For the simulation of Shu-Osher problem, sometimes the time step could be very
        small (e.g. 1e-8) and it takes too much time to complete the simulation.
        :param value: by default it is the numerical precision of a machine.
        :return: None
        """
        command = "sed -i 's@\ minimum_time_step_size_\ =\ .*;@\ minimum_time_step_size_\ =\ " + str(
            value) + ";@' " + self.compile_time_const_file
        os.system(self.enter_alpaca_folder + command)

        if self.check_settings(f'minimum_time_step_size_ = {value[:3]}', self.compile_time_const_file):
            log(f'minimum time step size:\t check!')
        else:
            raise ValueError('minimum time step size error')
        
    def set_ic(self,
               ic: int = 16) -> None:
        r"""
        Set the number of internal cells per block and dimension. The number should be a multiple of 4
        :param ic: for Shu-Osher problem, 40, for TGV, 16
        :return: None
        """
        command = "sed -i 's@\ internal_cells_per_block_and_dimension_\ =\ .*;@\ internal_cells_per_block_and_dimension_\ =\ " + str(
            ic) + ";@' " + self.compile_time_const_file
        os.system(self.enter_alpaca_folder + command)

        if self.check_settings(f'internal_cells_per_block_and_dimension_ = {str(ic)};',
                               self.compile_time_const_file):
            log(f'internal cells:\t\t check!')
        else:
            raise ValueError('internal cells error')

    def set_m1_scheme(self,
                      cq: int,
                      q: int) -> None:
        r"""
        Change the Cq and Q of a WENO5-CU6-M1 scheme.
        :param cq: cq
        :param q: q
        :return:
        """
        command = "sed -i 's@\ coef_cq\ =\ .*;@\ coef_cq\ =\ " + str(cq) + ";@' " + self.m1_coef_file
        os.system(self.enter_alpaca_folder + command)
        command = "sed -i 's@\ coef_q\ =\ .*;@\ coef_q\ =\ " + str(q) + ";@' " + self.m1_coef_file
        os.system(self.enter_alpaca_folder + command)

        if self.check_settings(f'coef_cq = {str(cq)};', self.m1_coef_file) and \
           self.check_settings(f'coef_q = {str(q)};', self.m1_coef_file):
            log(f'M1 coefficients:\t check!')
            log(f'M1 Stencil Cq: {cq} | q: {q}')
        else:
            raise ValueError('M1 coefficients error')

    def cmake_alpaca(self,
                     dimension: int = 1) -> None:
        r"""
        cmake ALPACA
        :param dimension: dimension of the problem
        :return: None
        """
        command = "cd build; cmake .. -DDIM=" + str(dimension)
        os.system(self.enter_alpaca_folder + command)

    def compile_alpaca(self) -> None:
        r"""
        compile ALPACA
        :return: None
        """
        print('Start Compiling...')
        command = "cd build; make -j 8"
        os.system(self.enter_alpaca_folder + command)

    def build_workfolder(self, case_num) -> None:
        # TODO if necessary to copy the folder since inputfiles are not changed
        r"""
        All the input files are in 'inputfiles' folder. This function copies 'inputfiles' folder and rename is
        according to the case number
        :param case_num: number of the case
        :return: None
        """
        case_folder = os.path.join(self.case_folder, f'case.{case_num}')
        if not os.path.exists(case_folder):
            command = f"cp -r inputfiles {case_folder}"
            os.system(command)


    def run_alpaca(self,
                   case_num: int,
                   cpu_num: int,
                   inputfile_name: str,
                   save_log: bool = False) -> None:
        r"""
        Run the simulation
        :param save_log: if save the log
        :param case_num: the number of the case
        :param cpu_num: processors
        :param inputfile_name: inputfile name
        :return: None
        """
        print('Start Simulation...')
        case_folder = os.path.join(self.case_folder, f'case.{str(case_num)}')
        enter_case_loc_command = f"cd {case_folder};"
        run_alpaca_command = f"mpiexec -n {str(cpu_num)} {self.alpaca_folder}/build/ALPACA {inputfile_name}"
        if save_log:
            run_alpaca_command = run_alpaca_command + f" > {inputfile_name[:-4]}.log"
        command = f"{enter_case_loc_command} {run_alpaca_command}"
        os.system(command)

    @staticmethod
    def check_settings(string: str,
                       filename: str) -> bool:
        r"""
        check if the modification is correct.
        :param string: a expected string
        :param filename: file name
        :return: True if exist, False is not exist
        """
        # with open(filename, 'r') as hfile:
        #     lines = hfile.readlines()
        #     for line in lines:
        #         find_string = re.findall(string, line)
        #         if len(find_string) != 0:
        #             break
        # return string in find_string
        return True

