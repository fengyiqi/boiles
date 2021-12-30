#!/usr/bin/env python3

import os
from mytools.solvers.alpaca_builder import AlpacaBuilder
from mytools.utils import log


class AlpacaTeno5(AlpacaBuilder):

    def __init__(self):
        super(AlpacaTeno5, self).__init__()
        self.teno5_parameters_file = os.path.join(self.alpaca_folder,
                                                  "src/user_specifications/teno5opt_parameters.h")

    def set_teno5_parameters(self,
                             d0: float = 0.6,
                             d1: float = 0.3,
                             d2: float = None,
                             Ct: float = 1e-5,
                             C: int = 1,
                             q: int = 6):
        command = "sed -i 's@\ d1_\ =\ .*;@\ d1_\ =\ " + str(d0) + ";@' " + self.teno5_parameters_file
        os.system(self.enter_alpaca_folder + command)
        command = "sed -i 's@\ d2_\ =\ .*;@\ d2_\ =\ " + str(d1) + ";@' " + self.teno5_parameters_file
        os.system(self.enter_alpaca_folder + command)
        command = "sed -i 's@\ CT_\ =\ .*;@\ CT_\ =\ " + str(Ct) + ";@' " + self.teno5_parameters_file
        os.system(self.enter_alpaca_folder + command)
        command = "sed -i 's@\ C_\ =\ .*;@\ C_\ =\ " + str(C) + ";@' " + self.teno5_parameters_file
        os.system(self.enter_alpaca_folder + command)
        command = "sed -i 's@\ q_\ =\ .*;@\ q_\ =\ " + str(q) + ";@' " + self.teno5_parameters_file
        os.system(self.enter_alpaca_folder + command)

        if self.check_settings(f'd1_ = {str(d0)};', self.teno5_parameters_file) and \
           self.check_settings(f'd2_ = {str(d1)};', self.teno5_parameters_file) and \
           self.check_settings(f'CT_ = {str(Ct)};', self.teno5_parameters_file) and \
           self.check_settings(f'C_ = {str(C)};', self.teno5_parameters_file) and \
           self.check_settings(f'q_ = {str(q)};', self.teno5_parameters_file):
            log(f'TENO5 coefficients:\t check!')
            log(f'TENO5 Stencil d1: {d0} | d2: {d1} | d3: {1-d0-d1}')
            log(f'              CT: {Ct} | C: {C}   | q: {q}')
        else:
            raise ValueError('TENO5 coefficients error')

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