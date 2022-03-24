#!/usr/bin/env python3

import os
from mytools.solvers.alpaca_builder import AlpacaBuilder
from mytools.utils import log


class AlpacaTeno5Sensor(AlpacaBuilder):

    def __init__(self):
        super(AlpacaTeno5Sensor, self).__init__()
        self.teno5sensor_parameters_file = os.path.join(self.alpaca_folder,
                                                        "src/user_specifications/teno5sensor_parameters.h")

    def set_teno5sensor_parameters(self,
                                   st: float):

        command = "sed -i 's@\ ST_\ =\ .*;@\ ST_\ =\ " + str(st) + ";@' " + self.teno5sensor_parameters_file
        os.system(self.enter_alpaca_folder + command)

        if self.check_settings(f'ST_ = {str(st)};', self.teno5sensor_parameters_file):
            log(f'Turbulent sensor threshold:\t{st}\tcheck!')
        else:
            raise ValueError('Sensor threshold error')

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
