#!/usr/bin/env python3

from .simulation2d import Simulation2D
import numpy as np


class FreeShearFlow(Simulation2D):

    def __init__(self,
                 results_folder: str,
                 result_filename: str = 'data_1.00*.h5',
                 git: bool = False,
                 ):
        self.dimension = 2
        super(FreeShearFlow, self).__init__(results_folder, result_filename, git=git)
        if self.result_exit:
            self.result = self.get_results(self.result_path)