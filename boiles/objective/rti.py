#!/usr/bin/env python3

from .simulation2d import Simulation2D


class RTI(Simulation2D):

    def __init__(self,
                 results_folder: str,
                 result_filename: str = 'data_2.00*.h5',
                 git: bool = False,
                 shape: int = (128, 32)
                 ):
        super(RTI, self).__init__(results_folder, result_filename, git=git)
        self.dimension = 2
        self.plot_savepath = results_folder
        if self.result_exit:
            self.shape = shape
            self.result = self.get_results(self.result_path)
