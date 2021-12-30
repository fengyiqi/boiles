import os
from mytools.solvers.alpaca_teno5 import AlpacaTeno5


class AlpacaTeno5Con(AlpacaTeno5):

    def __init__(self, index):
        super(AlpacaTeno5, self).__init__()
        self.alpaca_folder = os.path.join(os.getcwd(), f"alpaca_aer{index}")
        self.compile_time_const_file = os.path.join(self.alpaca_folder,
                                                    "src/user_specifications/compile_time_constants.h")
        self.stencil_setup_file = os.path.join(self.alpaca_folder, "src/user_specifications/stencil_setup.h")
        self.teno5_parameters_file = os.path.join(self.alpaca_folder,
                                                          "src/user_specifications/teno5opt_parameters.h")