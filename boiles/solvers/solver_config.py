import os


class SolverConfiguration:
    r"""
    A class contains basic information of the Riemann solver. This is a singleton class. If user needs to update
    these configurations, please import this class at the beginning of the script and assign corresponding
    configurations.

    solver_folder: the folder
    """
    # the folder where ALPACA executor exists, typically it is the parent directory
    # of the project.
    solver_folder = os.path.dirname(os.getcwd())
    # if your ALPACA was cloned from gitlab, then here should be 'git'
    alpaca_version = 'aer'
    Riemann = 'hllc'

    def __new__(cls):
        if not hasattr(cls, 'instance'):
            cls.instance = super().__new__(cls)
        return cls.instance


SC = SolverConfiguration
