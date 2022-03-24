#!/usr/bin/env python3


class OptConfiguration:
    r"""
    A class that defines the optimization configuration. This is a singleton class. If user needs to update these
    configurations, please import this class at the beginning of the script and assign corresponding configurations.
    One is required to check all these setting before running the script.

    seed: the random seed which is used for generating initial samples;
    training_samples: the number of the training samples;
    opt_iteration: the number of the loops for optimization;
    dim_inputs: dimension of parameter space;
    dim_outputs: dimension of objective space;
    opt_bounds: this bound is directly used for optimization if one wants a discrete parameter domain. One can imagine 
                that the parameter domain is spaced by "opt_bounds" grids. The parameter domain for the real function 
                is defined by "fun_bounds", see below. This bound can be calculated by the range of fun_bounds divided 
                by increment;
    fun_bounds: the true bounds of the function. It can be calculated by increment * opt_bounds;
    increment: the step size of each parameter;
    case_folder: defines a folder where all the data are stored. Currently it is only valid for ALPACA, see how the
                 objective is calculated in mytools/test_problems/shu_tgv.py
    debug: debug mode flag;

    max_iteration: iteration number limitation;
    lr: learning rate;
    disp: flag if display the

    discrete: define the configuration of the discrete assistant. The discrete assistant is designed for getting rid of
              repeat evaluation on a same sample, which often occur if your parameter domain is discrete.
              
              activate: flag if this assistant is activated;
              method: option "tree" and "constrained". "constrained" is recommended;
              plot: if plot the figures during optimization when using this discrete assistant;
              max_level: the depth of the tree. Only valid for method "tree";
              counter: a counter for plot.
    """
    seed = 1234
    training_samples = 30
    opt_iterations = 30
    dim_inputs = 2
    dim_outputs = 1
    opt_bounds = [(0, 100), (0, 100)]
    fun_bounds = [(50, 250), (0, 100)]
    increment = (2, 1)
    case_folder = 'runtime_data'
    debug = False

    max_iteration = 1000000
    lr = 0.001
    disp = False

    discrete = {
        "activate": False,
        "method": "constrained",  # selection from "tree" and "constrained"
        "plot": True,
        "max_level": 6,  # only valid if method is tree.
        "counter": 1
    }

    # Singleton mode
    def __new__(cls):
        if not hasattr(cls, 'instance'):
            cls.instance = super().__new__(cls)
        return cls.instance


OC = OptConfiguration
