#!/usr/bin/env python3

from ..utils import log, append_to_csv
import numpy as np
import random
from typing import Any, Callable, Dict, List, Optional, Tuple, Union
from ..config.opt_config import OC

display = False


class TestFunction:
    def __init__(self, name: str, bound, step, int_valued=True, noise=False, scale=None):
        self.name = name
        self.bound = bound
        self.step = step
        self.int_valued = int_valued
        self.noise = noise
        self.scale = scale

    def outputs(self, y, noise=False, scale=None):
        if noise:
            y += random.normalvariate(0, scale)
        return y


class TestFunction2D(TestFunction):

    def __init__(self,
                 name: str,
                 bound: List[List] = OC.fun_bounds,
                 step: List[Tuple] = OC.increment,
                 int_valued: bool = True,
                 noise: bool = False,
                 scale: float = 0.1):
        super(TestFunction2D, self).__init__(name, bound, step, int_valued, noise, scale)

        if self.name.lower() == "branin":
            self.function = self.Branin
        if self.name.lower() == "himmelblau":
            self.function = self.Himmelblau
        if self.name.lower() == "goldstein":
            self.function = self.Goldstein

    def Branin(self, X):
        name = "branin"
        x1, x2 = self.inputs(X, self.bound, self.step)

        y = (x2 - 5.1 / (4 * np.pi ** 2) * x1 ** 2 + 5 * x1 / np.pi - 6) ** 2
        y += 10 * (1 - 1 / (8 * np.pi)) * np.cos(x1) + 10
        y = self.outputs(y, noise=self.noise, scale=self.scale)

        if isinstance(X, dict) or self.int_valued:
            if OC.discrete["activate"]:
                append_to_csv(f"discrete/data.csv", [X["x1"], X["x2"], y])
            return {f"{name}": (y, 0.0)}
        else:
            return y

    def Himmelblau(self, X):
        name = "himmelblau"
        x1, x2 = self.inputs(X, self.bound, self.step)

        y = (x1 ** 2 + x2 - 11) ** 2 + (x1 + x2 ** 2 - 7) ** 2
        y = self.outputs(y, noise=self.noise, scale=self.scale)

        if isinstance(X, dict) or self.int_valued:
            if OC.discrete["activate"]:
                append_to_csv(f"data.csv", [X["x1"], X["x2"], y])
            return {f"{name}": (y, 0.0)}
        else:
            return y

    def Goldstein(self, X):
        name = "goldstein"
        x1, x2 = self.inputs(X, self.bound, self.step)

        y = (1 + (x1 + x2 + 1) ** 2 * (19 - 14 * x1 + 3 * x1 ** 2 - 14 * x2 + 6 * x1 * x2 + 3 * x2 ** 2))
        y *= (30 + (2 * x1 - 3 * x2) ** 2 * (18 - 32 * x1 + 12 * x1 ** 2 + 48 * x2 - 36 * x1 * x2 + 27 * x2 ** 2))
        y = self.outputs(y, noise=self.noise, scale=self.scale)

        if isinstance(X, dict) or self.int_valued:
            if OC.discrete["activate"]:
                append_to_csv(f"data.csv", [X["x1"], X["x2"], y])
            return {f"{name}": (y, 0.0)}
        else:
            return y

    def inputs(self, X, bound: list, step: list):
        x1_bound = bound[0]
        x2_bound = bound[1]
        # X is dict or int_valued only for optimization
        if isinstance(X, dict) or self.int_valued:
            x1, x2 = X["x1"] * step[0] + x1_bound[0], X["x2"] * step[1] + x2_bound[0]
            if display:
                log(f"x1: {round(x1, 1)} ({X['x1']})\t | x2: {round(x2, 1)} ({X['x2']})")
        else:
            x1, x2 = X[0], X[1]
        return x1, x2


class TestFunction3D(TestFunction):
    def __init__(self,
                 name: str,
                 bound: List[List] = OC.fun_bounds,
                 step: List[Tuple] = OC.increment,
                 int_valued: bool = True,
                 noise: bool = False,
                 scale: float = 0.1):
        super(TestFunction3D, self).__init__(name, bound, step, int_valued, noise, scale)
        if self.name.lower() == "branin3d":
            self.function = self.Branin3D

    def Branin3D(self, X):
        name = "branin3d"
        x1, x2, x3 = self.inputs(X, self.bound, self.step)
        y = (x2 - 5.1 / (4 * np.pi ** 2) * x1 ** 2 + 5 * x1 / np.pi - 6) ** 2
        y += 10 * (1 - 1 / (8 * np.pi)) * np.cos(x1) + 10 + 2 * x3**2
        y = self.outputs(y, noise=self.noise, scale=self.scale)

        if isinstance(X, dict) or self.int_valued:
            if OC.discrete["activate"]:
                append_to_csv(f"discrete/data.csv", [X["x1"], X["x2"], X["x3"], y])
            return {f"{name}": (y, 0.0)}
        else:
            return y

    def inputs(self, X, bound: list, step: list):
        x1_bound = bound[0]
        x2_bound = bound[1]
        x3_bound = bound[2]
        # X is dict or int_valued only for optimization
        if isinstance(X, dict) or self.int_valued:
            x1, x2, x3 = X["x1"] * step[0] + x1_bound[0], X["x2"] * step[1] + x2_bound[0], X["x3"] * step[2] + x3_bound[0]
            # log(f"x1: {round(x1, 1)} ({X['x1']})\t\t | x2: {round(x2, 1)} ({X['x2']})\t\t | x3: {round(x3, 1)} ({X['x3']})")
            log(f"x1: {round(x1, 1):<6} {'(' + str(X['x1']) + ')':<6} |   "
                f"x2: {round(x2, 1):<6} {'(' + str(X['x2']) + ')':<6} |   "
                f"x3: {round(x3, 1):<6} {'(' + str(X['x3']) + ')':<6}")
        else:
            x1, x2, x3 = X[:, 0], X[:, 1], X[:, 2]
        return x1, x2, x3


solutions = {
    "branin": np.array([(-np.pi, 12.275), (np.pi, 2.275), (9.42478, 2.475)]),
    "himmelblau": np.array([(3, 2), (-2.805118, 3.131312), (-3.779310, -3.283186), (3.584428, -1.848126)]),
    "branin3d": np.array([(-3.14, 12.275, 0), (3.14, 2.275, 0), (9.425, 2.475, 0)]),
}