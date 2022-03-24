from ..utils import *
from .sod_base_60 import Sod60

bound = {
    "density": (0.05176, 0.32190),
    "pressure": (0.06333, 0.29718),
    "velocity": (0.08593, 0.90981)
}

class SodDisper60(Sod60):
    training_data = [
        "data/sod_60_sum_roe_st_teno_1234.csv"
    ]

    name = "disper"
    lowest_error_from_initial = 0
    highest_error_from_initial = 3
    ref_point = get_ref_point(lowest_error_from_initial, highest_error_from_initial, 0.5)
    # sensitivity_control = "noise_level"
    # sensitivity_control = "prior_distribution"
    gradient_diff = False

    @staticmethod
    def normalize(value, prop):
        return (value - bound[prop][0]) / (bound[prop][1] - bound[prop][0])
