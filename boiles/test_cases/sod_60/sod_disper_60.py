from ..utils import *
from .sod_base_60 import Sod60

bound = {
    "density": (0.048, 0.055),
    "pressure": (0.060, 0.0623),
    "x_velocity": (0.080, 0.092)
}

class SodDisper60(Sod60):
    training_data = [
        "data/sod_60_sum_roe_st_teno_1234.csv"
    ]

    name = "disper"
    lowest_error_from_initial = 0
    highest_error_from_initial = 3
    ref_point = get_ref_point(lowest_error_from_initial, highest_error_from_initial, 0.5)
    sensitivity_control = "noise_level"
    # sensitivity_control = "prior_distribution"
    gradient_diff = True

    @staticmethod
    def normalize(value, prop):
        return (value - bound[prop][0]) / (bound[prop][1] - bound[prop][0])
