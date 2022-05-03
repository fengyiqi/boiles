from ..utils import *
from .sod_base_60 import Sod60

# for Roe solver
bound = {
    "density": (0.05176, 0.32190),
    "pressure": (0.06333, 0.29718),
    "velocity": (0.09284, 0.96046)
}
# ord=1
# bound = {
#     "density": (0.23053219900762575, 1.1310494069220383),
#     "pressure": (0.22839516490967557, 1.0751713337853304),
#     "velocity": (0.3548227077672967, 2.6644395892625288)
# }

# for HLLC solver
# bound = {
#     "density": (0.05335, 0.11767),
#     "pressure": (0.06428, 0.13882),
#     "velocity": (0.09383, 0.37454)
# }


class SodDisper60(Sod60):
    training_data = [
        # "data/sod_60_sum_roe_st_teno_1234.csv"
    ]

    name = "disper"
    lowest_error_from_initial = 0
    highest_error_from_initial = 0.6
    ref_point = get_ref_point(lowest_error_from_initial, highest_error_from_initial, 0.5)
    # sensitivity_control = "noise_level"
    # sensitivity_control = "prior_distribution"
    gradient_diff = False

    @staticmethod
    def normalize(value, prop):
        return (value - bound[prop][0]) / (bound[prop][1] - bound[prop][0])
