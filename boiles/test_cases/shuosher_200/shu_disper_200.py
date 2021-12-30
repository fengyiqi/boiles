from ..utils import *
from .shu_base_200 import ShuBase200


class ShuDisper200(ShuBase200):
    name = "disper"
    lowest_error_from_initial = 15.27
    highest_error_from_initial = 62.52
    ref_point = get_ref_point(lowest_error_from_initial, highest_error_from_initial, 0.5)
    # sensitivity_control = "prior_distribution"
    gradient_diff = True
