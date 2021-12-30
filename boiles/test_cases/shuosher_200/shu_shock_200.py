from ..utils import *
from .shu_base_200 import ShuBase200


class ShuShock200(ShuBase200):
    name = "shock"
    lowest_error_from_initial = 0.0260
    highest_error_from_initial = 0.0471
    ref_point = get_ref_point(lowest_error_from_initial, highest_error_from_initial, 0.7)
    # sensitivity_control = "prior_distribution"