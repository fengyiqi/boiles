from ..utils import *
from .sod_base_60 import Sod60


class SodShock60(Sod60):
    name = "shock"
    lowest_error_from_initial = 0.0960971851134122
    highest_error_from_initial = 0.133935268052641
    ref_point = get_ref_point(lowest_error_from_initial, highest_error_from_initial, 0.5)
    sensitivity_control = "noise_level"
