from .case_base import CaseBase
from .utils import get_ref_point

class TGV(CaseBase):
    name = "tgv"
    inputfile = "tgv_64_gas.xml"
    start_wn = 3
    training_data = [
        'data/m1_tgv_seed_1234.csv',
    ]
    lowest_error_from_initial = 1.48
    highest_error_from_initial = 9.28

    cpu_num = 4

    ref_point = get_ref_point(lowest_error_from_initial, highest_error_from_initial, 0.5)

    lengthscale_prior = False

