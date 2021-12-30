from .case_base import CaseBase
from .tgv import TGV


class TGVTENO5(TGV):
    training_data = [
        "data/tgvteno5_st_64_20s_wn3_1234.csv"
    ]

    lowest_error_from_initial = 3.65
    highest_error_from_initial = 8.98

    # mean_prior = True
