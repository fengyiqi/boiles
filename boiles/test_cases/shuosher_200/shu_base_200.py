from ..case_base import CaseBase


class ShuBase200(CaseBase):
    inputfile = "shu_200.xml"
    training_data = [
        'data/m1_shu200_seed_1234.csv'
    ]
    si_threshold = 10.0
    cpu_num = 1
    ref_data = 'data/shu_aer_0.2/domain/data_1.800000.h5'
    gradient_diff = True
