from ..case_base import CaseBase


class Sod60(CaseBase):
    inputfile = "sod_60.xml"
    training_data = [
        # 'data/sod_60_step1_roe_teno_1234.csv'
    ]
    # si_threshold = 2.8
    cpu_num = 1
    # ref_data = 'data/shu_aer_0.2/domain/data_1.800000.h5'
    gradient_diff = True
