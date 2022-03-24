#!/usr/bin/env python3

from mytools.utils import *
from mytools.config import *


def initialize_cv():

    data = read_from_csv('runtime_data/tgv_runtime_samples.csv')

    tgv_raw = data[:, 3].reshape(-1, 1).astype(float)
    tgv = standaardize_y(tgv_raw)
    cq_q_raw = data[:, 1:3].astype(int)
    cq_q = unit_x(cq_q_raw)

    if OC.num_outputs == 3:
        data = read_from_csv('runtime_data/shuosher_runtime_samples.csv')

        shu_disper_raw = data[:, 3].reshape(-1, 1).astype(float)
        shu_shock_raw = data[:, 6].reshape(-1, 1).astype(float)

        shu_disper = standaardize_y(shu_disper_raw)
        shu_shock = standaardize_y(shu_shock_raw)

        obj = np.concatenate((shu_disper, shu_shock, tgv), axis=1)
        cq_q = torch.tensor(cq_q, dtype=torch.float)
        obj = torch.tensor(obj, dtype=torch.float)
    elif OC.num_outputs == 2:
        data = read_from_csv('runtime_data/sod_runtime_samples.csv')

        sod_dissipation_raw = data[:, 3].reshape(-1, 1).astype(float)

        sod_dissipation = standaardize_y(sod_dissipation_raw)

        obj = np.concatenate((sod_dissipation, tgv), axis=1)
        cq_q = torch.tensor(cq_q, dtype=torch.float)
        obj = torch.tensor(obj, dtype=torch.float)
    else:
        obj = tgv
        obj = torch.tensor(obj, dtype=torch.float)
        cq_q = torch.tensor(cq_q, dtype=torch.float)

    return cq_q, obj


def runtime_cross_validation(ehvi_model, iteration):
    save_path = f'{OC.case_folder}/figures'
    if not os.path.exists(save_path):
        os.makedirs(save_path)
    if OC.num_outputs == 3:
        names = ['disper', 'shock', 'tgv']
    elif OC.num_outputs == 2:
        names = ['sod', 'tgv']
    else:
        names = ['tgv']
    cq_q, obj = initialize_cv()
    for model in ehvi_model.model.model.models:
        i = names.index(model.name)
        print(i, model.name)
        cross_validation(model, cq_q[:iteration+OC.initial_samples], obj[:iteration+OC.initial_samples, i], title=f"Cross Validation")
        plt.savefig(f"runtime_data/figures/{names[i]}_cv_{iteration}.jpg")