from ...utils import read_from_csv
from ...config.opt_config import OC
import numpy as np
import torch
import gpytorch
from ax.models.torch.utils import predict_from_model


def normalize(x, bounds):
    return (x - bounds[0]) / (bounds[1] - bounds[0])


def get_inputs(filename: str):
    data = read_from_csv(filename)
    data = np.array(data, dtype=float).copy()
    return data[:, :-1]


def get_normalized_inputs(filename: str):
    bounds = [bound for bound in OC.opt_bounds]
    dim = len(bounds)
    inputs = get_inputs(filename)
    for i in range(dim):
        inputs[:, i] = normalize(inputs[:, i], OC.opt_bounds[i])

    return inputs


def get_bestf(filename: str):
    data = read_from_csv(filename)
    data = np.array(data, dtype=float).copy()
    data[:, -1] = (data[:, -1] - data[:, -1].mean()) / data[:, -1].std()
    y_min = data[:, -1].min()
    return y_min


def cast_through_int(value_float, bounds):
    # transfer float value to integer using a simply rounding
    value_float_via_int = []
    temp = np.array(value_float).copy()
    for i, bound in enumerate(bounds):

        array = (temp[:, i] * (bound[1] - bound[0]) + bound[0]).round().astype(int)
        temp[:, i] = normalize(array, bound)
    return temp


def to_int(value_float, bounds):
    if len(value_float) == 1:
        value_float = value_float.squeeze()
    value_int = []
    for i, bound in enumerate(bounds):
        temp = int(round(float(value_float[i]) * (bound[1] - bound[0]) + bound[0]))
        value_int.append(temp)
    # value_float should be a integer-valued float, e.g. value 0.5 is 5 for [0, 10]
    return value_int


def get_gpdata(model, n=100):
    x = torch.linspace(0, 1, n).reshape(-1, 1)

    x_test = gpytorch.utils.grid.create_data_from_grid(x.repeat(1, OC.dim_inputs))

    pred, var = predict_from_model(model, x_test)

    if OC.dim_inputs == 2:
        pred = pred.numpy().reshape((n, n))
        x1, x2 = np.meshgrid(x, x)
        return x1, x2, pred
    elif OC.dim_inputs == 3:
        pred = pred.numpy().reshape((n, n, n))
        Y, Z, X = np.meshgrid(x, x, x)
        return X, Y, Z, pred
    else:
        raise Exception(f"Such dimension is invalid.")
