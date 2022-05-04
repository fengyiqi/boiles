import numpy as np
from .smoothness import periodic

N = 331
h = 1 / N
n = np.arange(0, int(N / 2) + 1)
t = 0.00001
phi_n = 2 * np.pi * n / N
lambda_ = 1 / n
xs = np.arange(0, 1, h) + h / 2
delta = 1 * t / h


def spectral_analyze(stencil, size, index):
    r"""
    refer to "On the spectral properties of shock-capturing schemes", Pirozzoli, 2006
    :param stencil: reconstruction stencil
    :param size: boundary width
    :param index: index of the shift cells
    :return: real part (dispersion) and imaginary part (dissipation)
    """
    v_hat = []
    for i, phi in enumerate(phi_n):
        y_ori = np.sin(2 * np.pi * xs / lambda_[i])
        y_bound = periodic(y_ori, halo_size=size)
        temp1, temp2 = 0, 0
        for j in range(size, N + size):
            pos_array = [y_bound[j + k] for k in index]
            neg_array = [y_bound[j + k - 1] for k in index]
            pos = stencil.apply(pos_array)
            neg = stencil.apply(neg_array)
            temp1 += (y_bound[j] - 1 * t / h * (pos - neg)) * np.exp(complex(0, -(j - size) * phi))
            temp2 += y_bound[j] * np.exp(complex(0, -(j - size) * phi))
        v_hat.append(temp1 / temp2)
    v_hat = np.array(v_hat)
    out = -1 / complex(0, delta) * np.log(v_hat)
    return out.real, out.imag