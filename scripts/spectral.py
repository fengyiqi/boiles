import numpy as np
import matplotlib.pyplot as plt
from boiles.postprocessing.smoothness import periodic
from boiles.stencils import *

N = 331
h = 1 / N
n = np.arange(0, int(N / 2) + 1)
t = 0.00001
phi_n = 2 * np.pi * n / N
lambda_ = 1 / n
xs = np.arange(0, 1, h) + h / 2
delta = 1 * t / h


def cal_spectral(stencil, size, index):
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


real_upwind, imag_upwind = cal_spectral(Upwind, 1, [0])
real_central, imag_central = cal_spectral(Central, 1, [0, 1])
real_weno3, imag_weno3 = cal_spectral(WENO3(), 2, [-1, 0, 1])
real_weno5, imag_weno5 = cal_spectral(WENO5(), 3, [-2, -1, 0, 1, 2])
real_teno5, imag_teno5 = cal_spectral(TENO5(), 3, [-2, -1, 0, 1, 2])

plt.figure(dpi=150)
plt.plot(phi_n, phi_n, 'k.', label="Spectral", alpha=0.3)
plt.plot(phi_n, real_upwind, label="UW1")
plt.plot(phi_n, real_central, label="C2")
plt.plot(phi_n, real_weno3, label="WENO3")
plt.plot(phi_n, real_weno5, label="WENO5")
plt.plot(phi_n, real_teno5, label="TENO5")
plt.ylim(-0.5, 2)
plt.xlim(phi_n[0], phi_n[-1])
plt.legend(fontsize=6)
plt.grid()
plt.show()

plt.figure(dpi=150)
plt.plot(phi_n, np.zeros(int(N / 2) + 1), 'k.', label="Spectral", alpha=0.3)
plt.plot(phi_n, imag_upwind, label="UW1")
plt.plot(phi_n, imag_central, label="C2")
plt.plot(phi_n, imag_weno3, label="WENO3")
plt.plot(phi_n, imag_weno5, label="WENO5")
plt.plot(phi_n, imag_teno5, label="TENO5")
plt.grid()
plt.ylim(-2.5, 0.5)
plt.xlim(phi_n[0], phi_n[-1])
plt.legend(fontsize=6)
plt.show()
