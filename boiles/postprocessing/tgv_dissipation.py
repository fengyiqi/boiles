import numpy as np


# vel should be a 1D array
# Implements a fourth order central difference stencil
# Only valid for periodic boundary condition
def cal_gradient(vel, dx):
    front = vel[:2]
    rear = vel[-2:]
    extend_vel = np.insert(vel, 0, rear, axis=0)
    extend_vel = np.append(extend_vel, front, axis=0)
    gradient = []
    for i in range(2, 66):
        temp = extend_vel[i-2] - 8 * extend_vel[i-1] + 8 * extend_vel[i+1] - extend_vel[i+2]
        grad = temp / (12 * dx)
        gradient.append(grad)
    return np.array(gradient)


def cal_sixth_order_gradient(vel, dx):
    front = vel[:3]
    rear = vel[-3:]
    extend_vel = np.insert(vel, 0, rear, axis=0)
    extend_vel = np.append(extend_vel, front, axis=0)
    gradient = []
    for i in range(3, 67):
        temp = -extend_vel[i-3] + 9 * extend_vel[i-2] -45 * extend_vel[i-1] + 45 * extend_vel[i+1] - 9 * extend_vel[i+2] + extend_vel[i+3]
        grad = temp / (60 * dx)
        gradient.append(grad)
    return np.array(gradient)


# directional derivatives, velocity is (68, 68, 68) tensor, direction is 'x', 'y'
# or 'z', returns a (64, 64, 64) tensor
def first_der(velocity, direction, dx):
    gradient_matrix = np.zeros((64, 64, 64))
    for i in range(64):
        for j in range(64):
            if direction == 'x':
                vel = velocity[i, j, :]
                gradient_matrix[i, j, :] = cal_sixth_order_gradient(vel, dx)
            if direction == 'y':
                vel = velocity[i, :, j]
                gradient_matrix[i, :, j] = cal_sixth_order_gradient(vel, dx)
            if direction == 'z':
                vel = velocity[:, i, j]
                gradient_matrix[:, i, j] = cal_sixth_order_gradient(vel, dx)
    return gradient_matrix


# according to Felix's "Assessing the numerical dissipation rate and viscosity in
# numerical simulations of fluid flows" equation 14
def dissipation_function(density, velocity_x, velocity_y, velocity_z, delta_x):
    pu1_px1 = first_der(velocity_x, 'x', delta_x)
    pu2_px2 = first_der(velocity_y, 'y', delta_x)
    pu3_px3 = first_der(velocity_z, 'z', delta_x)

    pu1_px2 = first_der(velocity_x, 'y', delta_x)
    pu2_px1 = first_der(velocity_y, 'x', delta_x)

    pu1_px3 = first_der(velocity_x, 'z', delta_x)
    pu3_px1 = first_der(velocity_z, 'x', delta_x)

    pu2_px3 = first_der(velocity_y, 'z', delta_x)
    pu3_px2 = first_der(velocity_z, 'y', delta_x)

    term_1 = 2 * (pu1_px1 ** 2 + pu2_px2 ** 2 + pu3_px3 ** 2)
    term_2 = -2.0 / 3.0 * (pu1_px1 + pu2_px2 + pu3_px3) ** 2
    term_3 = (pu1_px2 + pu2_px1) ** 2
    term_4 = (pu1_px3 + pu3_px1) ** 2
    term_5 = (pu2_px3 + pu3_px2) ** 2

    epsilon = density[:, :, :] * (term_1 + term_2 + term_3 + term_4 + term_5) * (delta_x ** 3)
    return epsilon.sum()
