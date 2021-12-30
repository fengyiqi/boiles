from .base import ReconstructionStencil, epsilon_

coef_smoothness_11_ = -1.0
coef_smoothness_12_ = 1.0
coef_smoothness_21_ = -1.0
coef_smoothness_22_ = 1.0

coef_weights_1_ = 1.0 / 3.0
coef_weights_2_ = 2.0 / 3.0

coef_stencils_1_ = -1.0 / 2.0
coef_stencils_2_ = 3.0 / 2.0
coef_stencils_3_ = 1.0 / 2.0
coef_stencils_4_ = 1.0 / 2.0


class WENO3(ReconstructionStencil):
    def __init__(self, nonlinear=True):
        self.nonlinear = nonlinear

    def apply(self, value):

        if not isinstance(value, list):
            raise Exception("Inputs must be a list.")
        elif len(value) != 3:
            raise Exception("Inputs must have at least 3 values.")
        else:
            v1 = value[0]
            v2 = value[1]
            v3 = value[2]
            if self.nonlinear:
                s11 = coef_smoothness_11_ * v1 + coef_smoothness_12_ * v2
                s1 = s11 * s11

                s21 = coef_smoothness_21_ * v2 + coef_smoothness_22_ * v3
                s2 = s21 * s21

                a1 = coef_weights_1_ / ((s1 + epsilon_) * (s1 + epsilon_))
                a2 = coef_weights_2_ / ((s2 + epsilon_) * (s2 + epsilon_))

                one_a_sum = 1.0 / (a1 + a2)

                w1 = a1 * one_a_sum
                w2 = a2 * one_a_sum
            else:
                w1 = coef_weights_1_
                w2 = coef_weights_2_

            return w1 * (coef_stencils_1_ * v1 + coef_stencils_2_ * v2) + w2 * (
                    coef_stencils_3_ * v2 + coef_stencils_4_ * v3)
