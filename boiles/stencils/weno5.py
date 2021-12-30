from .base import ReconstructionStencil, epsilon_

coef_smoothness_1_ = 13.0 / 12.0
coef_smoothness_2_ = 0.25

coef_smoothness_11_ = 1.0
coef_smoothness_12_ = -2.0
coef_smoothness_13_ = 1.0
coef_smoothness_14_ = 1.0
coef_smoothness_15_ = -4.0
coef_smoothness_16_ = 3.0

coef_smoothness_21_ = 1.0
coef_smoothness_22_ = -2.0
coef_smoothness_23_ = 1.0
coef_smoothness_24_ = 1.0
coef_smoothness_25_ = -1.0

coef_smoothness_31_ = 1.0
coef_smoothness_32_ = -2.0
coef_smoothness_33_ = 1.0
coef_smoothness_34_ = 3.0
coef_smoothness_35_ = -4.0
coef_smoothness_36_ = 1.0

coef_weights_1_ = 0.1
coef_weights_2_ = 0.6
coef_weights_3_ = 0.3

coef_stencils_1_ = 2.0 / 6.0
coef_stencils_2_ = -7.0 / 6.0
coef_stencils_3_ = 11.0 / 6.0
coef_stencils_4_ = -1.0 / 6.0
coef_stencils_5_ = 5.0 / 6.0
coef_stencils_6_ = 2.0 / 6.0
coef_stencils_7_ = 2.0 / 6.0
coef_stencils_8_ = 5.0 / 6.0
coef_stencils_9_ = -1.0 / 6.0

epsilon_weno5_ = 1.0e-6


class WENO5(ReconstructionStencil):
    def __init__(self, nonlinear=True):
        self.nonlinear = nonlinear

    def apply(self, value):
        if not isinstance(value, list):
            raise Exception("Inputs must be a list.")
        elif len(value) != 5:
            raise Exception("Inputs must have at least 5 values.")
        else:
            v1 = value[0]
            v2 = value[1]
            v3 = value[2]
            v4 = value[3]
            v5 = value[4]

            if self.nonlinear:

                s11 = coef_smoothness_11_ * v1 + coef_smoothness_12_ * v2 + coef_smoothness_13_ * v3
                s12 = coef_smoothness_14_ * v1 + coef_smoothness_15_ * v2 + coef_smoothness_16_ * v3
                s1 = coef_smoothness_1_ * s11 * s11 + coef_smoothness_2_ * s12 * s12

                s21 = coef_smoothness_21_ * v2 + coef_smoothness_22_ * v3 + coef_smoothness_23_ * v4
                s22 = coef_smoothness_24_ * v2 + coef_smoothness_25_ * v4
                s2 = coef_smoothness_1_ * s21 * s21 + coef_smoothness_2_ * s22 * s22

                s31 = coef_smoothness_31_ * v3 + coef_smoothness_32_ * v4 + coef_smoothness_33_ * v5
                s32 = coef_smoothness_34_ * v3 + coef_smoothness_35_ * v4 + coef_smoothness_36_ * v5
                s3 = coef_smoothness_1_ * s31 * s31 + coef_smoothness_2_ * s32 * s32

                a1 = coef_weights_1_ / ((s1 + epsilon_weno5_) * (s1 + epsilon_weno5_))
                a2 = coef_weights_2_ / ((s2 + epsilon_weno5_) * (s2 + epsilon_weno5_))
                a3 = coef_weights_3_ / ((s3 + epsilon_weno5_) * (s3 + epsilon_weno5_))

                one_a_sum = 1.0 / (a1 + a2 + a3)

                w1 = a1 * one_a_sum
                w2 = a2 * one_a_sum
                w3 = a3 * one_a_sum
            else:
                w1 = coef_weights_1_
                w2 = coef_weights_2_
                w3 = coef_weights_3_

            return w1 * (coef_stencils_1_ * v1 + coef_stencils_2_ * v2 + coef_stencils_3_ * v3) + w2 * (
                    coef_stencils_4_ * v2 + coef_stencils_5_ * v3 + coef_stencils_6_ * v4) + w3 * (
                           coef_stencils_7_ * v3 + coef_stencils_8_ * v4 + coef_stencils_9_ * v5)
