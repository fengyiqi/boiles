from .base import ReconstructionStencil, epsilon_

coef_smoothness_0_01_ = 547.0
coef_smoothness_0_02_ = -3882.0
coef_smoothness_0_03_ = 4642.0
coef_smoothness_0_04_ = -1854.0
coef_smoothness_0_06_ = 7043.0
coef_smoothness_0_07_ = -17246.0
coef_smoothness_0_08_ = 7042.0
coef_smoothness_0_10_ = 11003.0
coef_smoothness_0_11_ = -9402.0
coef_smoothness_0_13_ = 2107.0

coef_smoothness_1_01_ = 267.0
coef_smoothness_1_02_ = -1642.0
coef_smoothness_1_03_ = 1602.0
coef_smoothness_1_04_ = -494.0
coef_smoothness_1_06_ = 2843.0
coef_smoothness_1_07_ = -5966.0
coef_smoothness_1_08_ = 1922.0
coef_smoothness_1_10_ = 3443.0
coef_smoothness_1_11_ = -2522.0
coef_smoothness_1_13_ = 547.0

coef_smoothness_2_01_ = 547.0
coef_smoothness_2_02_ = -2522.0
coef_smoothness_2_03_ = 1922.0
coef_smoothness_2_04_ = -494.0
coef_smoothness_2_06_ = 3443.0
coef_smoothness_2_07_ = -5966.0
coef_smoothness_2_08_ = 1602.0
coef_smoothness_2_10_ = 2843.0
coef_smoothness_2_11_ = -1642.0
coef_smoothness_2_13_ = 267.0

coef_smoothness_3_01_ = 2107.0
coef_smoothness_3_02_ = -9402.0
coef_smoothness_3_03_ = 7042.0
coef_smoothness_3_04_ = -1854.0
coef_smoothness_3_06_ = 11003.0
coef_smoothness_3_07_ = -17246.0
coef_smoothness_3_08_ = 4642.0
coef_smoothness_3_10_ = 7043.0
coef_smoothness_3_11_ = -3882.0
coef_smoothness_3_13_ = 547.0

coef_weights_1_ = 1.0 / 35.0
coef_weights_2_ = 12.0 / 35.0
coef_weights_3_ = 18.0 / 35.0
coef_weights_4_ = 4.0 / 35.0

coef_stencils_1_ = -3.0 / 12.0
coef_stencils_2_ = 13.0 / 12.0
coef_stencils_3_ = -23.0 / 12.0
coef_stencils_4_ = 25.0 / 12.0

coef_stencils_6_ = 1.0 / 12.0
coef_stencils_7_ = -5.0 / 12.0
coef_stencils_8_ = 13.0 / 12.0
coef_stencils_9_ = 3.0 / 12.0

coef_stencils_11_ = -1.0 / 12.0
coef_stencils_12_ = 7.0 / 12.0
coef_stencils_13_ = 7.0 / 12.0
coef_stencils_14_ = -1.0 / 12.0

coef_stencils_16_ = 3.0 / 12.0
coef_stencils_17_ = 13.0 / 12.0
coef_stencils_18_ = -5.0 / 12.0
coef_stencils_19_ = 1.0 / 12.0


epsilon_weno7_ = 1.0e-10


class WENO7(ReconstructionStencil):
    def __init__(self, nonlinear=True):
        self.nonlinear = nonlinear

    def apply(self, value):
        if not isinstance(value, list):
            raise Exception("Inputs must be a list.")
        elif len(value) != 7:
            raise Exception("Inputs must have at least 7 values.")
        else:
            v1 = value[0]
            v2 = value[1]
            v3 = value[2]
            v4 = value[3]
            v5 = value[4]
            v6 = value[5]
            v7 = value[6]

            if self.nonlinear:

                s11 = coef_smoothness_0_01_ * v1 + coef_smoothness_0_02_ * v2 + coef_smoothness_0_03_ * v3 + coef_smoothness_0_04_ * v4
                s12 = coef_smoothness_0_06_ * v2 + coef_smoothness_0_07_ * v3 + coef_smoothness_0_08_ * v4
                s13 = coef_smoothness_0_10_ * v3 + coef_smoothness_0_11_ * v4
                s14 = coef_smoothness_0_13_ * v4

                s1 = v1 * s11 + v2 * s12 + v3 * s13 + v4 * s14

                s21 = coef_smoothness_1_01_ * v2 + coef_smoothness_1_02_ * v3 + coef_smoothness_1_03_ * v4 + coef_smoothness_1_04_ * v5
                s22 = coef_smoothness_1_06_ * v3 + coef_smoothness_1_07_ * v4 + coef_smoothness_1_08_ * v5
                s23 = coef_smoothness_1_10_ * v4 + coef_smoothness_1_11_ * v5
                s24 = coef_smoothness_1_13_ * v5

                s2 = v2 * s21 + v3 * s22 + v4 * s23 + v5 * s24

                s31 = coef_smoothness_2_01_ * v3 + coef_smoothness_2_02_ * v4 + coef_smoothness_2_03_ * v5 + coef_smoothness_2_04_ * v6
                s32 = coef_smoothness_2_06_ * v4 + coef_smoothness_2_07_ * v5 + coef_smoothness_2_08_ * v6
                s33 = coef_smoothness_2_10_ * v5 + coef_smoothness_2_11_ * v6
                s34 = coef_smoothness_2_13_ * v6

                s3 = v3 * s31 + v4 * s32 + v5 * s33 + v6 * s34

                s41 = coef_smoothness_3_01_ * v4 + coef_smoothness_3_02_ * v5 + coef_smoothness_3_03_ * v6 + coef_smoothness_3_04_ * v7
                s42 = coef_smoothness_3_06_ * v5 + coef_smoothness_3_07_ * v6 + coef_smoothness_3_08_ * v7
                s43 = coef_smoothness_3_10_ * v6 + coef_smoothness_3_11_ * v7
                s44 = coef_smoothness_3_13_ * v7

                s4 = v4 * s41 + v5 * s42 + v6 * s43 + v7 * s44

                a1 = coef_weights_1_ / ( ( s1 + epsilon_weno7_ ) * ( s1 + epsilon_weno7_ ) )
                a2 = coef_weights_2_ / ( ( s2 + epsilon_weno7_ ) * ( s2 + epsilon_weno7_ ) )
                a3 = coef_weights_3_ / ( ( s3 + epsilon_weno7_ ) * ( s3 + epsilon_weno7_ ) )
                a4 = coef_weights_4_ / ( ( s4 + epsilon_weno7_ ) * ( s4 + epsilon_weno7_ ) )

                one_a_sum = 1.0 / ( a1 + a2 + a3 + a4 )

                w1 = a1 * one_a_sum
                w2 = a2 * one_a_sum
                w3 = a3 * one_a_sum
                w4 = a4 * one_a_sum
            else:
                w1 = coef_weights_1_
                w2 = coef_weights_2_
                w3 = coef_weights_3_
                w4 = coef_weights_4_

            return w1 * ( coef_stencils_1_ * v1 + coef_stencils_2_ * v2 + coef_stencils_3_ * v3 + coef_stencils_4_ * v4 ) + w2 * ( coef_stencils_6_ * v2 + coef_stencils_7_ * v3 + coef_stencils_8_ * v4 + coef_stencils_9_ * v5 ) + w3 * ( coef_stencils_11_ * v3 + coef_stencils_12_ * v4 + coef_stencils_13_ * v5 + coef_stencils_14_ * v6 ) + w4 * ( coef_stencils_16_ * v4 + coef_stencils_17_ * v5 + coef_stencils_18_ * v6 + coef_stencils_19_ * v7 )
