from .base import ReconstructionStencil, epsilon_

coef_smoothness_1_ = 13.0
coef_smoothness_2_ = 3.0

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

coef_smoothness_411_ = 341.0 / 5760.0
coef_smoothness_412_ = -2785.0 / 5760.0
coef_smoothness_413_ = -2590.0 / 5760.0
coef_smoothness_414_ = 6670.0 / 5760.0
coef_smoothness_415_ = -1895.0 / 5760.0
coef_smoothness_416_ = 259.0 / 5760.0

coef_smoothness_421_ = -1.0 / 16.0
coef_smoothness_422_ = 12.0 / 16.0
coef_smoothness_423_ = -22.0 / 16.0
coef_smoothness_424_ = 12.0 / 16.0
coef_smoothness_425_ = -1.0 / 16.0

coef_smoothness_431_ = -5.0 / 144.0
coef_smoothness_432_ = -11.0 / 144.0
coef_smoothness_433_ = 70.0 / 144.0
coef_smoothness_434_ = -94.0 / 144.0
coef_smoothness_435_ = 47.0 / 144.0
coef_smoothness_436_ = -7.0 / 144.0

coef_smoothness_441_ = 1.0 / 24.0
coef_smoothness_442_ = -4.0 / 24.0
coef_smoothness_443_ = 6.0 / 24.0
coef_smoothness_444_ = -4.0 / 24.0
coef_smoothness_445_ = 1.0 / 24.0

coef_smoothness_461_ = -1.0 / 120.0
coef_smoothness_462_ = 5.0 / 120.0
coef_smoothness_463_ = -10.0 / 120.0
coef_smoothness_464_ = 10.0 / 120.0
coef_smoothness_465_ = -5.0 / 120.0
coef_smoothness_466_ = 1.0 / 120.0

coef_smoothness_41_ = 12.0
coef_smoothness_42_ = 52.0
coef_smoothness_43_ = 6.0
coef_smoothness_44_ = 37548.0 / 80.0
coef_smoothness_45_ = 252.0 / 5.0
coef_smoothness_46_ = 12.0 / 8.0
coef_smoothness_47_ = 1051404.0 / 140.0
coef_smoothness_48_ = 169524.0 / 224.0
coef_smoothness_49_ = 3028045620.0 / 16128.0

coef_smoothness_511_ = 1.0 / 6.0
coef_smoothness_512_ = 4.0 / 6.0
coef_smoothness_513_ = 1.0 / 6.0

weight_const_ = 20.0

coef_weights_1_ = 0.05
coef_weights_2_ = 0.45
coef_weights_3_ = 0.45
coef_weights_4_ = 0.05

coef_stencils_01_ = 2.0 / 6.0
coef_stencils_02_ = -7.0 / 6.0
coef_stencils_03_ = 11.0 / 6.0
coef_stencils_04_ = -1.0 / 6.0
coef_stencils_05_ = 5.0 / 6.0
coef_stencils_06_ = 2.0 / 6.0
coef_stencils_07_ = 2.0 / 6.0
coef_stencils_08_ = 5.0 / 6.0
coef_stencils_09_ = -1.0 / 6.0
coef_stencils_10_ = 11.0 / 6.0
coef_stencils_11_ = -7.0 / 6.0
coef_stencils_12_ = 2.0 / 6.0


class WENOCU6(ReconstructionStencil):
    def __init__(self):
        pass

    @staticmethod
    def apply(value):
        if not isinstance(value, list):
            raise Exception("Inputs must be a list.")
        elif len(value) != 6:
            raise Exception("Inputs must have at least 6 values.")
        else:
            v1 = value[0]
            v2 = value[1]
            v3 = value[2]
            v4 = value[3]
            v5 = value[4]
            v6 = value[5]

            s11 = coef_smoothness_11_ * v1 + coef_smoothness_12_ * v2 + coef_smoothness_13_ * v3
            s12 = coef_smoothness_14_ * v1 + coef_smoothness_15_ * v2 + coef_smoothness_16_ * v3
            s1 = coef_smoothness_1_ * s11 * s11 + coef_smoothness_2_ * s12 * s12

            s21 = coef_smoothness_21_ * v2 + coef_smoothness_22_ * v3 + coef_smoothness_23_ * v4
            s22 = coef_smoothness_24_ * v2 + coef_smoothness_25_ * v4
            s2 = coef_smoothness_1_ * s21 * s21 + coef_smoothness_2_ * s22 * s22

            s31 = coef_smoothness_31_ * v3 + coef_smoothness_32_ * v4 + coef_smoothness_33_ * v5
            s32 = coef_smoothness_34_ * v3 + coef_smoothness_35_ * v4 + coef_smoothness_36_ * v5
            s3 = coef_smoothness_1_ * s31 * s31 + coef_smoothness_2_ * s32 * s32

            s41 = coef_smoothness_416_ * v6 + coef_smoothness_415_ * v5 + coef_smoothness_414_ * v4 + \
                  coef_smoothness_413_ * v3 + coef_smoothness_412_ * v2 + coef_smoothness_411_ * v1
            s42 = coef_smoothness_425_ * v5 + coef_smoothness_424_ * v4 + coef_smoothness_423_ * v3 + \
                  coef_smoothness_422_ * v2 + coef_smoothness_421_ * v1
            s43 = coef_smoothness_436_ * v6 + coef_smoothness_435_ * v5 + coef_smoothness_434_ * v4 + \
                  coef_smoothness_433_ * v3 + coef_smoothness_432_ * v2 + coef_smoothness_431_ * v1
            s44 = coef_smoothness_445_ * v5 + coef_smoothness_444_ * v4 + coef_smoothness_443_ * v3 + \
                  coef_smoothness_442_ * v2 + coef_smoothness_441_ * v1
            s45 = coef_smoothness_466_ * v6 + coef_smoothness_465_ * v5 + coef_smoothness_464_ * v4 + \
                  coef_smoothness_463_ * v3 + coef_smoothness_462_ * v2 + coef_smoothness_461_ * v1
            s4 = s41 * s41 * coef_smoothness_41_ + s42 * s42 * coef_smoothness_42_ + s41 * s43 * coef_smoothness_43_ + \
                 s43 * s43 * coef_smoothness_44_ + s42 * s44 * coef_smoothness_45_ + s41 * s45 * coef_smoothness_46_ + \
                 s44 * s44 * coef_smoothness_47_ + s43 * s45 * coef_smoothness_48_ + s45 * s45 * coef_smoothness_49_

            s51 = coef_smoothness_511_ * s1 + coef_smoothness_512_ * s2 + coef_smoothness_513_ * s3
            s5 = abs(s4 - s51)

            r1 = weight_const_ + s5 / (s1 + epsilon_)
            r2 = weight_const_ + s5 / (s2 + epsilon_)
            r3 = weight_const_ + s5 / (s3 + epsilon_)
            r4 = weight_const_ + s5 / (s4 + epsilon_)

            a1 = coef_weights_1_ * r1
            a2 = coef_weights_2_ * r2
            a3 = coef_weights_3_ * r3
            a4 = coef_weights_4_ * r4

            one_a_sum = 1.0 / (a1 + a2 + a3 + a4)

            w1 = a1 * one_a_sum
            w2 = a2 * one_a_sum
            w3 = a3 * one_a_sum
            w4 = a4 * one_a_sum

            return w1 * (coef_stencils_01_ * v1 + coef_stencils_02_ * v2 + coef_stencils_03_ * v3) + w2 * (
                    coef_stencils_04_ * v2 + coef_stencils_05_ * v3 + coef_stencils_06_ * v4) + w3 * (
                           coef_stencils_07_ * v3 + coef_stencils_08_ * v4 + coef_stencils_09_ * v5) + w4 * (
                           coef_stencils_10_ * v4 + coef_stencils_11_ * v5 + coef_stencils_12_ * v6)
