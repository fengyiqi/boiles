from .teno5 import *


class TENO5Sensor(ReconstructionStencil):
    def __init__(
            self,
            coef_shock: list,
            coef_turb: list,
            ST
    ):
        self.coef_shock = coef_shock
        self.coef_turb = coef_turb
        self.ST = ST

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

        s11 = coef_smoothness_11_ * v2 + coef_smoothness_12_ * v3 + coef_smoothness_13_ * v4
        s12 = coef_smoothness_14_ * v2 + coef_smoothness_15_ * v4
        s1 = coef_smoothness_1_ * s11 * s11 + coef_smoothness_2_ * s12 * s12

        s21 = coef_smoothness_21_ * v3 + coef_smoothness_22_ * v4 + coef_smoothness_23_ * v5
        s22 = coef_smoothness_24_ * v3 + coef_smoothness_25_ * v4 + coef_smoothness_26_ * v5
        s2 = coef_smoothness_1_ * s21 * s21 + coef_smoothness_2_ * s22 * s22

        s31 = coef_smoothness_31_ * v1 + coef_smoothness_32_ * v2 + coef_smoothness_33_ * v3
        s32 = coef_smoothness_34_ * v1 + coef_smoothness_35_ * v2 + coef_smoothness_36_ * v3
        s3 = coef_smoothness_1_ * s31 * s31 + coef_smoothness_2_ * s32 * s32

        epsilon_weno5_ = 1.0e-6
        one_s1 = 1.0 / ((s1 + epsilon_weno5_) * (s1 + epsilon_weno5_))
        one_s2 = 1.0 / ((s2 + epsilon_weno5_) * (s2 + epsilon_weno5_))
        one_s3 = 1.0 / ((s3 + epsilon_weno5_) * (s3 + epsilon_weno5_))

        a1_weno5 = one_s1 / (one_s1 + one_s2 + one_s3)
        a2_weno5 = one_s2 / (one_s1 + one_s2 + one_s3)
        a3_weno5 = one_s3 / (one_s1 + one_s2 + one_s3)

        if a1_weno5 < self.ST or a2_weno5 < self.ST or a3_weno5 < self.ST:
            return TENO5(
                d0=self.coef_shock[0],
                d1=self.coef_shock[1],
                d2=self.coef_shock[2],
                CT=self.coef_shock[3],
                Cq=self.coef_shock[4],
                q=self.coef_shock[5]
            ).apply(value)
        else:
            return TENO5(
                d0=self.coef_turb[0],
                d1=self.coef_turb[1],
                d2=self.coef_turb[2],
                CT=self.coef_turb[3],
                Cq=self.coef_turb[4],
                q=self.coef_turb[5]
            ).apply(value)
