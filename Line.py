import numpy as np

class Line():
    def __init__(self):
        self.detected = False
        self.poly_coeff = []
        self.coeff_A = []
        self.coeff_A_ave = 0.
        self.coeff_B = []
        self.coeff_B_ave = 0.
        self.coeff_C = []
        self.coeff_C_ave = 0.

    def average_fit(self):

        self.coeff_A_ave = np.mean(self.coeff_A)
        self.coeff_B_ave = np.mean(self.coeff_B)
        self.coeff_C_ave = np.mean(self.coeff_C)

        return self.coeff_A_ave, self.coeff_B_ave, self.coeff_C_ave

    def update_coefficients(self, new_coeff):
        # A, B and C
        # to smooth the lane detection over frames
        self.poly_coeff.append(new_coeff)
        self.coeff_A.append(self.poly_coeff[0][0])
        self.coeff_B.append(self.poly_coeff[0][1])
        self.coeff_C.append(self.poly_coeff[0][2])


        if len(self.poly_coeff) > 29:
            _ = self.poly_coeff.pop(0)
            _ = self.coeff_A.pop(0)
            _ = self.coeff_B.pop(0)
            _ = self.coeff_C.pop(0)

        self.coeff_A_ave, self.coeff_B_ave, self.coeff_C_ave = self.average_fit()

    def fetch_poly_coeff(self):
        return [self.coeff_A_ave, self.coeff_B_ave, self.coeff_C_ave]
