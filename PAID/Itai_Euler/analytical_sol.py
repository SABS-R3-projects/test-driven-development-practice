import math
import numpy as np


class AnalyticalSol:

    def __init__(self, lambda_val = 0.5, n_init = 1, c = 10):
        self.lambda_val = 0.5
        self.t = 0
        self.n_init = 1
        self.c = 10
        self.data_size = 400

    def analytical_euler(self):

        local_t = self.t
        a = (self.c - self.n_init)/self.n_init
        analytical_array = [self.n_init]
        while len(analytical_array) <= self.data_size:
            n = self.c/(1+a*math.exp(-self.lambda_val*local_t))
            analytical_array.append(n)
            local_t += 0.05
        return np.asanyarray(analytical_array)
