import math
import numpy as np

class AnalyticalSol:

    """
    A class used to generate an array of the solution to the logistic model analytically
    Instances:
              lambda: value for lambda in the analytical solution
              n_init: the initial starting value for the logistic model
              c: The maximum or carrying capacity of the logistic model
    """

    def __init__(self, lambda_val = 0.095, n_init = 1, c = 10):
        self.lambda_val = lambda_val
        self.t = 0
        self.n_init = n_init
        self.c = c
        self.data_size = 150

    def analytical_euler(self):
        """
        Method to populate the analytical_array with values corresponding to the
        analytical solution to the Logistic model. Returns an array of solution values
        """

        local_t = self.t
        a = (self.c - self.n_init)/self.n_init
        analytical_array = [self.n_init]
        while len(analytical_array) <= self.data_size:
            n = self.c/(1+a*math.exp(-self.lambda_val*local_t))
            analytical_array.append(n)
            local_t += 0.5
        return np.asanyarray(analytical_array)





