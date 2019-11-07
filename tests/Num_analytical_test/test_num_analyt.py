import unittest
import math
import numpy as np
import PAID.Itai_Euler.euler_num as euler_num


def analytical_euler(lambda_val = 0.5, n_init = 1, c = 10, t=0):

    a = (c - n_init)/n_init
    analytical_array = [n_init]
    while len(analytical_array) <= 400:
        n = c/(1+a*math.exp(-lambda_val*t))
        analytical_array.append(n)
        t += 0.05
    return analytical_array


class MyTestCase(unittest.TestCase):

    def test_num_vs_analytical(self):

        analytical_sol = np.asanyarray(analytical_euler())
        a = euler_num.ODENumerical()
        numerical = np.asanyarray(a.euler_solution())
        assert(len(numerical) ==  len(analytical_sol))

if __name__ == '__main__':
    unittest.main()
