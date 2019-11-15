import unittest
from PAID.MCMC.Markov_Chain_2D import *

def test_something():
    data = target_dist()
    new = guess(0.15, 0.35, data.times)
    new.curve = np.ones(len(data.times))
    data.data = np.zeros(len(data.times))
    new.calc_error(data.data, data.times)

    assert True


if __name__ == '__main__':
    unittest.main()
