import unittest
from PAID.MCMC.Markov_Chain_2D import *

def test_error_calc():
    data = target_dist()
    new = guess(0.15, 0.35, data.times)
    new.curve = np.zeros(len(data.times)) + 0.1
    data.data = np.zeros(len(data.times))
    new.error = new.calc_error(data.data, data.times)
    assert 4.9 < new.error < 5.0

def test_ac_re_1():
    data = target_dist()
    new, newer = guess(0.15, 0.35, data.times), guess(0.15, 0.35, data.times)
    new.error = 2
    newer.error = 1
    dec = decision()
    acc = dec.accept_or_reject_1(new, newer, result_dist(), 5)
    assert acc == newer

def test_ac_re_2():
    data = target_dist()
    new, newer = guess(0.15, 0.35, data.times), guess(0.15, 0.35, data.times)
    res = result_dist()
    dec = decision()
    for i in range(100):
        new.error = 1
        newer.error = 9
        dec.accept_or_reject_1(new, newer, res, 10100, data)
    assert len(res.N_0_data) > 20 and len(res.Lambda_data) > 20


if __name__ == '__main__':
    unittest.main()