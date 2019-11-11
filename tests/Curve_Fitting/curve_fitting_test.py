import unittest
from PAID.Curve_Fitting.class_fitting_data import CurveBestFit
import numpy as np


def test_initialization():
    new_curve = main()
    assert len(new_curve.times) > 0


def test_error_calculation():
    new_curve = main()
    new_curve.minimize_error(0.01)
    assert new_curve.min_error < 15**2


def test_curve():
    new_curve = main()
    new_curve.minimize_error(0.01)
    assert len(new_curve.fit) == len(new_curve.euler_times) and 0 < sum([new_curve.fit[i] for i in new_curve.fit]) < len(new_curve.fit)


def test_estimations():
    new_curve = main()
    new_curve.minimize_error(0.01)
    assert abs(new_curve.r_est-0.8) < 0.1 and abs(new_curve.N_0_est-0.1) < 0.1


if __name__ == '__main__':
    unittest.main()


def main():
    N_accuracy = 0.1
    data_points = 10
    euler_accuracy = 0.01
    times = np.arange(0, 15, 15.0 / data_points)
    euler_times = np.arange(0.0, 15.0, euler_accuracy)
    t_0 = 0.0
    N_0 = 0.1
    Lambda = 0.8

    return CurveBestFit(times, t_0, N_0, Lambda, N_accuracy, euler_times)
