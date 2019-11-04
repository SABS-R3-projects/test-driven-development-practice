from hypothesis import given, settings
import hypothesis.strategies as st
import unittest
import numpy as np

from PAID.EulerMethod import euler

def logistic_growth(times, t_0=0.0, N_0=1.0, C=100.0, Lambda=1.0):
    '''
    Example ODE for testing that we know the analytical solution of.

    args:
     times: np.array of time points.
     t_0: Initial time.
     N_0: Initial population size.
     C: Carrying capacity / maximal population size.
     Lambda: exponential growth factor.

    return:
     np.array of logistic function evaluated at times 'times + t_0'.
    '''
    times += t_0 # applying set-off
    result = C / (1 + (C / N_0 - 1) * np.exp(-Lambda * times))
    return result

def test_euler():
    '''
    Testing the numerical solution of the implemented Euler method.
    '''
    times = np.linspace(start=0, stop=10, num=100)
    analytic = logistic_growth(times)
    numeric = euler()

    tolerance = 1e-05
    assert (analytic - numeric) ** 2 < tolerance ** 2