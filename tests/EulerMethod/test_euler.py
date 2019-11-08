from hypothesis import given, settings
import hypothesis.strategies as st
import unittest
import numpy as np

from PAID.EulerMethod import euler

# unittest
def test_euler():
    '''
    Testing the numerical solution of the implemented Euler method.
    '''
    ### model parameters
    Lambda = 1.0
    C = 100.0

    # integration paramters
    dt = 0.001 # time steps
    t_0 = 0.0 # initial time
    t_final = 10.0 # final time
    y_0 = 1.0 # initial condition

    times = np.arange(start=t_0, stop=t_final, step=dt)
    analytic = logistic_growth(times, C=C, Lambda=Lambda)
    # logistic growth ODE
    func = lambda t, N: Lambda * N * (1 - N / C)
    logistic_model = euler.euler(func)
    numeric = logistic_model.integrate(dt=dt, t_0=t_0, t_final=t_final, y_0=y_0)

    tolerance = 0.1
    assert np.all((analytic - numeric) ** 2 < tolerance ** 2)


# hypothesis
class TestSimple(unittest.TestCase):
    @given(Lambda = st.floats(min_value=0.0, max_value=10.0))
    def test_euler_Lambda(self, Lambda):
        # integration paramters
        dt = 0.001 # time steps
        t_0 = 0.0 # initial time
        t_final = t_0 + 10.0 # final time (limit time interval just to avoid computing time issues)
        y_0 = 0.1 # initial condition

        times = np.arange(start=t_0, stop=t_final, step=dt)
        analytic = logistic_growth_dimensionless(times, Lambda=Lambda)

        # logistic growth ODE
        func = lambda t, x: Lambda * x * (1 - x)
        logistic_model = euler.euler(func)
        numeric = logistic_model.integrate(dt=dt, t_0=t_0, t_final=t_final, y_0=y_0)

        tolerance = 0.1
        assert np.all((analytic - numeric) ** 2 <= tolerance ** 2)


    @given(y_0 = st.floats(min_value=0.0, max_value=1.0))
    def test_euler_initial_y(self, y_0):
        # model parameters
        Lambda = 1.0

        # integration paramters
        dt = 0.001 # time steps
        t_0 = 0.0 # initial time
        t_final = t_0 + 10.0 # final time (limit time interval just to avoid computing time issues)
        y_0 = 0.1 # initial condition

        times = np.arange(start=t_0, stop=t_final, step=dt)
        analytic = logistic_growth_dimensionless(times, Lambda=Lambda)

        # logistic growth ODE
        func = lambda t, x: Lambda * x * (1 - x)
        logistic_model = euler.euler(func)
        numeric = logistic_model.integrate(dt=dt, t_0=t_0, t_final=t_final, y_0=y_0)

        tolerance = 0.1
        assert np.all((analytic - numeric) ** 2 <= tolerance ** 2)


    @given(t_0 = st.floats(min_value=0.0, max_value=1.0))
    def test_euler_intial_time(self, t_0):

        ##### the hard coded analytical solution assumes that t_0 = 0. That is why the test fails!!!

        # model parameters
        Lambda = 1.0

        # integration paramters
        dt = 0.001 # time steps
        t_final = t_0 + 10.0 # final time (limit time interval just to avoid computing time issues)
        y_0 = 0.1 # initial condition

        times = np.arange(start=t_0, stop=t_final, step=dt)
        analytic = logistic_growth_dimensionless(times, Lambda=Lambda)

        # logistic growth ODE
        func = lambda t, x: Lambda * x * (1 - x)
        logistic_model = euler.euler(func)
        numeric = logistic_model.integrate(dt=dt, t_0=t_0, t_final=t_final, y_0=y_0)

        tolerance = 0.1
        assert np.all((analytic - numeric) ** 2 <= tolerance ** 2)




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


def logistic_growth_dimensionless(times, t_0=0.0, x_0=0.1, Lambda=1.0):
    '''
    Example ODE for testing that we know the analytical solution of. This version rescales N by C, i.e. x:= N/C.

    args:
     times: np.array of time points.
     t_0: Initial time.
     x_0: Initial population size.
     Lambda: exponential growth factor.

    return:
     np.array of logistic function evaluated at times 'times + t_0'.
    '''
    times += t_0 # applying set-off
    result = x_0 / (x_0 + (1 - x_0) * np.exp(-Lambda * times))
    return result