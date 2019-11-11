from hypothesis import given, settings
import hypothesis.strategies as st
import unittest
import numpy as np
from matplotlib import pyplot as plt

from PAID.EulerMethod import euler
from tests.exact_models.models import logistic_growth, logistic_growth_dimensionless

# unittest
def test_euler():
    '''
    Testing the numerical solution of the implemented Euler method.
    '''
    ### model parameters
    Lambda = 1.0
    C = 100.0

    # integration paramters
    h = 0.001 # time steps
    t_0 = 0.0 # initial time
    t_final = 10.0 # final time
    y_0 = 1.0 # initial condition

    times = np.arange(start=t_0, stop=t_final, step=h)
    analytic = logistic_growth(times, C=C, Lambda=Lambda)
    # logistic growth ODE
    model = lambda t, N: Lambda * N * (1 - N / C)
    logistic_model = euler.euler(model)
    numeric = logistic_model.integrate(h=h, t_0=t_0, t_final=t_final, y_0=y_0)

    tolerance = 0.1
    assert np.all((analytic - numeric) ** 2 < tolerance ** 2)

# scaling of error with step size
# def test_euler_scaling():
#     '''
#     Testing whether error of numerical solution scales as expected for Euler method (quadratic with h).
#     '''
#     ### model parameters
#     Lambda = 1.0*1e-03
#     C = 100.0

#     # integration paramters
#     h_array = np.linspace(1, 100, 100) # time steps
#     t_0 = 0.0 # initial time
#     t_final = 1000.0 # final time
#     y_0 = 1.0 # initial condition

#     cumulative_error = np.empty(len(h_array)) # init error container
#     for id_h, h in enumerate(h_array):
#         times = np.arange(start=t_0, stop=t_final, step=h)
#         analytic = logistic_growth(times, C=C, Lambda=Lambda)
#         # logistic growth ODE
#         model = lambda t, N: Lambda * N * (1 - N / C)
#         logistic_model = euler.euler(model)
#         numeric = logistic_model.integrate(h=h, t_0=t_0, t_final=t_final, y_0=y_0)

#         #cumulative_error[id_h] = np.linalg.norm(x=analytic - numeric, ord=1)
#         cumulative_error[id_h] = abs((analytic - numeric)[-1])

#     plt.plot(h_array, cumulative_error)
#     plt.show()

#     assert True



# hypothesis
class TestSimple(unittest.TestCase):
    @given(Lambda = st.floats(min_value=0.0, max_value=10.0))
    def test_euler_Lambda(self, Lambda):
        # integration paramters
        h = 0.001 # time steps
        t_0 = 0.0 # initial time
        t_final = 10.0 # final time (limit time interval just to avoid computing time issues)
        y_0 = 0.1 # initial condition

        times = np.arange(start=t_0, stop=t_final, step=h)
        analytic = logistic_growth_dimensionless(times, Lambda=Lambda)

        # logistic growth ODE
        model = lambda t, x: Lambda * x * (1 - x)
        logistic_model = euler.euler(model)
        numeric = logistic_model.integrate(h=h, t_0=t_0, t_final=t_final, y_0=y_0)

        tolerance = 0.1
        assert np.all((analytic - numeric) ** 2 <= tolerance ** 2)


    @given(y_0 = st.floats(min_value=0.0, max_value=1.0))
    def test_euler_initial_y(self, y_0):
        # model parameters
        Lambda = 1.0

        # integration paramters
        h = 0.001 # time steps
        t_0 = 0.0 # initial time
        t_final = 10.0 # final time (limit time interval just to avoid computing time issues)
        y_0 = 0.1 # initial condition

        times = np.arange(start=t_0, stop=t_final, step=h)
        analytic = logistic_growth_dimensionless(times, Lambda=Lambda)

        # logistic growth ODE
        model = lambda t, x: Lambda * x * (1 - x)
        logistic_model = euler.euler(model)
        numeric = logistic_model.integrate(h=h, t_0=t_0, t_final=t_final, y_0=y_0)

        tolerance = 0.1
        assert np.all((analytic - numeric) ** 2 <= tolerance ** 2)


    @given(t_0 = st.floats(min_value=0.0, max_value=1.0))
    def test_euler_intial_time(self, t_0):

        ##### the hard coded analytical solution assumes that t_0 = 0. That is why the test fails!!!

        # model parameters
        Lambda = 1.0

        # integration paramters
        h = 0.001 # time steps
        t_final = 10.0 # final time (limit time interval just to avoid computing time issues)
        y_0 = 0.1 # initial condition

        times = np.arange(start=t_0, stop=t_final, step=h)
        analytic = logistic_growth_dimensionless(times, t_0=t_0, Lambda=Lambda)

        # logistic growth ODE
        model = lambda t, x: Lambda * x * (1 - x)
        logistic_model = euler.euler(model)
        numeric = logistic_model.integrate(h=h, t_0=t_0, t_final=t_final, y_0=y_0)

        tolerance = 0.1
        assert np.all((analytic - numeric) ** 2 <= tolerance ** 2)
