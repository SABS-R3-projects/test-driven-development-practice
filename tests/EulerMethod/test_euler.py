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


def test_convergence():
    """Testing whether expected scaling behaviour of Euler's method is obeyed by our implementation (linear).
    """
    number_sampled_h = 20

    # integration paramters
    h_array = np.linspace(start=1.0e-05, stop=1.0e-03, num=number_sampled_h) # tested time steps
    t_0 = 0.0 # initial time
    t_final = 10.0 # final time (limit time interval just to avoid computing time issues)
    y_0 = 0.1 # initial condition
    Lambda = 1.0

    # logistic growth ODE
    model = lambda t, x: Lambda * x * (1 - x)
    logistic_model = euler.euler(model)

    error = np.empty(shape=number_sampled_h)
    for id_h, h in enumerate(h_array):
        times = np.arange(start=t_0, stop=t_final, step=h)
        analytic = logistic_growth_dimensionless(times, Lambda=Lambda)
        numeric = logistic_model.integrate(h=h, t_0=t_0, t_final=t_final, y_0=y_0)
        error[id_h] = abs(analytic[-1] - numeric[-1])

    # find median gradient and make sure it doesn't vary too much, i.e. linear scaling.
    gradients = np.gradient(error)
    median_gradient = np.median(gradients)

    assert np.allclose(a=gradients, b=median_gradient, rtol=1.0e-05)



# hypothesis
class TestSimple(unittest.TestCase):
    @given(Lambda = st.floats(min_value=-490, max_value=78.0)) # realm of validity depends on tolerance threshold.
    def test_euler_Lambda(self, Lambda):
        """Test the range of valid lambda values for the logistic growth model.

        Arguments:
            Lambda {float} -- growth factor.
        """
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
        squared_distance = (analytic - numeric) ** 2

        tolerance = 0.01
        assert np.all(squared_distance <= tolerance ** 2)


    @given(y_0 = st.floats(min_value=0.0, max_value=1.0))
    def test_euler_initial_y(self, y_0):
        """testing the valid realm of y_0. Here the logistic function can only assume values in (0, 1).

        Arguments:
            y_0 {float} -- initial value of the state variable.
        """
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
        squared_distance = (analytic - numeric) ** 2

        tolerance = 0.01
        assert np.all(squared_distance <= tolerance ** 2)


    @given(t_0 = st.floats(min_value=-100.0, max_value=100.0))
    def test_euler_intial_time(self, t_0):
        """Testing valid values for t_0. Within the physical limits of the computer any number t_0 shoud work. We just choose a rather
        small range [-100, 100] for efficiency of the test.

        Arguments:
            t_0 {float} -- initial time of integration.
        """
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
        squared_distance = (analytic - numeric) ** 2

        tolerance = 0.01
        assert np.all(squared_distance <= tolerance ** 2)
