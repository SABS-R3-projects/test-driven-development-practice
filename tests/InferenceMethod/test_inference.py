import unittest
import numpy as np

from PAID.EulerMethod.euler import euler
from PAID.data.generator import DataGenerator
from PAID.InferenceMethod.inference import InferenceProblem
from tests.exact_models.models import logistic_growth, logistic_growth_dimensionless


def test_infer_parameters():
    """test model: logistic growth
    """
    ### exact parameters
    exact_parameters = [1.0, 0.1]
    Lambda, x_0 = exact_parameters

    ### generate data
    scale = 0.01
    data_time = np.linspace(0, 10, 100)
    exact_solution = logistic_growth_dimensionless(times=data_time, x_0=x_0, Lambda=Lambda)

    gen = DataGenerator()
    data_y = gen.generate_data(exact_solution, scale=scale)
    data = np.vstack(tup=(data_time, data_y))

    ODEmodel = lambda t, x, Lambda: Lambda * x * (1 - x)

    problem = InferenceProblem(ODEmodel, data)

    initial_parameters = [3.0] # initial lambda
    initial_y0 = 0.3
    step_size = 0.1
    estimated_parameters = problem.infer_parameters(y_0=initial_y0, initial_parameters=initial_parameters, step_size=step_size)

    assert np.allclose(a=estimated_parameters, b=exact_parameters, rtol=5.0e-2)


def test_objective_function():
    """test model: static: y = 1.
      test case 1 - null: make sure squared distance is zero for the exact model parameters.
      test case 2 - Comparing to manufactured data points y = 2. Hence squared distance is 1 * number of data points.
    """
    ### exact parameters
    exact_parameters = [np.nan, 1.0] # this model only has one parameter - x_0.
    Lambda, x_0 = exact_parameters

    """test case 1:"""
    ### generate data
    data_time = np.linspace(0, 10, 1000)
    exact_solution = np.ones(1000)

    data_y = exact_solution
    data = np.vstack(tup=(data_time, data_y))

    ODEmodel = lambda t, x, Lambda: 0

    problem = InferenceProblem(ODEmodel, data)

    squared_distance = problem._objective_function(exact_parameters)
    expected_squared_distance = 0.0

    if not np.isclose(a=squared_distance, b=expected_squared_distance, atol=1.0e-08):
        raise ValueError("Objective function does not return 0 when it is supposed to.")

    """test case 2:"""
    ### generate data
    data_time = np.linspace(0, 10, 1000)
    exact_solution = np.ones(1000)

    data_y = exact_solution + 1
    data = np.vstack(tup=(data_time, data_y))

    ODEmodel = lambda t, x, Lambda: 0

    problem = InferenceProblem(ODEmodel, data)

    squared_distance = problem._objective_function(exact_parameters)
    expected_squared_distance = 1.0 * len(data_y)

    if not np.isclose(a=squared_distance, b=expected_squared_distance, atol=1.0e-08):
        raise ValueError("Objective function does not seem to compute the squared distance properly.")


def test_interpolate_numerical_solution():
    """Test whether function is capable of producing linear interpolations between solution that matches input the data time points.

    Test case 1: Non-dynamic model.
    Test case 2: Model that is linear in time.
    """
    ### exact parameters and integration step size
    y_0 = 0.1
    t_0 = 0.0
    t_final = 10.0

    h = 0.001

    """test case 1:"""
    ### generate data
    data_time = np.linspace(0, 10, 3000)
    data_time = np.random.choice(a=data_time, size=100) # random sample of times.
    exact_solution = np.full(shape=100, fill_value=y_0)

    data_y = exact_solution
    data = np.vstack(tup=(data_time, data_y))

    ### solve IVP
    ODEmodel = lambda t, x: 0
    model = euler(ODEmodel)
    numerical_solution = model.integrate(h=h, t_0=t_0, t_final=t_final, y_0=y_0)

    ### Instantiating inverse problem
    ODEmodel = lambda t, x, Lambda: 0
    inv_problem = InferenceProblem(ODEmodel, data)

    interpolated_solution = inv_problem._interpolate_numerical_solution(numerical_solution)

    assert np.allclose(a=interpolated_solution, b=data, rtol=5.0e-02)

    """test case 2:"""
    ### generate data
    data_time = np.linspace(0, 10, 3000)
    data_time = np.random.choice(a=data_time, size=100) # random sample of times.
    exact_solution = np.full(shape=100, fill_value=y_0) + data_time

    data_y = exact_solution
    data = np.vstack(tup=(data_time, data_y))

    ### solve IVP
    ODEmodel = lambda t, x: 1.0
    model = euler(ODEmodel)
    numerical_solution = model.integrate(h=h, t_0=t_0, t_final=t_final, y_0=y_0)

    ### Instantiating inverse problem
    ODEmodel = lambda t, x, Lambda: 1.0
    inv_problem = InferenceProblem(ODEmodel, data)

    interpolated_solution = inv_problem._interpolate_numerical_solution(numerical_solution)

    assert np.allclose(a=interpolated_solution, b=data, rtol=5.0e-02)
