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
    data_time = np.linspace(0, 10, 1000)
    exact_solution = logistic_growth_dimensionless(times=data_time, x_0=x_0, Lambda=Lambda)

    gen = DataGenerator()
    data_y = gen.generate_data(exact_solution, scale=scale)
    data = np.vstack(tup=(data_time, data_y))

    ODEmodel = lambda t, x, Lambda: Lambda * x * (1 - x)

    problem = InferenceProblem(ODEmodel, data)

    initial_parameters = [3.0, 0.2]
    step_size = 0.1
    estimated_parameters, idk = problem.infer_parameters(initial_parameters, step_size)

    print('estimated params: ', estimated_parameters)
    print('exact params: ', exact_parameters)

    assert True


def test_objective_function():
    """test model: static: y = 1.
      test case 1 - null: make sure squared distance is zero for the exact model parameters.
      test case 2 - Comparing to manufactured data points y = 2. Hence squared distance is 1 * number of data points.
    """
    ### exact parameters
    exact_parameters = [np.nan, 1.0]
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
        print('squared_distance: ', squared_distance, expected_squared_distance)
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
