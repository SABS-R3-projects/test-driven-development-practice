import numpy as np

from PAID.EulerMethod.euler import euler
from PAID.data.generator import DataGenerator
from PAID.InferenceMethod.inference import InferenceProblem
from tests.exact_models.models import logistic_growth_dimensionless

### exact parameters
exact_parameters = [1.0, 0.1]
Lambda, x_0 = exact_parameters

### generate data
scale = 0.05
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

print('infered parameters: ', estimated_parameters)
print('exact parameters: ', exact_parameters)

problem.plot()

