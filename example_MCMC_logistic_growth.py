import numpy as np

from PAID.EulerMethod.euler import euler
from PAID.data.generator import DataGenerator
from PAID.InferenceMethod.inference import MCMCInferenceProblem
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

problem = MCMCInferenceProblem(ODEmodel, data)

initial_parameters = [3.0] # initial lambda
initial_y0 = 0.3
initial_noise = 0.2
step_size = 0.1

lambda_interval = np.array([0.0, 10.0])
y0_interval = np.array([0.0, 1.0])
std_interval = np.array([1.0e-5, 10])
valid_parameter_interval = np.vstack(tup=(lambda_interval, y0_interval, std_interval))

sampling_stepsize = np.array([0.05, 0.05, 0.01])
estimated_parameters = problem.infer_parameters(y_0=initial_y0,
                                                initial_parameters=initial_parameters,
                                                initial_noise=initial_noise,
                                                valid_parameter_interval=valid_parameter_interval,
                                                sampling_stepsize=sampling_stepsize)

print('infered parameters: ', estimated_parameters)
print('exact parameters: ', exact_parameters)

#problem.plot()

