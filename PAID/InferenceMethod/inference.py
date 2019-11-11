import numpy as np
import cma

from PAID.EulerMethod.euler import euler
from PAID.data.generator import DataGenerator
from tests.exact_models.models import logistic_growth, logistic_growth_dimensionless


class InferenceProblem(object):
    def __init__(self, model, data):
        if data.ndim != 2:
            raise AssertionError('data must be a 2 dimensional np.array of the form [[t1, t2, ...], [y1, y2, ...]]')
        if data.shape[0] != 2:
            raise AssertionError('data must be a 2 dimensional np.array of the form [[t1, t2, ...], [y1, y2, ...]]')

        self.model = model
        self.data_time = data[0, :]
        self.data_y = data[0, :]


    def __objective_function(self, parameters):
        # setting lambda
        model = lambda t, x: self.model(t, x, parameters[0])
        # instantiate Eulerintegrable model
        ODEmodel = euler(model)
        # quick fix: data must have equal step size measurements so we can infer h.
        t_0 = self.data_time[0]
        t_final = self.data_time[-1]
        number_steps = len(self.data_time)
        h = (t_final - t_0) / number_steps
        numerical_estimate = ODEmodel.integrate(h=h, t_0=t_0, t_final=t_final, y_0=parameters[1])

        squared_distance = np.sum((self.data_y - numerical_estimate) ** 2)

        return squared_distance


    def infer_parameters(self, initial_parameters, step_size):
        xopt, es = cma.fmin2(self.__objective_function, initial_parameters, step_size)
        return xopt, es


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

initial_parameters = [3.0, 0.01]
step_size = 0.1

estimated_parameters, idk = problem.infer_parameters(initial_parameters, step_size)

print('estimated params: ', estimated_parameters)
print('exact params: ', exact_parameters)
