import numpy as np
import cma

from PAID.EulerMethod.euler import euler

class InferenceProblem(object):
    def __init__(self, model, data):
        if data.ndim != 2:
            raise AssertionError('data must be a 2 dimensional np.array of the form [[t1, t2, ...], [y1, y2, ...]]')
        if data.shape[0] != 2:
            raise AssertionError('data must be a 2 dimensional np.array of the form [[t1, t2, ...], [y1, y2, ...]]')

        self.model = model
        self.data_time = data[0, :]
        self.data_y = data[1, :]


    def _objective_function(self, parameters):
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
        xopt, es = cma.fmin2(self._objective_function, initial_parameters, step_size)
        return xopt, es


