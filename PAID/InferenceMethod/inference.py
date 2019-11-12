import cma
import numpy as np
import matplotlib.pyplot as plt
from abc import ABC, abstractmethod
from typing import Callable, List

from PAID.EulerMethod.euler import euler

class AbstractInferenceProblem(ABC):
    # This method infers the parameters of a model that capture some observed data according to some objective function the best.
    @abstractmethod
    def infer_parameters(self, y_0: float, initial_parameters: List[float], step_size: float) -> np.ndarray:
        raise NotImplementedError('Method not implemented')

    # This method graphically visualises the infered model and the provided data.
    @abstractmethod
    def plot(self) -> None:
        raise NotImplementedError('Method not implemented')

class InferenceProblem(AbstractInferenceProblem):
    def __init__(self, model: Callable[[float, float, List[float]], float], data: np.ndarray, h=0.001) -> None:
        if data.ndim != 2:
            raise AssertionError('data must be a 2 dimensional np.array of the form [[t1, t2, ...], [y1, y2, ...]]')
        if data.shape[0] != 2:
            raise AssertionError('data must be a 2 dimensional np.array of the form [[t1, t2, ...], [y1, y2, ...]]')
        if h <= 0:
            raise ValueError('Step-size must be positive.')

        self.model = model
        self.data_time = data[0, :]
        self.data_y = data[1, :]
        self.h = h # step-size to integrate ODE.

    def infer_parameters(self, y_0: float, initial_parameters: List[float], step_size=0.5) -> np.ndarray:
        """Infers set of parameters that minimises an objective function (so far least squares).

        Arguments:
            y_0 {float} -- Starting point of inderence in parameter space.
            initial_parameters {[type]} -- Starting point of inference in parameter space.
        
        Keyword Arguments:
            step_size {float} -- Step-size of optimizer in parameter space. (default: {0.5})
        
        Returns:
            optimal_parameters -- Set of parameters that minimise the objective function.
        """
        print('Parameters are being infered...\n')
        initial_parameters.append(y_0)
        self.optimal_parameters, _ = cma.fmin2(self._objective_function, initial_parameters, step_size)
        return self.optimal_parameters

    def plot(self) -> None:
        """Method to visualise the success/failure of the parameter inference.
        """
        # Solve ODEmodel with otimal parameters.
        ODEmodel = lambda t, x: self.model(t, x, self.optimal_parameters[0])
        # instantiate ODE model
        model = euler(ODEmodel)
        t_0 = self.data_time[0]
        t_final = self.data_time[-1]
        numerical_estimate = model.integrate(h=self.h, t_0=t_0, t_final=t_final, y_0=self.optimal_parameters[-1])

        # Generate plot.
        plt.figure(figsize=(6,6))
        # Scatter plot data.
        plt.scatter(x=self.data_time, y=self.data_y, color='gray', edgecolors='darkgreen', alpha=0.5, label='data')
        # Line plot fitted model
        plt.plot(numerical_estimate[0, :], numerical_estimate[1, :], color='black', label='model')

        plt.xlabel('time')
        plt.ylabel('state variable')
        plt.legend()

        plt.show()

    def _objective_function(self, parameters: List[float]) -> float:
        """Least squares objective function to be minimised in the process of parameter inference.

        Arguments:
            parameters {List[float]} -- Set of parameters that are used to solve the ODE model.
                The last parameter i.e. parameters[-1] is assumed to be the initial value of the
                state variable.
        Return:
            Point-wise squared distance between the data and the ODE model.
        """
        # setting lambda
        ODEmodel = lambda t, x: self.model(t, x, parameters[0])
        # instantiate ODE model
        model = euler(ODEmodel)
        t_0 = self.data_time[0]
        t_final = self.data_time[-1]
        numerical_estimate = model.integrate(h=self.h, t_0=t_0, t_final=t_final, y_0=parameters[-1])
        interpolated_solution = self._interpolate_numerical_solution(numerical_estimate)
        interpolated_solution_yvalues = interpolated_solution[1, :]
        squared_distance = np.sum((self.data_y - interpolated_solution_yvalues) ** 2)

        return squared_distance

    def _interpolate_numerical_solution(self, numerical_solution: np.ndarray) -> np.ndarray:
        """Interpolates the numerical solution of the ODE to evaluate it at the time points corresponding to the data.

        Arguments:
            numerical_solution {np.ndarray} -- 2 dimensional array found by numerically solving the IVP (times x y_values).

        Returns:
            numerical_estimate {np.ndarray} -- Returns interpolated solution matching the time points of the input data.
        """
        integration_time = numerical_solution[0, :]
        integration_values = numerical_solution[1, :]
        numerical_estimate = np.interp(x=self.data_time, xp=integration_time, fp=integration_values)

        interpolated_solution = np.vstack(tup=(self.data_time, numerical_estimate))

        return interpolated_solution


class MCMCInferenceProblem(InferenceProblem):
    def __init__(self, model: Callable[[float, float, List[float]], float], data: np.ndarray, h=0.001) -> None:
        super(MCMCInferenceProblem, self).__init__(self, model, h)

    def infer_parameters(self):
        # find posterior distributions for parameters (choice for prior, choice for noise model?, number iterations)
        ## propose step (choice for step distribution)
        ## compute posterior
        ## accept/ reject step based on ratio
        pass

    def plot(self):
        # plot parameter posteriors
        # plot ideal solution
        # plot distribution of solution based on sampling parameters from posterior.
        pass

