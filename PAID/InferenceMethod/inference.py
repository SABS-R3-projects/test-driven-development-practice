import cma
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from abc import ABC, abstractmethod
from typing import Callable, List
from tqdm import tqdm

from PAID.EulerMethod.euler import euler

class AbstractInferenceProblem(ABC):
    # This method infers the parameters of a model that capture some observed data according to some objective function the best.
    @abstractmethod
    def infer_parameters(self,
                         y_0: float,
                         initial_parameters: List[float],
                         step_size: float) -> np.ndarray:
        raise NotImplementedError('Method not implemented')

    # This method graphically visualises the infered model and the provided data.
    @abstractmethod
    def plot(self) -> None:
        raise NotImplementedError('Method not implemented')

class InferenceProblem(AbstractInferenceProblem):
    def __init__(self,
                 model: Callable[[float, float, List[float]], float],
                 data: np.ndarray,
                 h=0.001) -> None:
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
    def __init__(self,
                 model: Callable[[float, float, List[float]], float],
                 data: np.ndarray,
                 h=0.001) -> None:
        super(MCMCInferenceProblem, self).__init__(model, data, h)

    def infer_parameters(self,
                         initial_parameters: List[float],
                         y_0: float,
                         initial_noise: float,
                         valid_parameter_interval: np.ndarray,
                         sampling_stepsize: np.ndarray,
                         max_iterations=1000) -> List[np.ndarray]:
        """Infer parameters of model from data using a Markov Chain Monte Carlo sampler. Priors are assumed
        to be uniform.


        Arguments:
            initial_parameters {List[float]} -- Starting point of sampler in parameter space. Here only the
                                                parameters of the ODE are passed. Initial value of the state
                                                variable y_0 and data noise are treated separately.
            y_0 {float} -- Initial value of state variable for sampling.
            initial_noise {float} -- Initial value for standard deviation around model for sampling.
            valid_parameter_interval {np.ndarray} -- Bounds on model parameters for implementation of the
                                                     priors. Shape (# parameters, 2), where first value is minimum
                                                     and second value is maximum.
            sampling_stepsize {np.ndarray} -- Average step size, i.e. std of multivariate normal, of sampler.
        
        Keyword Arguments:
            max_iterations {int} -- Number of iterations run by the sampler. (default: {1000})
        
        Returns:
            List[np.ndarray] -- [infered parameters, mean of posterior, std of posterior]
        """
        initial_parameters.append(y_0)
        initial_parameters.append(initial_noise)
        self.min_parameters = valid_parameter_interval[:, 0]
        self.max_parameters = valid_parameter_interval[:, 1]
        self.number_parameters = len(initial_parameters)
        self.sampling_covariance = sampling_stepsize ** 2
        self.posteriors = self._find_posterior(initial_parameters, max_iterations)

        self.optimal_parameters = np.empty(shape=self.number_parameters)
        mean_parameters = np.empty(shape=self.number_parameters)
        parameter_std = np.empty(shape=self.number_parameters)
        for param_id, parameter_posterior in enumerate(self.posteriors):
            hist, param_values = parameter_posterior
            max_id = np.argmax(hist)
            self.optimal_parameters[param_id] = param_values[max_id]
            mean_parameters[param_id] = np.sum(param_values * hist) / np.sum(hist)
            parameter_std[param_id] = np.sqrt(
                np.sum((param_values-mean_parameters[param_id]) ** 2 * hist)
                / np.sum(hist)
            )

        return [self.optimal_parameters, mean_parameters ,parameter_std]

    def plot(self) -> None:
        """Visualise infered model. Plots data and the model with infered parameters as well as posterior distributions of
        parameters.

        Returns:
            None
        """
        fig = plt.figure(figsize=(18,8) ,tight_layout=True)
        gs = gridspec.GridSpec(2, self.number_parameters)

        ### plot data and fit.
        ax = fig.add_subplot(gs[0, :])
        # Solve ODEmodel with otimal parameters.
        ODEmodel = lambda t, x: self.model(t, x, self.optimal_parameters[0])
        y_0 = self.optimal_parameters[-2]
        std = self.optimal_parameters[-1]
        # instantiate ODE model
        model = euler(ODEmodel)
        t_0 = self.data_time[0]
        t_final = self.data_time[-1]
        numerical_estimate = model.integrate(h=self.h, t_0=t_0, t_final=t_final, y_0=y_0)

        # Scatter plot data.
        ax.scatter(x=self.data_time, y=self.data_y, color='gray', edgecolors='darkgreen', alpha=0.5, label='data')
        # Line plot fitted model
        ax.plot(numerical_estimate[0, :], numerical_estimate[1, :], color='black', label='model')
        # plot standard deviation
        ax.fill_between(numerical_estimate[0, :], numerical_estimate[1, :], numerical_estimate[1, :] - std, color='grey', alpha=0.3)
        ax.fill_between(numerical_estimate[0, :], numerical_estimate[1, :], numerical_estimate[1, :] + std, color='grey', alpha=0.3)

        ax.set_xlabel('time')
        ax.set_ylabel('state variable')
        ax.legend()

        for id_p, posterior in enumerate(self.posteriors):
            ax = fig.add_subplot(gs[1, id_p])
            ax.set_ylabel('posterior of parameter %d [# counts]' % id_p)
            ax.set_xlabel('parameter %d [dimensionless]' % id_p)
            hist, param_values = posterior
            ax.plot(param_values, hist, color='black', label='histogram')
            ax.axvline(x=self.optimal_parameters[id_p], color='darkgreen', label='optimum')

            ax.legend()

        plt.show()

    def _find_posterior(self, initial_parameters: List[float], max_iterations: int) -> List[np.ndarray]:
        """Computes posterior distributions of parameters using MCMC sampling.

        Arguments:
            initial_parameters {List[float]} -- Starting point in parameter space of MCMC sampler.
            max_iterations {int} -- Total number of steps in parameter space that are proposed in the sampling process.
        
        Returns:
            posteriors {List[np.ndarray]} -- Posterior distributions of parameters as arrays (counts, parameter value).
        """
        parameter_history = np.empty(shape=(self.number_parameters, max_iterations))
        parameter_history[:, 0] = initial_parameters

        current_parameters = np.array(initial_parameters) # to make future type explicit
        number_accepted_parameters = 0 # book-keeping
        for _ in tqdm(range(max_iterations)):
            proposed_parameters = self._propose_step(current_parameters)
            log_posterior_ratio = self._compute_log_posterior_ratio(proposed_parameters, current_parameters)
            is_accepted = self._are_parameters_accepted(log_posterior_ratio)

            if is_accepted:
                number_accepted_parameters += 1
                current_parameters = proposed_parameters
                parameter_history[:, number_accepted_parameters] = current_parameters

        # through away unused space in the container
        print(number_accepted_parameters, " proposed steps have been accepted.")
        parameter_history = parameter_history[:, :number_accepted_parameters + 1]
        posteriors = self._convert_history_to_distribution(parameter_history)

        return posteriors

    def _propose_step(self, current_parameters: np.ndarray) -> np.ndarray:
        """Draws a step proposal in parameter space from a multivariate normal distribution centered around the current parameter set.

        Arguments:
            current_parameters {np.ndarray} -- Current point in parameter space.
        
        Returns:
            parameter proposal {np.ndarray} -- Proposed point in parameter space.
        """
        covariance_matrix = np.diag(self.sampling_covariance)
        parameter_proposal = np.random.multivariate_normal(mean=current_parameters, cov=covariance_matrix)

        return parameter_proposal

    def _compute_log_posterior_ratio(self, proposed_parameters: np.ndarray, current_parameters: np.ndarray):
        """Computes the log-posterior ratio of the current parameter set and the proposed one assuming Gaussian noise. The variance
        of the noise is also inferred by the algorithm. We assume uniform priors for all parameters.
        TODO: provide option for priors other than uniform.

        Returns:
            log_posterior_ratio {float} -- ratio of log-posteriors of proposed and current parameter set.
        """
        for id_p, parameter in enumerate(proposed_parameters):
            if (parameter < self.min_parameters[id_p]) or (parameter > self.max_parameters[id_p]):
                # parameters out of bounds, i.e. prior weight is zero and thus leads to rejection.
                return -np.inf

        # splitting parameters into parameters needed for ODE integration and noise std.
        proposed_model_parameters = proposed_parameters[:-1]
        proposed_std = proposed_parameters[-1]

        current_model_parameters = current_parameters[:-1]
        current_std = current_parameters[-1]

        # compute log-ratio
        number_data_points = len(self.data_y)
        if proposed_std ** 2 == 0:
            proposed_std += 1e-5 # avoid dividing by zero

        log_posterior_ratio = - (self._objective_function(proposed_model_parameters) / (2 * proposed_std ** 2)
                              - self._objective_function(current_model_parameters) / (2 * current_std ** 2)
                              + number_data_points * (np.log(proposed_std) - np.log(current_std)))

        return log_posterior_ratio

    def _are_parameters_accepted(self, log_posterior_ratio: float) -> bool:
        """Checks whether the posterior weight of the proposed parameter set is larger/equal or smaller than the posterior weight
        of the current parameter set. Accept if proposed posterior is larger or equal. If proposed posterior is smaller accept with
        a probability equal to the ratio of the posteriors.

        Arguments:
            log_posterior_ratio {float} -- Logarithm of the ratio of the posterior weight of the proposed parameter set and the
                                           posterior weight of the current parameter set.
        
        Returns:
            bool -- Booleian whether or not proposed parameter set is accepted.
        """
        if log_posterior_ratio >= 0:
            return True
        else:
            random_number = np.random.uniform(low=0.0, high=1.0, size=1)
            if np.log(random_number) <= log_posterior_ratio:
                return True
            else:
                return False

    def _convert_history_to_distribution(self, parameter_history: np.ndarray) -> List[np.ndarray]:
        """Converts bin edges to their center values to make the histogram better interpretable as distbrution.

        Arguments:
            parameter_history {np.ndarray} -- List of histogram of sampled parameters.
        
        Returns:
            posteriors List[np.ndarray] -- List of posterior distributions of parameters.
        """
        number_accepted_parameters = parameter_history.shape[1]
        warm_up_phase = int(0.25 * number_accepted_parameters)
        parameter_history = parameter_history[:, warm_up_phase:]

        posteriors = []
        for id_p, parameter in enumerate(parameter_history):
            hist, bin_egdes = np.histogram(parameter, bins='auto', density=False)
            bin_size = (bin_egdes[1] - bin_egdes[0]) / 2
            print('The bin size of parameter %d is %f' % (id_p, bin_size))
            parameter_values = bin_egdes[:-1] + bin_size / 2 # set value to center of bin
            parameter_posterior = np.vstack(tup=(hist, parameter_values))
            posteriors.append(parameter_posterior)

        return posteriors

    def _objective_function(self, parameters: np.ndarray) -> float:
        """Least squares objective function to be minimised in the process of parameter inference.

        Arguments:
            parameters {np.ndarray} -- Set of parameters that are used to solve the ODE model.
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

