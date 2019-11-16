from PAID.EulerMethod import euler
from PAID.data.generator import DataGenerator
from tests.exact_models.models import logistic_growth_dimensionless 
import numpy as np
import matplotlib.pyplot as plt
import math

class MarkocChainMonteCarlo(object)
    def __init__ (self, data):
        self.data = data

    def solution(x_0, Lambda,times=10, samples=100):
    
        data_time = np.linspace(0, times, samples)
        exact_solution = logistic_growth_dimensionless(times=data_time, x_0=x_0, Lambda=Lambda)

        return exact_solution

    def noise_Data(exact_solution, scale=0.01, times=10,samples=100):
        gen = DataGenerator()
        data_time =     data_time = np.linspace(0, times, samples)
        noise_data = exact + gen.generate_data(exact, scale)

        noisey_Data = np.vstack((noise_data,data_time))
        #noisey_Data = np.vstack(tup=(data_time, noise_Data))

        return noisey_Data

    def Likelihood(x_0, Lambda,sigma, sd):
        exact_model = solution(x_0, Lambda)
        nD = noise_Data(exact_model) ##put elsewhere...
        #ip = InferenceProblem(exact_model,noise_Data)
        #least_squares = ip._objective_function(parameters)
        least_square = np.sum((exact_model - nD)**2)
        pi=math.pi
        normalising_factor = 1/math.sqrt(2*pi*(sigma**2))
        Gaussian = normalising_factor * np.exp(-least_square/2*(sigma**2))
        logGaussian = np.log(Gaussian)

        return logGaussian

    def acceptance(theta, theta_new):
        if theta_new > theta:
            return True
        else:
            accept=np.random.uniform(0,1)
            return (accept < (np.exp(x_new-x)))

    transition_model = lambda x: [theta,np.random.multivariate_normal(theta,covariance,)]

    def prior(theta):
        if theta[0] > 0 and theta[0] < 1:
            return 0
        else:
            return 1