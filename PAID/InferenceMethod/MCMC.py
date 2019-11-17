from PAID.EulerMethod import euler
from PAID.data.generator import DataGenerator
from tests.exact_models.models import logistic_growth_dimensionless 
import numpy as np
import matplotlib.pyplot as plt
import math

class MarkocChainMonteCarlo(object)
    def __init__ (self, model, data):
        self.data = data
        self.model = model

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

    def log_Likelihood_ratio(theta, theta_new, self, sigma =0.05, sd =0.01):
        x_0 = theta[0]
        Lambda = theta[1]

        x_0_new = theta_new[0]
        Lambda_new = theta_new[1]

        exact_model = self.solution(x_0=x_0, Lambda=Lambda)
        exact_model_new = self.solution(x_0= x_0_new, Lambda =Lambda_new)

        nD = self.data

        log_posterior_ratio = -np.sum((exact_model_new - nD)**2)/(2 * sigma ** 2) + np.sum((exact_model - nD)**2)/(2*(sigma**2))

        return log_posterior_ratio


    def _acceptance(theta, theta_new):
        if log_Ratio >= 0:
            return True
        else:
            x = np.random.uniform(0,1)
            if np.log(x) < log_Ratio: 
                return True
            else:
                return False



    def met_hast_estimation(x_0, Lambda, noise_Data, samples):


        theta = [x_0, Lambda]
        cov = np.eye(len(theta)) * 0.01 ** 2
        transition_model = lambda theta, cov: [theta,np.random.multivariate_normal(theta,cov)]

        noise_Data = self.data

        samples = samples
        accepted = np.empty(shape=(samples,2))
        accepted_params = 0

        for i in range(samples):
            theta_new = transition_model(theta, cov)
            log_ratio = log_Likelihood_ratio(theta,theta_new,noise_Data)
            if acceptance_criteria(log_ratio):
                theta = theta_new
                accepted[accepted_params,:] = theta_new
                accepted_params += 1
                print(theta_new)

        return accepted[:accepted_params]
