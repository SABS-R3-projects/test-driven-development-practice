import matplotlib.pyplot as plt
import numpy as np
import cma
class ODENumerical:

    def __init__(self, lambda_val = 0.1, n_init = 1, c = 10):
        self.lambda_val = lambda_val
        self.n_init = n_init
        self.c = c
        self.dt = 0.5

    def differential_func(self, N):
        return self.lambda_val*(1 - (N/self.c))*N

    def euler_solution(self):
        #dt = 0.15
        euler_array = [self.n_init]
        n = self.n_init
        while len(euler_array) <= 500:
            n += self.dt*self.differential_func(n)
            euler_array.append(n)
        euler_array = np.asanyarray(euler_array)
        return euler_array


    def euler_with_noise(self):
        numerical_sol = np.asanyarray(self.euler_solution())
        mu, sigma = 0, 0.5
        # creating a noise with the same dimension as the dataset
        noise = np.random.normal(mu, sigma, [np.shape(numerical_sol)[0], ])
        numerical_noisy = numerical_sol + noise
        return numerical_noisy

    def scoring_func(self):

        numerical_noisy = self.euler_with_noise()
        clean_data = self.euler_solution()
        scoring = lambda lambda_val, N: sum((numerical_noisy[i]-lambda_val*(1 - (N/self.c))*N)**2 for i,N in zip (range(len(clean_data)), clean_data))
        xopt, es = cma.fmin2(scoring, [0.1, 1], 0.5)
        print(xopt)
        print(es)

a = ODENumerical()
a.scoring_func()