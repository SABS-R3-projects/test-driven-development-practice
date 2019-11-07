import matplotlib.pyplot as plt
import numpy as np
class ODENumerical:

    def __init__(self, lambda_val = 0.1, n_init = 1, c = 10):
        self.lambda_val = lambda_val
        self.n_init = n_init
        self.c = c

    def differential_func(self, N):
        return self.lambda_val*(1 - (N/self.c))*N

    def euler_solution(self):
        dt = 0.15
        euler_array = [self.n_init]

        n = self.n_init
        while len(euler_array) <= 500:

            n += dt*self.differential_func(n)
            euler_array.append(n)
        return euler_array


    def plot_numerical_noise(self):
        #a = ODENumerical()
        numerical_sol = np.asanyarray(self.euler_solution())
        mu, sigma = 0, 0.5
        # creating a noise with the same dimension as the dataset
        noise = np.random.normal(mu, sigma, [np.shape(numerical_sol)[0], ])
        numerical_noisy = numerical_sol + noise
        index_array = np.arange(1,np.shape(numerical_sol)[0]+1)
        plt.scatter(index_array, numerical_noisy)
        plt.plot(numerical_sol, color = "black")
        #plt.plot(numerical_noisy, "b.")
        plt.show()

a = ODENumerical()
a.plot_numerical_noise()

