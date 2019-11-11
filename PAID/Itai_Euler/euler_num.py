import matplotlib.pyplot as plt
import numpy as np
import cma
class ODENumerical:

    def __init__(self, lambda_val = 0.095, n_init = 1, c = 100):
        self.lambda_val = lambda_val
        self.n_init = n_init
        self.c = c
        self.dt = 0.5
        self.minimization_sol = []
        self.euler_solution_array = []
        self.euler_with_noise_array = []

    def differential_func(self, N):
        return self.lambda_val*(1 - (float(N)/self.c))*N

    def euler_solution(self):
        euler_array = [self.n_init]
        n = self.n_init
        while len(euler_array) <= 250:
            n += self.dt*self.differential_func(n)
            euler_array.append(n)
        euler_array = np.asanyarray(euler_array)
        self.euler_solution_array = euler_array
        return euler_array

    def euler_with_noise(self):
        numerical_sol = self.euler_solution()
        mu, sigma = 0, 5
        # creating a noise with the same dimension as the dataset
        noise = np.random.normal(mu, sigma, [np.shape(numerical_sol)[0], ])
        numerical_noisy = numerical_sol + noise
        self.euler_with_noise_array = numerical_noisy
        return numerical_noisy

    def plot_func(self):
        euler_array = self.euler_solution()
        euler_with_noise = self.euler_with_noise()
        x_values = np.arange(0,len(euler_array),1)
        plt.scatter(x_values,euler_with_noise, color = "yellow", edgecolors="black")
        plt.plot(euler_array, color = "red")
        plt.ylabel("N values")
        plt.xlabel("Time scale")
        plt.legend(["Euler solution + Noise", "Numerical Solution"])
        plt.title("Solving the Logistic function with Euler's Method")
        plt.show()

    def least_squares_calculator(self, array):
        self.euler_solution()
        least_squares = 0
        first_term = self.n_init
        for i, j in enumerate(self.euler_solution_array):
            if i == 0:
                least_squares += (self.euler_with_noise_array[0] - self.euler_solution_array[0])**2
            else:
                first_term += self.dt*array[0]*(1 - (float(first_term)/array[1]))*first_term
                least_squares += (self.euler_with_noise_array[i] - first_term) ** 2
        return least_squares

    def scoring_func(self):
        x_values = np.arange(0, len(self.euler_solution_array), 1)
        self.minimization_sol, es = cma.fmin2(self.least_squares_calculator,[0.095, 10], 0.5)
        print(self.minimization_sol)
        print("="*81)
        print("Updating the values of lambda and c")
        self.lambda_val = self.minimization_sol[0]
        self.c = self.minimization_sol[1]
        new_fit = self.euler_solution()
        plt.scatter(x_values, self.euler_with_noise_array, color = "yellow", edgecolors="black")
        plt.plot(x_values, new_fit, "r")
        plt.ylabel("N values")
        plt.xlabel("Time scale")
        plt.legend([ "CMA Fit", "Noisy distribution"])
        plt.title("Plotting best fit using CMA-ES Algorithm")
        plt.show()

a = ODENumerical()
print(a.lambda_val)
print(a.c)
a.plot_func()
a.scoring_func()
print(a.lambda_val)
print(a.c)