import matplotlib.pyplot as plt
import numpy as np
import cma
class ODENumerical:

    def __init__(self, lambda_val = 0.095, n_init = 1, c = 10):
        self.lambda_val = lambda_val
        self.n_init = n_init
        self.c = c
        self.dt = 0.5
        self.minimization_sol = []
        self.numerical_noisy = []

    def differential_func(self, N):
        return self.lambda_val*(1 - (float(N)/self.c))*N

    def euler_solution(self):
        euler_array = [self.n_init]
        n = self.n_init
        while len(euler_array) <= 150:
            n += self.dt*self.differential_func(n)
            euler_array.append(n)
        euler_array = np.asanyarray(euler_array)
        return euler_array

    def euler_with_noise(self):
        numerical_sol = self.euler_solution()
        mu, sigma = 0, 0.5
        # creating a noise with the same dimension as the dataset
        noise = np.random.normal(mu, sigma, [np.shape(numerical_sol)[0], ])
        numerical_noisy = numerical_sol + noise
        return numerical_noisy

    def plot_func(self):
        euler_array = self.euler_solution()
        euler_with_noise = self.euler_with_noise()
        x_values = np.arange(0,len(euler_array),1)
        plt.scatter(x_values,euler_with_noise, color = "yellow", edgecolors="black")
        #plt.scatter(x_values,euler_array, color = "red")
        plt.ylabel("N values")
        plt.xlabel("Time scale")
        plt.legend(["Euler solution + Noise", "Numerical Solution"])
        plt.title("Solving the Logistic function with Euler's Method")
        plt.show()

    def scoring_func(self):
        numerical_noisy = self.euler_with_noise()
        clean_data = self.euler_solution()
        objective_function = lambda array: sum((numerical_noisy[i]-array[0]*array[1]*(1 - (float(N)/self.c))*N)**2 for i,N in enumerate(clean_data))
        self.minimization_sol, es = cma.fmin2(objective_function, [0.095, 0.5], 0.5)
        print(self.minimization_sol)

a = ODENumerical()
a.scoring_func()
a.plot_func()
