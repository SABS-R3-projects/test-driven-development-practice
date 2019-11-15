import matplotlib.pyplot as plt
import numpy as np
from PAID.Curve_Fitting.creating_data import create_data
from tests.EulerMethod.test_euler import logistic_growth_dimensionless as lgd
import random


class guess:
    def __init__(self, Lambda, N_0, times):
        self.error = None
        self.N_0 = N_0
        self.Lambda = Lambda
        self.curve = lgd(times, 0.0, N_0, Lambda)

    def calc_error(self, data, times):
        error = 0
        for i in range(len(times)):
            error += (data[i]-self.curve[i])**2
        return error


class proposal_dist:
    def __init__(self):
        self.mean = [0, 0]
        self.cov = [[5e-4, 0], [0, 5e-4]]
        self.x, self.y = np.random.multivariate_normal(self.mean, self.cov, 500).T


class target_dist:
    def __init__(self):
        self.times = np.arange(0, 15, 15.0/500)
        self.data = create_data(self.times)


class result_dist:
    def __init__(self):
        self.N_0_data = []
        self.Lambda_data = []

def main():
    data = target_dist()

    accepted_N_0, accepted_lambda = 0.9, 0.9 #initial guesses
    accepted_guess = guess(accepted_lambda, accepted_N_0, data.times)
    accepted_guess.error = accepted_guess.calc_error(data.data, data.times)

    results = result_dist()
    prop = proposal_dist()
    guess_number = 1

    while guess_number < 10000:
        next_N_0 = accepted_guess.N_0 + random.choice(prop.x)
        if 0.0 < next_N_0 < 1.0:
            next_guess = guess(accepted_guess.Lambda + random.choice(prop.y), next_N_0, data.times)
            next_guess.error = next_guess.calc_error(data.data, data.times)

            if next_guess.error < accepted_guess.error:
                accepted_guess = next_guess
                if guess_number > 2000:
                    results.N_0_data.append(accepted_guess.N_0)
                    results.Lambda_data.append(accepted_guess.Lambda)

            else:
                ratio = accepted_guess.error/next_guess.error
                draw = random.uniform(0, 1)
                if draw > ratio:
                    accepted_guess = next_guess

                    if guess_number > 2000:
                        results.N_0_data.append(accepted_guess.N_0)
                        results.Lambda_data.append(accepted_guess.Lambda)

            guess_number += 1
    return results, accepted_guess


results, accepted_guess = main()
print(len(results.N_0_data))

fig, (ax1, ax2) = plt.subplots(1, 2)
ax1.hist(results.N_0_data, bins=45)
ax2.hist(results.Lambda_data, bins=50)
plt.show()
