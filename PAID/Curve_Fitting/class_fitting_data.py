from PAID.Curve_Fitting.creating_data import create_data
from PAID.Curve_Fitting.euler_method import logistic_growth_euler
import numpy as np
import math


class CurveBestFit:
    def __init__(self, times, t_0, N_0, Lambda, N_accuracy, euler_times):
        self.times = times
        self.euler_times = euler_times
        self.data = create_data(times, t_0, N_0, Lambda)
        self.fit = None
        self.N_0_est = None
        self.r_est = None
        self.N_values = np.arange(0+N_accuracy, 1-N_accuracy, N_accuracy)
        self.min_error = 15**2
        self.times = times

    def minimize_error(self, euler_accuracy):
        for i in range(len(self.N_values)):
            r_points = []
            N_0 = round(self.N_values[i], 3)
            for j in range(1, len(self.times)-1):
                self.find_r_value(j, N_0, r_points)
            for r in r_points:
                t_N = {}
                r_fit = logistic_growth_euler(self.euler_times, t_N, euler_accuracy, No=N_0, C=1.0, l=r)
                r_error = self.calculate_error(r_fit)
                if r_error < self.min_error:
                    self.fit = r_fit
                    self.min_error = r_error
                    self.N_0_est = N_0
                    self.r_est = r

    def find_r_value(self, j, N_0, r_points):
        r = -(1/self.times[j])*math.log(abs((N_0*(1-self.data[j]))/(self.data[j]*(1-N_0))))
        r_points.append(round(r, 2))
        return r_points

    def calculate_error(self, r_fit):
        total_error = 0
        for i in range(len(self.times)):
            diff = (self.data[i] - r_fit[round(self.times[i], 3)]) ** 2
            total_error += diff
        return total_error


def main():
    N_accuracy = 0.1
    data_points = 100
    euler_accuracy = 0.001
    times = np.arange(0, 15, 15.0 / data_points)
    euler_times = np.arange(0.0, 15.0, euler_accuracy)
    t_0 = 0.0
    N_0 = 0.1
    Lambda = 0.8

    print("Initial N:", N_0, "\nGrowth Rate: ", Lambda)

    new_curve = CurveBestFit(times, t_0, N_0, Lambda, N_accuracy, euler_times)
    new_curve.minimize_error(euler_accuracy)

    print(new_curve.N_0_est, new_curve.r_est)


main()
