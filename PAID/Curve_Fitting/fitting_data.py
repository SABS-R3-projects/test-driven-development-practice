from PAID.Curve_Fitting.creating_data import create_data, lgd
from PAID.Curve_Fitting.euler_method import logistic_growth_euler
import json
import numpy as np
import matplotlib.pyplot as plt
import math

def fitted_curve(r_mean, times, N_0):
    fit = lgd(times, Lambda=r_mean, x_0 = N_0)
    return fit

def calculate_error(signal, fit, times):
    total_error = 0
    for i in range(len(times)):
#        print(i, times[i], signal[i], fit[round(times[i], 3)])
        diff = (signal[i]-fit[round(times[i], 3)])**2
        total_error += diff
    return total_error

def find_params(data, times, N_accuracy, euler_accuracy, euler_times):
    min_error = 15**2
    fit_min = None
    r_real = None
    N_0_real = None
    N_values = np.arange(0+N_accuracy, 1-N_accuracy, N_accuracy)
    for i in range(len(N_values)):
        r_points = []
        N_0 = round(N_values[i], 3)
        for j in range(1, len(times)-1):
            r = -(1/times[j])*math.log(abs((N_0*(1-data[j]))/(data[j]*(1-N_0))))
            r_points.append(round(r, 2))
        for r in r_points:
            t_N = {}
            fit = logistic_growth_euler(euler_times, t_N, euler_accuracy, No=N_0, C=1.0, l=r)
            r_error = calculate_error(data, fit, times[:-1])
            if r_error < min_error:
                min_error = r_error
                r_real = r
                N_0_real = N_0
                fit_min = [fit[i] for i in fit]
    plt.plot(euler_times, fit_min)
    return min_error, fit_min, r_real, N_0_real

N_accuracy = 0.1
data_points = 100
euler_accuracy = 0.001
times = np.arange(0, 15, 15.0/data_points)
euler_times = np.arange(0.0, 15.0, euler_accuracy)
t_0 = 0.0
N_0 = 0.2
Lambda = 0.4
print("Initial N:", N_0, "\nGrowth Rate: ", Lambda)
N_0_min = 0.1

data,  signal = create_data(times, t_0, N_0, Lambda)
min_error, fit_min, r_fit, N_0_fit = find_params(data, times, N_accuracy, euler_accuracy, euler_times)

print("Estimated inital N:", N_0_fit, "\nEstimated growth rate: ",r_fit)
plt.show()

