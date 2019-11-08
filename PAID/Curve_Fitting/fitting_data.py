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
        diff = (signal[i]-fit[i])**2
        total_error += diff
    return total_error

def find_params(data, times, signal, step_accuracy):
    min_error = 15**2
    fit_min = None
    r_real = None
    N_0_real = None
    all_errors = {}
    N_values = np.arange(0+step_accuracy, 1-step_accuracy, step_accuracy)
    for i in range(len(N_values)):
        r_points = []
        N_0 = round(N_values[i], 3)
        for j in range(1, len(times)-1):
            if 0 < times[j] < 1 and 0 < data[j] < 1:
                r = -(1/times[j])*math.log(abs((N_0*(1-data[j]))/(data[j]*(1-N_0))))
                r_points.append(r)
        r_mean = round(sum(r_points)/len(r_points), 2)
        print(N_0, r_mean)
        N_lst = []
        fit = logistic_growth_euler(times, N_lst, step_accuracy, No=N_0, C=1.0, l=r_mean)
        error = calculate_error(data, fit, times[:-1])
        all_errors[N_0] = error
        if error < min_error:
            min_error = error
            fit_min = fit
            r_real = round(r_mean, 2)
            N_0_real = round(N_0, 2)
    plt.plot(times, fit_min)
    return min_error, fit_min, r_real, N_0_real, all_errors

step_accuracy = 0.001
times = np.arange(0, 15, step_accuracy)
t_0 = 0.0
N_0 = 0.2
Lambda = 0.4
print("Initial N:", N_0, "\nGrowth Rate: ", Lambda)
N_0_min = 0.1

data,  signal = create_data(times, t_0, N_0, Lambda)
min_error, fit_min, r_fit, N_0_fit, all_errors = find_params(data, times, signal, step_accuracy)

print("Estimated inital N:", N_0_fit, "\nEstimated growth rate: ",r_fit)
json.dump(all_errors, open('N_0_error', 'w'))
plt.show()

