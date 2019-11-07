from PAID.Curve_Fitting.creating_data import create_data, lgd
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

def find_params(data, times, signal, N_0_min, step_accuracy):
    min_error = max(times)
    fit_min = None
    r_real = None
    N_0_real = None
    values = np.arange(0+step_accuracy, 1-step_accuracy, step_accuracy)
    for i in range(len(values)):
        N_0 = round(values[i], 2)
        for j in range(len(times)):
            r_points = []
            if 0 < times[j] < 1 and 0 < data[j] < 1:
                r = -(1/times[j])*math.log(abs((N_0*(1-data[j]))/(data[j]*(1-N_0))))
                r_points.append(r)
                r_mean = round(sum(r_points)/len(r_points), 2)
                fit = fitted_curve(r_mean, times, N_0)
                error = calculate_error(signal, fit, times)
                if error < min_error:
                    min_error = error
                    fit_min = fit
                    r_real = round(r_mean, 2)
                    N_0_real = round(N_0, 2)
    plt.plot(times, fit_min)
    return min_error, fit_min, r_real, N_0_real

times = np.arange(0, 15, 0.05)
t_0 = 0.0
N_0 = 0.2
Lambda = 0.5
print("Initial N:", N_0, "\nGrowth Rate: ", Lambda)
N_0_min = 0.1
step_accuracy = 0.01

data,  signal = create_data(times, t_0, N_0, Lambda)
min_error, fit_min, r_fit, N_0_fit = find_params(data, times, signal, N_0_min, step_accuracy)

print("Estimated inital N:", N_0_fit, "\nEstimated growth rate: ",r_fit)
plt.show()

