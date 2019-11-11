from tests.EulerMethod.test_euler import logistic_growth_dimensionless as lgd
import numpy as np
import matplotlib.pyplot as plt
import math

def create_data(times, t, x, l):
    signal = lgd(times, t_0=t, x_0=x, Lambda=l)
    noise = np.random.normal(0, 0.05, len(times)) + signal
    plt.plot(times, signal)
    plt.scatter(times, noise, s=10)
    return noise
