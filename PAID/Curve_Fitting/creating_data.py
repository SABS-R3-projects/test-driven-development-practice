from tests.EulerMethod.test_euler import logistic_growth_dimensionless as lgd
import numpy as np
import matplotlib.pyplot as plt
import math

def create_data(times=np.arange(0, 15, 15.0/500), t=0.0, x=0.1, l=0.4):
    signal = lgd(times, t_0=t, x_0=x, Lambda=l)
    noise = np.random.normal(0, 0.05, len(times)) + signal
    # plt.plot(times, signal)
    # plt.scatter(times, noise, s=10)
    return noise
