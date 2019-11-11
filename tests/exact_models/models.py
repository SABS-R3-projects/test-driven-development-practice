import numpy as np

def logistic_growth_dimensionless(times, t_0=0.0, x_0=0.1, Lambda=1.0):
    '''
    Example ODE for testing that we know the analytical solution of. This version rescales N by C, i.e. x:= N/C.

    args:
     times: np.array of time points.
     t_0: Initial time.
     x_0: Initial population size.
     Lambda: exponential growth factor.

    return:
     np.array of logistic function evaluated at times 'times + t_0'.
    '''
    times -= t_0 # applying time off-set
    result = x_0 / (x_0 + (1 - x_0) * np.exp(-Lambda * times))
    return result


def logistic_growth(times, t_0=0.0, N_0=1.0, C=100.0, Lambda=1.0):
    '''
    Example ODE for testing that we know the analytical solution of.

    args:
     times: np.array of time points.
     t_0: Initial time.
     N_0: Initial population size.
     C: Carrying capacity / maximal population size.
     Lambda: exponential growth factor.

    return:
     np.array of logistic function evaluated at times 'times + t_0'.
    '''
    times -= t_0 # applying time off-set
    result = C / (1 + (C / N_0 - 1) * np.exp(-Lambda * times))
    return result