import numpy as np

class euler(object):
    def __init__(self, func):
        self.func = func

    def integrate(self, h, t_0, t_final, y_0):
        """Method to find the numerical solution of the ODE in the interval [t_0, t_final].

        Arguments:
            h {float} -- step size of integrator.
            t_0 {float} -- initial time of integration.
            t_final {float} -- final time of integration.
            y_0 {float} -- initial value of state variable.

        Returns:
            y_solution {np.array} -- numerical solution to ODE.
        """

        times = np.arange(t_0, t_final, h)
        y_solution = np.empty(len(times))
        for step, time_steps in enumerate(times):
            if step == 0:
                y_solution[step] = y_0
            else:
                y_solution[step] = y_solution[step-1] + h * self.func(time_steps-h, y_solution[step-1])

        return y_solution

def logistic_growth_euler(time_arr, N_lst, time_step, No=1.0, C=100.0, l=1.0):
    Nt = No
    t = 0
    N_lst.append(No)
    t += time_step
    while t < time_arr[-1]:
        N_ = l * Nt * (1 - (Nt / C))
        N_lst.append(N_lst[-1] + (time_step * N_))
        Nt = N_lst[-1]
        t += time_step
    return N_lst

