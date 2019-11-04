import numpy as np

class euler(object):
    def __init__(self, func):
        self.func = func

    def integrate(self, dt, t_0, t_final, y_0):
        times = np.arange(t_0, t_final, dt)
        y_solution = np.empty(len(times))
        for step, time_steps in enumerate(times):
            if step == 0:
                y_solution[step] = y_0
            else:
                y_solution[step] = y_solution[step-1] + dt * self.func(time_steps-dt, y_solution[step-1])

        return y_solution
