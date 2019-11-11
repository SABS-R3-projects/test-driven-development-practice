import numpy as np

class euler(object):
    """Class to numerically solve Initial Value Problems (IVP) with Euler's method.

    Attributes:
        model {func} -- Function to implement the right hand side of the ODE.

    Methods:
        integrate -- finds approximate solution to IVP (t_0, y_0).
    """
    def __init__(self, model):
        self.model = model

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
                y_solution[step] = y_solution[step-1] + h * self.model(time_steps-h, y_solution[step-1])

        return y_solution


