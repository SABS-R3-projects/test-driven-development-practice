import numpy as np

class DataGenerator(object):
    def __init__(self):
        pass

    def generate_data(self, exact_solution, scale):
        number_data_points = len(exact_solution)
        noise = np.random.normal(loc=0.0, scale=scale, size=number_data_points)
        return exact_solution + noise