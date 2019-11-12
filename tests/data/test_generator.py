import unittest
import numpy as np

from PAID.data.generator import DataGenerator

def test_generate_data():
    exact_array = np.linspace(0, 10, 100)
    scale = 1.0
    gen = DataGenerator()
    data = gen.generate_data(exact_array, scale)

    assert len(exact_array) == len(data)
