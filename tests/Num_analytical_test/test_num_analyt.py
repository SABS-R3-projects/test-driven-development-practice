import unittest
from PAID.Itai_Euler import AnalyticalSol
from PAID.Itai_Euler.euler_num import ODENumerical

class MyTestCase(unittest.TestCase):

    def setUp(self):
        analytical_sol = AnalyticalSol()
        numerical_sol = ODENumerical()
        self.analytical_array = analytical_sol.analytical_euler()
        self.numerical_array = numerical_sol.euler_solution()
        self.differential_function = numerical_sol.differential_func(4)
        self.numerical_with_noise = numerical_sol.euler_with_noise()
        self.least_squares = numerical_sol.least_squares_calculator([0.95, 10])

    def test_differential_func(self):
        self.assertTrue(self.differential_function - 0.228 <= 0.01)

    def test_numerical_vs_analytical(self):
        self.assertTrue(len(self.numerical_array)==len(self.analytical_array))
        self.assertAlmostEqual(self.numerical_array.all(), self.analytical_array.all())
        self.assertTrue(self.numerical_array[-1] < 10)

    def test_numerical_vs_numerical_with_noise(self):
        self.assertTrue(len(self.numerical_array) == len(self.numerical_with_noise))
        self.assertAlmostEqual(self.numerical_array.all(), self.numerical_with_noise.all())

    def tearDown(self):
        self.analytical_array = None
        self.numerical_array = None
        self.numerical_with_noise = None
        self.differential_function = None

if __name__== 'main':
    #Run unittest
    unittest.main()

