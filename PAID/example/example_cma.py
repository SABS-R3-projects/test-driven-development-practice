import cma
import numpy as np

objective_function = lambda x: (x[0]-2.0) ** 2 + (x[1]-5.0) ** 2 + (x[2] + 4.0) ** 2

xopt, es = cma.fmin2(objective_function, [10.0, 5.0, 20.0], 0.5)

print('this is the solution: ', xopt)
print('so what is this ', es)

print(np.array([[1,2,3],[1,2,3]]).ndim)