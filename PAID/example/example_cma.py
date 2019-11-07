import cma

objective_function = lambda x: (x[0]-2.0) ** 2 + (x[1]-5.0) ** 2 + (x[2] + 4.0) ** 2

xopt, es = cma.fmin2(objective_function, [10.0, 5.0, 20.0], 0.5)

print(xopt)
print(es)