import matplotlib.pyplot as plt
import json
import numpy as np

errors = json.load(open('N_0_error', 'r'))

x_ = []
y_ = []
for i in errors:
    x_.append(float(i))
    y_.append(errors[i])


plt.plot(x_, y_)
plt.xticks(np.arange(0.0, 1.0, step=0.1))
plt.show()