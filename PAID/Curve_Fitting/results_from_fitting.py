import json
import matplotlib.pyplot as plt

class result:
    def __init__(self, N_0):
        self.N = N_0
        self.r_errors = {}

data = json.load(open("error_data", 'r'))

fig, (ax1, ax2) = plt.subplots(1, 2)
lst = []
for i in data:
    obj = result(i)
    for j in data[i]:
        obj.r_errors[j[0]] = j[1]
    lst.append(obj)


newlst = []

for n in range(len(lst)):
    for i in lst[n].r_errors:
        ax1.scatter(i, lst[n].r_errors[i])
        if -2 < i < 2:
            newlst.append(i)


ax2.hist(newlst, bins=50)


plt.show()

