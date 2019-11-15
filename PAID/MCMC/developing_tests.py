from PAID.MCMC.Markov_Chain_2D import *


data = target_dist()
new = guess(0.15, 0.35, data.times)
new.curve = np.zeros(len(data.times)) + 0.1
data.data = np.zeros(len(data.times))
new.error = new.calc_error(data.data, data.times)

print('new error', new.error, 'length of change', len(data.times))
print(new.error)
new_new = new
print(new_new.error)