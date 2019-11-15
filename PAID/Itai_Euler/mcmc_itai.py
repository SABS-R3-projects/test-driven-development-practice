import matplotlib.pyplot as plt
import numpy as np
from PAID.Itai_Euler.euler_num import ODENumerical
a = ODENumerical()
observation = a.euler_with_noise()
time_array = a.times_array()
# Returns new values of lambda and c
transition_model = lambda param_array: np.random.normal(param_array,[0.5,0.5],(2,))

# Likelihood of data
def likelihood (param, observation):
    #param = [lambda, c]
    total_sum = 0
    for i, N in enumerate(observation):
        total_sum += ((N - (param[1]/(1 + (param[1]-1)*np.exp(-param[0]*time_array[i]))))**2)/2
    return -1* total_sum

def prior(param):
    if (param[0] <= 0 or param[1] <= 0):
        return -np.inf
    return 1

#How we will accept or reject values of sigma
#Both of them have to be true
def acceptance(old_likelihood, new_likehood):
    if new_likehood - old_likelihood >= 0:
        return True
    else:
        accept = np.random.uniform(0, 1)
        if np.log(accept) < (new_likehood - old_likelihood):
            return True
        else:
            return False


def metropolis_hastings(likelihood_computer, prior, transition_model, param_init, iterations, data, acceptance_rule):
    x = param_init # [1,8]
    accepted = []
    rejected = []
    for i in range(iterations):
        x_new = transition_model(x) # Gives new array of lambda and c
        x_lik = likelihood(x, data)  #Calculates the likelihood of old parameters
        x_new_lik = likelihood_computer(x_new, data) #Calculates the likelihoof of new parameters
        if (acceptance_rule(x_lik + prior(x), x_new_lik + prior(x_new))):
            x = x_new
            accepted.append(x_new)
        else:
            rejected.append(x_new)
    return np.array(accepted), np.array(rejected)

#accepted, rejected = metropolis_hastings(manual_log_like_normal, prior, transition_model, [1, 10], 500, observation, acceptance)
accepted, rejected = metropolis_hastings(likelihood, prior, transition_model, [1, 10], 100000, observation, acceptance)
print(accepted)
plt.hist(accepted[:, 0])
plt.show()






