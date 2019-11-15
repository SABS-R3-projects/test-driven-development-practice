import matplotlib.pyplot as plt
import numpy as np
from PAID.Itai_Euler.euler_num import ODENumerical
a = ODENumerical()
observation = a.euler_with_noise()
time_array = np.arange(0, len(observation), 1)

# Returns new values of lambda and c
transition_model = lambda param_array: np.random.normal(param_array,[0.5,0.5],(2,))


def prior(array_mu_sig):
    if (array_mu_sig[1] <= 0 or array_mu_sig[1] <= 0):
        return 0
    return 1

# Likelihood of data p(D|sig)
def manual_log_like_normal(param_array, data):
    #param_array = [lambda, c]
    return np.sum(-0.5 * np.log(2 * np.pi) + (data - (param_array[1]/(1 + (param_array[1]-1)*np.exp(-param_array[0]*t))))**2 for t in time_array)

def least_squares(param_array, data):
    sum = 0
    local_t = 0
    a = (param_array[1] - 1)
    analytical_array = [1]
    while len(analytical_array) <= 150:
        n = param_array[1]/(1+a*np.exp(-param_array[0]*local_t))
        analytical_array.append(n)
        local_t += 0.5
    for i, A in enumerate(analytical_array):
        sum +=  (A - data[i])**2
    return sum

#How we will accept or reject values of sigma
#Both of them have to be true
def acceptance(sig, new_sig):
    if new_sig[0] > sig[0] and new_sig[1] > sig[1]:
        return True
    else:
        accept = np.random.uniform(0, 1)
        return (accept < (np.exp(new_sig[0] - sig[0])) or accept < (np.exp(new_sig[1] - sig[1])))


def metropolis_hastings(likelihood_computer, prior, transition_model, param_init, iterations, data, acceptance_rule):
    x = param_init # [1,8]
    accepted = []
    rejected = []
    for i in range(iterations):
        x_new = transition_model(x) # Gives new array of lambda and c
        x_lik = likelihood_computer(x, data)  #Calculates the likelihood of old parameters
        x_new_lik = likelihood_computer(x_new, data) #Calculates the likelihoof of new parameters
        if (acceptance_rule(x_lik + np.log(prior(x)), x_new_lik + np.log(prior(x_new)))):
            x = x_new
            accepted.append(x_new)
        else:
            rejected.append(x_new)
    return np.array(accepted), np.array(rejected)

# Figure out what mu to use
#accepted, rejected = metropolis_hastings(manual_log_like_normal, prior, transition_model, [1, 10], 500, observation, acceptance)
accepted, rejected = metropolis_hastings(least_squares, prior, transition_model, [1, 10], 500, observation, acceptance)
print(accepted)

#print(accepted)
#print(manual_log_like_normal([1, 10], observation))
#plt.plot(accepted[0:])
#plt.show()

