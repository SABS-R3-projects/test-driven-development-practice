import numpy as np
import random


def calc_post(target_dist, guess):
    freq = 0
    for i in target_dist:
        if i == guess:
            freq += 1
    return freq


target_dist = np.round(np.random.normal(70, 7, 1000))
proposal_dist = np.round(np.random.normal(0, 5, 1000))

accepted_guess = 50
accepted_post = calc_post(target_dist, accepted_guess)
guess_number = 1

while guess_number < 10000:

    next_guess = accepted_guess + random.choice(proposal_dist)
    next_post = calc_post(target_dist, next_guess)

    if next_post > accepted_post:
        accepted_guess = next_guess
        accepted_post = next_post
    else:
        decision = random.choice([1]*next_post+[0]*accepted_post)
        if decision == 0:
            pass
        elif decision == 1:
            accepted_guess = next_guess
            accepted_post = next_post
    guess_number += 1

print(accepted_guess)
