#!/Library/Frameworks/Python.framework/Versions/3.9/bin/python3
import scipy as scp
from Tools import *
import os
import pandas as pd
import scipy.stats as scp
import numpy as np
import sys
import signal


class TimeoutError(Exception):
    pass


# Gestionnaire de signal pour déclencher l'exception TimeoutError
def timeout_handler(signum, frame):
    raise TimeoutError



def simulator_with_timeout(timeout, mu, n_initial, n_final, death_rate, fitness, param, n_sample):
    # Enregistrement du gestionnaire de signal
    signal.signal(signal.SIGALRM, timeout_handler)

    # Réglage de l'alarme avec le délai d'attente spécifié
    signal.alarm(timeout)

    try:
        result = simulate(mu, n_initial, n_final, death_rate, fitness, param, n_sample)
        signal.alarm(0)  # Désactive l'alarme si la fonction se termine à temps
        return result
    except TimeoutError:
        return pd.DataFrame()




"""Transition kernel is separately defined
    in a function so it is easy to redefine"""


def transition_kernel_rvs(mu_1,kernel_name):
    mu = 0
    while (mu < 1e-9) or (mu > 1e-5):
        if kernel_name == "uniform":
            mu = np.random.uniform(mu_1 - mu_1 / 10, mu_1 + mu_1 / 10)
        if kernel_name == "gamma":
            mu = np.random.gamma(3, mu_1 / 3)
        if kernel_name == "normal":
            mu = np.random.normal(mu_1, 5e-9)
        if kernel_name == "exponential":
            mu = np.random.exponential(mu_1, 1)
    return mu


def transition_kernel_pdf(mu_1, mu_2,kernel_name):
    if kernel_name == "uniform":
        x = mu_1 - mu_1 / 10
        y = mu_1 + mu_1 / 10
        return scp.uniform.pdf(mu_2, loc=x, scale=y - x)
    elif kernel_name == "gamma":
        shape = 3
        scale = mu_2 / shape
        return scp.gamma.pdf(mu_1, shape, scale)
    elif kernel_name == "normal":
        return scp.norm.pdf(mu_2, loc=mu_1, scale=5e-9)
    elif kernel_name == "exponential":
        return scp.expon.pdf(mu_2, scale=1 / mu_1)


"""Prior is separately defined in a function so it
    is easier to redefine"""


def prior(mu, prior_type, observed_data):
    p = 0
    steps = 0
    while (p < 1e-9) or (p > 1e-5):
        epsilon = 1e-9
        steps += 1
        if prior_type == "uniform":
            p = np.random.uniform(1e-9, 1e-5, size=1)
        if prior_type == "reciprocal":
            p = scp.loguniform.rvs(1e-9, 1e-5, size=1)
        if prior_type == "gamma":
            alpha = (np.median(observed_data / n_final) + epsilon) ** 2 / (
                        scp.iqr(observed_data / n_final) ** 2 + 1e-18)
            beta = (np.median(observed_data / n_final) + epsilon) / (scp.iqr(observed_data / n_final) ** 2 + 1e-18)
            p = np.random.gamma(alpha, 1 / beta, size=1) - epsilon
        if prior_type == "normal":
            p = np.exp(np.random.normal(np.median(np.log((observed_data + 0.01) / n_final)),
                                        np.sqrt(np.log(scp.gstd((observed_data + 0.01) / n_final) ** 2 + 1)), size=1))
        if prior_type == "exponential":
            p = np.random.exponential(np.median(observed_data) / n_final + epsilon, size=1) - epsilon
        if steps > 1000:
            return scp.loguniform.rvs(1e-9, 1e-5, size=1)
    return p


"""Metropolis-Hastings impelementation in Python
    Given a prior and a sample, uses Metropolis_Hastings
    sampling to sample from an updated version of the prior
    p : Number of simulations """


def metropolis_hastings(p, mu_prior, prior_type, observed_data, n_initial, n_final, death_rate, fitness, param,
                        n_sample, error, burn_in,kernel_name,timeout):
    current_mu = np.array([mu_prior])
    mu_values = [current_mu]
    proposed_values = [current_mu]  # <- Used to compare accepted values to all values
    accepted_values = []  # Used to avoid duplication of accepted values
    accepted = 1  # Used to calculate acceptance rate
    if (timeout!=0):
        previous_p_values = simulator_with_timeout(timeout,current_mu, n_initial, n_final, death_rate, fitness, param, n_sample)
    else:
        previous_y = simulate(current_mu, n_initial, n_final, death_rate, fitness, param, n_sample)

    for i in range(p):
        print(i)
        # We generate a candidate mu from which we generate a distribution y
        proposed_mu = transition_kernel_rvs(current_mu,kernel_name)
        proposed_values.append(proposed_mu)
        if (timeout!=0):
            proposed_y = simulator_with_timeout(timeout,proposed_mu, n_initial, n_final, death_rate, fitness, param, n_sample)
        else:
            proposed_y = simulate(proposed_mu, n_initial, n_final, death_rate, fitness, param, n_sample)

        if (proposed_y.empty):
            continue
        p_values = proposed_y.apply(lambda x: scp.kstest(x, observed_data)[1])
        previous_p_values = previous_y.apply(lambda x: scp.kstest(x, observed_data)[1])
        if option == 1:
            Iy_star = (
                        p_values.values / previous_p_values.values > 0.7).all()  # Checking if generated y is consistent with observed data

        if option == 2:
            Iy_star = (p_values > error).all()  # Checking if generated y is consistent with observed data

        if Iy_star:

            # If it is consistent, we calculate MH ratios
            a = prior(current_mu, prior_type, observed_data)
            # in case prior a equal to zero we accept the current_mu
            if a == 0:
                alpha = 1
            else:
                r1 = prior(proposed_mu, prior_type, observed_data) / a
                r2 = transition_kernel_pdf(current_mu, proposed_mu,kernel_name) / transition_kernel_pdf(proposed_mu, current_mu,kernel_name)
                alpha = min(1, r1 * r2)

            # Accept or reject proposal
            u = np.random.uniform(0, 1)
            if u <= alpha:
                current_mu = proposed_mu
                previous_y = proposed_y
                if (i > burn_in):
                    accepted_values.append(current_mu)
                    accepted += 1

            mu_values.append(current_mu)

        else:

            # If generated data is not consisted with observed data, we reject the candidate
            mu_values.append(current_mu)

    return mu_values[burn_in:], accepted_values[1:], accepted / p, proposed_values


if __name__ == '__main__':
    # Read command line arguments
    if len(sys.argv) < 6 or (sys.argv[1] == "-h"):
        print("Please provide input data file names in this order :")
        print(" - name of data file : it contains simulations (each line correspond to a data).")
        print(" - name of param file : each line contains the parameter used to do simulation of line in data file.")
        print(" - name of dt file :  contains the death rate for each line in data file.")
        print(" - name of nf file : name of file containing final number of populations for each row in data file.")
        print(" - name of fitness file : name of file containing fitness parameter used each row in data file.")
        print(" an integer refering to whether use time limit to the simulator (integer > 0 refering to seconds) or no (0 in this case).")
        print(" The expected output is two files: res.mrate , containing one value per line and maintaining correspondence with the input file.")
        sys.exit(1)

    # Read parameter file and extract samples
    data_file = sys.argv[1]
    observed_data = np.loadtxt(data_file, delimiter=',')


    # Read parameter file and extract parameter p
    param_file = sys.argv[2]
    p_values = np.loadtxt(param_file)

    # Read parameter file and extract parameter dt
    dt_file = sys.argv[3]
    dt_values = np.loadtxt(dt_file)

    Nf_file = sys.argv[4]
    N_values = np.loadtxt(Nf_file, dtype=int)

    fitness_file = sys.argv[5]
    f_values = np.loadtxt(fitness_file)
    
    timeout = int(sys.argv[6])

    n_initial = 10
    mu_vals = []
    true_mu = 0
    n_sample = 200
    mu_prior = scp.uniform.rvs(1e-11, 1e-6)
    prior_type = "exponential"
    kernel_name = "gamma"
    option = 1
    avancement = 0
    with open('res.mrate', 'a') as mu_file:
        for n_final, dt, f, param, data in zip(N_values, dt_values, f_values, p_values, observed_data):
            print("jeu donnee;", avancement)
            avancement+=1
            mu_values, accepted_values, _, _ = metropolis_hastings(timeout,2000, mu_prior, prior_type, pd.Series(data), n_initial, n_final, dt, f, param, n_sample, 0.0005, 1900, kernel_name)
            
            # Calculate the mean of accepted_values and append it to the file
            mu_mean = np.mean(accepted_values)
            mu_file.write(f"{mu_mean}\n")
            
            # Ensure that the file is flushed to disk
            mu_file.flush()
            
            # Print the current mean of accepted_values
            print(mu_mean)
