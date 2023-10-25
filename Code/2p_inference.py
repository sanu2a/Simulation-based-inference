#!/Library/Frameworks/Python.framework/Versions/3.9/bin/python3
import os
import random
import pandas as pd
import scipy.stats as scp
import numpy as np
import random as rd
import scipy.special as sc
from math import sqrt
import subprocess
import sys
import subprocess
import signal

class TimeoutError(Exception):
    pass
# Gestionnaire de signal pour déclencher l'exception TimeoutError
def timeout_handler(signum, frame):
    raise TimeoutError
"""Transition kernel is separately defined
    in a function so it is easy to redefine"""
def transition_kernel_rvs_mu(mu_1):
    mu = np.random.exponential(mu_1)
    while (mu < 1e-11) or (mu > 1e-6) :
        mu = np.random.exponential(mu_1)
    return mu


def transition_kernel_rvs_f(f_1):
#     f = np.random.normal(f_1,0.4)
#     while  (f > 2) or (f < 0.1):
#         f = np.random.normal(f_1,0.4)
#     return f
    f = np.random.exponential(f_1)
    while  (f > 2) or (f < 0.1):
        f = np.random.exponential(f_1)
    return f
    
# def transition_kernel_pdf(mu_1,mu_2,f_1,f_2):
# #    return scp.norm.pdf(mu_1,mu_2,0.04)*scp.norm.pdf(f_1,f_2,0.4)
#    return scp.norm.pdf(mu_1,mu_2,0.04)*scp.expon.pdf(f_1,f_2)

def transition_kernel_pdf(mu_1, mu_2, f_1, f_2):
    gauss_pdf = scp.expon.pdf(mu_1, loc=0, scale=mu_2)
    expon_pdf = scp.expon.pdf(f_1, loc=0, scale=f_2)
    return gauss_pdf * expon_pdf

"""Prior is separately defined in a function so it
    is easier to redefine"""
def Prior(mu,f,prior_type_mu,prior_type_f):
    if prior_type_mu == "log_uniform" :
        a_mu = scp.loguniform.pdf(mu,1e-11, 1e-6)
    elif prior_type_mu == "uniform" :
        a_mu = scp.uniform.pdf(mu,1e-11,1e-6)
    elif prior_type_mu == "gamma":
        a_mu = scp.gamma.pdf(mu, alpha, scale=1/beta)
    elif prior_type_mu == "exponential":
        a_mu = scp.expon.pdf(np.median(observed_data)/n_final)
    else:
        raise ValueError("Invalid prior_type_mu, must be one of 'log_uniform', 'uniform', 'gamma'")
    
    if prior_type_f == "log_uniform" :
        a_f = scp.loguniform.pdf(f,0.1, 2)
    elif prior_type_f == "uniform" :
        a_f = scp.uniform.pdf(f,0.1,2)
    elif prior_type_f == "gamma":
        a_f = scp.gamma.pdf(f, alpha, scale=1/beta)

    else:
        raise ValueError("Invalid prior_type_f, must be one of 'log_uniform', 'uniform', 'gamma'")
    
    return a_mu * a_f 
def metropolis_hastingsVariant(timeout,p, mu_0,prior_type_mu,prior_type_f, observed_data, n_initial, n_final, f_0, dt, param, n_sample):
    current_val = [mu_0,f_0]
    if (timeout!=0):
        datacurrent = simulator_with_timeout(timeout,mu_0, n_initial, n_final, dt, f_0, param, n_sample)
    else:
        datacurrent = simulator(mu_0, n_initial, n_final, dt, f_0, param, n_sample)
    mu_values = [current_val[0]]
    f_values = [current_val[1]]
    proposed_values = [current_val] # <- Used to compare accepted values to all values
    
    for i in range(p):
        print(i)
        # We generate a candidate mu from which we generate a distribution y
        proposed_val = (transition_kernel_rvs_mu(current_val[0]),transition_kernel_rvs_f(current_val[1]))
        proposed_values.append(proposed_val)
        if (timeout!=0):
            dataproposed = simulator_with_timeout(timeout,proposed_val[0], n_initial, n_final, dt, proposed_val[1], param, n_sample)
        else :
            dataproposed = simulator(proposed_val[0], n_initial, n_final, dt, proposed_val[1], param, n_sample)
        if (dataproposed == None):
            continue
        ksProposed, p_values = scp.ks_2samp(observed_data, dataproposed)
        ksCurrent, p_values = scp.ks_2samp(observed_data, datacurrent)
     
        prior2 = Prior(current_val[0],current_val[1],prior_type_mu,prior_type_f)
        prior1 = Prior(proposed_val[0],proposed_val[1],prior_type_mu,prior_type_f)
        trk1 =  transition_kernel_pdf(current_val[0],proposed_val[0],current_val[1],proposed_val[1])
        trk2 = transition_kernel_pdf(proposed_val[0],current_val[0],proposed_val[1],current_val[1])
        if ksProposed == 0:
            alpha =1
        elif ksCurrent/ksProposed>=1:
            # in case prior or ksproposed a equal to zero we accept the couple (current_mu , current_dr)
            if prior2 == 0 or trk2 ==0:
                alpha = 1

            else:
                r1 = prior1 / prior2
                r2 = (trk1*ksCurrent)/ (trk2*ksProposed)
                #we can also use expential
                #r2 = (trk1*np.exp(-ksProposed))/ (trk2*np.exp(-ksCurrent))
                alpha = min(1,r1*r2)

            # Accept or reject proposal
            u = np.random.uniform(0, 1)
            if u <= alpha:
                current_val[0] = proposed_val[0]
                current_val[1] = proposed_val[1]
                datacurrent = dataproposed
        mu_values.append(current_val[0])
        f_values.append(current_val[1])
            

    return mu_values,f_values ,proposed_values


def simulator(mu, n_initial, n_final, death_rate, fitness, param, n_sample):
    parameters = f"{n_initial},{n_final} {death_rate} {float(mu)} {fitness} {param} {n_sample} 1"
    command = ['./atreyu_forward_simulator'] + parameters.split()
    process = subprocess.Popen(command,stdout=subprocess.PIPE,stderr=subprocess.PIPE,text=True)
    stdout, _ = process.communicate()
    output_lines = stdout.strip().split('\n')
    int_list = [int(line.split()[0]) for line in output_lines]
    return int_list
def simulator_with_timeout(timeout, mu, n_initial, n_final, death_rate, fitness, param, n_sample):
    # Enregistrement du gestionnaire de signal
    signal.signal(signal.SIGALRM, timeout_handler)
    
    # Réglage de l'alarme avec le délai d'attente spécifié
    signal.alarm(timeout)

    try:
        result = simulator(mu, n_initial, n_final, death_rate, fitness, param, n_sample)
        signal.alarm(0)  # Désactive l'alarme si la fonction se termine à temps
        return result
    except TimeoutError:
        return None



if __name__ == '__main__':
    # Read command line arguments
    if len(sys.argv) < 5 or (sys.argv[1] == "-h"):
        print("Please provide input data file names in this order :")
        print(" - name of data file : it contains simulations (each line correspond to a data).")
        print(" - name of param file : each line contains the parameter used to do simulation of line in data file.")
        print(" - name of dt file :  contains the death rate for each line in data file.")
        print(" - name of nf file : it contains final number of populations for each row in data file.")
        print(" an integer refering to whether use time limit to the simulator (integer > 0 refering to seconds) or no (0 in this case).")
        print(" The expected output is two files: result.mrate and result.f, each containing one value per line and maintaining correspondence with the input file.")
        sys.exit(1)

    # Read parameter file and extract samples
    data_file = sys.argv[1]
    observed_data = []
    with open(data_file) as f:
        for line in f:
            observed_data.append([int(x) for x in line.strip().split(',')])

    # Read parameter file and extract parameter p
    param_file = sys.argv[2]
    p_values = np.loadtxt(param_file)

    # Read parameter file and extract parameter dt
    dt_file = sys.argv[3]
    dt_values = np.loadtxt(dt_file)

    Nf_file = sys.argv[4]
    N_values = np.loadtxt(Nf_file, dtype=int)
    timeout = int(sys.argv[5])
    n_initial = 10

    # Initialize empty lists to store mu and f values
    mu_vals = []
    f_vals = []
    mu_0 = (1e-11+ 1e-6)/2 #la valeur centrale de l'intervalle de mu
    f_0 = (0.1 + 2)/2 #la valeur centrale de l'intervalle de f  
    avancement = 0
    # Loop over input parameters
    avancement = 0
    with open('result.mrate', 'a') as mu_file, open('result.f', 'a') as f_file:
        for n_final, dt, p, data in zip(N_values, dt_values, p_values, observed_data):
            # Call grid_sampling_algo function and save the best mu and f values
            avancement += 1
            if (avancement < 21):
                continue
            print("jeu de donné ", avancement)
            best_mu, best_f, proposed_values = metropolis_hastingsVariant(timeout,10000, mu_0, "log_uniform", "log_uniform", data, n_initial, n_final, f_0, dt, p, 150)
            
            # Append the current best mu and f values to the files
            mu_file.write(f"{best_mu[-1]}\n")
            f_file.write(f"{best_f[-1]}\n")
            
            # Ensure that the files are flushed to disk
            mu_file.flush()
            f_file.flush()
            
            # Print the current best mu and f values
            print(best_mu[-1], best_f[-1])
