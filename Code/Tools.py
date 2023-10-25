import os
import pandas as pd
import matplotlib.pyplot as plt
import scipy.stats as scp
import numpy as np
import random as rd
import seaborn as sns

# a data frame containig the different results of simulation
def simulate(mu_values, n_initial, n_final,death_rate,fitness, param,n_sample):
    
    generated_df = pd.DataFrame(columns=mu_values)
    for mu in mu_values:
        parameters = str(n_initial) + ',' + str(n_final) + ' ' + str(death_rate) + ' ' + str(float(mu)) +  ' ' + str(fitness) + ' ' + str(param) + ' ' + str(n_sample) + ' ' + str(1)
        os.system('./atreyu_forward_simulator ' + parameters + '  > generated_data')
        df2 = pd.read_csv("./generated_data", header=None)
        df2 = df2[0].apply(lambda x : x.split()[0]).astype(int)
        # reading generated data with simulator atreyu
        generated_df[mu] = df2
    os.system('rm generated_data')
    return generated_df

def simulate_dth_rate(mu_value, n_initial, n_final,death_rates,fitness, param,n_sample):
    
    generated_df = pd.DataFrame(columns=death_rates)
    for dt in death_rates:
        parameters = str(n_initial) + ',' + str(n_final) + ' ' + str(dt) + ' ' + str(float(mu_value)) +  ' ' + str(fitness) + ' ' + str(param) + ' ' + str(n_sample) + ' ' + str(1)
        os.system('./atreyu_forward_simulator ' + parameters + '  > generated_data')
        df2 = pd.read_csv("./generated_data", header=None)
        df2 = df2[0].apply(lambda x : x.split()[0]).astype(int)
        # reading generated data with simulator atreyu
        generated_df[dt] = df2
    os.system('rm generated_data')
    return generated_df

def mann_whiteney_test(observed_df,observed_data):
    #p_values of the different values of mu using Mann–Whitney U test inedxed by values of mu
    p_values_mnht = observed_df.apply(lambda x : scp.mannwhitneyu(x,observed_data,alternative='two-sided')).loc[1,:]
    mu = p_values_mnht.loc[p_values_mnht >= 0.05]
    if (len(mu.index)==0):
        return pd.DataFrame(), 0
    return mu, max(p_values_mnht)


def kolm_smirnov_test(observed_df,observed_data):
    #p_values of the different values of mu using kolmogorov-Smirnov test inedxed by values of mu
    p_values_kst = observed_df.apply(lambda x : scp.kstest(x,observed_data)).loc[1,:]
    mu = p_values_kst.loc[p_values_kst >= 0.05]
    if (len(mu.index)==0):
        return None, 0
    return mu, max(p_values_kst)

def plot_cdf(distribution_kst,distribution_mwt,observed_data, ax):
    sns.ecdfplot(distribution_kst, ax=ax, legend=True)
    sns.ecdfplot(distribution_mwt, ax=ax, legend=True)
    sns.ecdfplot(observed_data, ax=ax, legend=True)
    ax.set_ylim([0,1.1])
    p95 = int(max(np.percentile(distribution_kst,95), np.percentile(distribution_mwt,95),np.percentile(observed_data,95)))
    ax.set_xlim([0,p95])
    ax.legend(["kolmogorov test","Mann-Whithney","unknown mutation rate"])
    ax.set_xlabel('Number of mutants')
    ax.set_ylabel('Cumulative probability')

def plot_pdf(distribution_kst,distribution_mwt,observed_data, ax):
    d1 = np.bincount(distribution_kst)
    d2 = np.bincount(distribution_mwt)
    d3 = np.bincount(observed_data)
    dw1 = d1 / sum(d1) 
    dw2 = d2 / sum(d2)
    dw3 = d3 / sum(d3)
    ax.plot(dw1)
    ax.plot(dw2)
    ax.plot(dw3)
    p95 = int(max(np.percentile(distribution_kst,95), np.percentile(distribution_mwt,95),np.percentile(observed_data,95)))
    ax.set_xlim([0,p95])
    ax.set_xlabel('Number of mutants')
    ax.set_ylabel('Probability')
    ax.legend(["kolmogorov test","Mann-Whithney","unknown mutation rate"])


def mann_whiteney_test_mean(observed_df,observed_data):
    #p_values of the different values of mu using Mann–Whitney U test inedxed by values of mu
    p_values_mnht = observed_df.apply(lambda x : scp.mannwhitneyu(x,observed_data)).loc[1,:]
    return p_values_mnht.loc[p_values_mnht >= 0.05].index.tolist()
def kolm_smirnov_test_mean(observed_df,observed_data):
    #p_values of the different values of mu using kolmogorov-Smirnov test inedxed by values of mu
    p_values_kst = observed_df.apply(lambda x : scp.kstest(x,observed_data)).loc[1,:]
    return p_values_kst.loc[p_values_kst >= 0.05].index.tolist()

