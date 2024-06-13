# lab.py


import pandas as pd
import numpy as np
import io
from pathlib import Path
import os


# ---------------------------------------------------------------------
# QUESTION 1
# ---------------------------------------------------------------------


def prime_time_logins(login):
    df = login.copy()
    df['Time'] = pd.to_datetime(df['Time'])
    df['Date'] = pd.to_datetime(df['Time']).dt.date
    
    def is_primetime(x):
        if (x.hour >= 16) & (x.hour <20):
            return 1
        else:
            return 0
            
    df['Time'] = df['Time'].apply(is_primetime)
    
    df = df.groupby(['Login Id', 'Date']).sum()
    df = df.groupby('Login Id').sum()#.reset_index(name='Time').set_index('Login Id')
    return df


# ---------------------------------------------------------------------
# QUESTION 2
# ---------------------------------------------------------------------


def calc_logins_per_day(series):
    total_logins = series.count()
    first_login = series.min()
    today = pd.Timestamp('2024-01-31 23:59:00')
    user_days = (today - first_login).days 
    
    # Calculate the number of logins per day
    logins_per_day = total_logins / user_days
    
    return logins_per_day

def count_frequency(login):
    df = login.copy()
    # Convert login time to datetime objects
    df['Time'] = pd.to_datetime(df['Time'])

    # Group by user and apply custom aggregator to calculate logins per day
    frequencies = df.groupby('Login Id')['Time'].agg(calc_logins_per_day)
    
    return frequencies

# ---------------------------------------------------------------------
# QUESTION 3
# ---------------------------------------------------------------------

def cookies_null_hypothesis():
    return [2]
    
                         
def cookies_p_value(N):
    observed_perfect = 235
    total_cookies = 250
    simulations = np.random.binomial(total_cookies, 0.96, N)
    
    # Calculate p-value
    p_value = np.sum(simulations <= observed_perfect) / N
    
    return p_value


# ---------------------------------------------------------------------
# QUESTION 4
# ---------------------------------------------------------------------


def car_null_hypothesis():
    return [1, 4]

def car_alt_hypothesis():
    return [2, 6]

def car_test_statistic():
    return [1, 4]

def car_p_value():
    return 4


# ---------------------------------------------------------------------
# QUESTION 5
# ---------------------------------------------------------------------

def superheroes_test_statistic():
    return [1]
            
def bhbe_col(heroes):
    df = heroes.copy()
    df['Hair color'] = df['Hair color'].str.lower()
    df['Eye color'] = df['Eye color'].str.lower()
    bhbe = (df['Hair color'].str.contains('blond', na=False)) & (df['Eye color'].str.contains('blue', na=False))
    return bhbe
    
def superheroes_observed_statistic(heroes):
    indices = np.where(bhbe_col(heroes))
    bb = heroes.iloc[indices] # finds rows of blond hair, blue eyes
    good_b = (bb['Alignment'] == 'good').sum() # num of good, blond hair, blue eyes
    num_bb = bhbe_col(heroes).sum()
    if good_b == 0 or num_bb == 0:
        stat = 0
    else:
        stat = good_b / num_bb # proportion of good / total blond hair, blue eyes
    return stat
    
def simulate_bhbe_null(heroes, N):
    good = (heroes['Alignment'] == 'good').sum() 
    total_heroes = heroes.shape[0]
    if ((good==0) or (total_heroes==0)): 
        heroes_prop = 0
    else:
        heroes_prop = good / total_heroes

    bhbe_size = bhbe_col(heroes).sum()
    
    test_statistics = [trial(heroes, heroes_prop, bhbe_size) for i in range(N)]
    
    return np.array(test_statistics)

def trial(heroes, heroes_prop, bhbe_size):
    sim = np.random.binomial(1, heroes_prop, bhbe_size)
    return sim.mean()
    
def superheroes_p_value(heroes):
    observed = superheroes_observed_statistic(heroes)
    simulation = simulate_bhbe_null(heroes, 100000)
    p_value = np.mean(simulation >= observed)
    if p_value >= 0.1:
        conclusion = 'Fail to reject'
    else:
        conclusion = 'Reject'
    return [p_value, conclusion]


# ---------------------------------------------------------------------
# QUESTION 6
# ---------------------------------------------------------------------


def diff_of_means(data, col='orange'):
    df = data.copy()
    df = df.groupby('Factory')[col].mean()
    return np.abs(df.loc['Waco'] - df.loc['Yorkville'])


def simulate_null(data, col='orange'):
    df = data.copy()
    df['Factory'] = np.random.permutation(df['Factory'])
    return diff_of_means(df, col)


def color_p_value(data, col='orange'):
    observed = diff_of_means(data)
    tests = np.array([simulate_null(data) for i in range(1000)])
    p_val = (tests >= observed).mean()
    return p_val


# ---------------------------------------------------------------------
# QUESTION 7
# ---------------------------------------------------------------------

def ordered_colors():
  return [('yellow', 0.000), ('red', 0.044), ('green', 0.224), ('purple', 0.451), ('orange', 0.98)]

  
# def ordered_colors():
    # return [('yellow', 0.000), ('red', 0.041), ('green', 0.043), ('orange', 0.044), ('purple', 0.046)]
    
# def ordered_colors():
    # return [('yellow', 0.000), ('green', 0.224), ('red', 0.044), ('purple', 0.451), ('orange', 0.98)]
    
# def ordered_colors():
    # return [('yellow', 0.000), ('red', 0.044), ('green', 0.224), ('purple', 0.451), ('orange', 0.98)]


# ---------------------------------s------------------------------------
# QUESTION 8
# ---------------------------------------------------------------------


    
def same_color_distribution():
    return (0.005, 'Fail to Reject')


# ---------------------------------------------------------------------
# QUESTION 9
# ---------------------------------------------------------------------


def perm_vs_hyp():
    return ['P', 'P', 'H', 'H', 'P']
