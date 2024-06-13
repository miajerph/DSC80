# lab.py


import pandas as pd
import numpy as np
import io
from pathlib import Path
import os

from datetime import date

# ---------------------------------------------------------------------
# QUESTION 1
# ---------------------------------------------------------------------

# helper to change to timestamp object 
def change_time(time):
    return pd.to_datetime(time, format='%Y-%m-%d')

def prime_time_logins(login):
    copy = login.copy(deep=True)
    copy['Time'] = copy['Time'].apply(change_time)
    valid_logins = copy[(copy['Time'].dt.hour >= 16) & (copy['Time'].dt.hour < 20)]
    counts = pd.DataFrame(valid_logins.groupby('Login Id')['Time'].count())
    unique_ids = pd.DataFrame(login['Login Id'].unique(), index=None, columns=['Login Id'])
    final = counts.merge(unique_ids, on='Login Id', how='outer').fillna(0)
    final = final.set_index('Login Id')
    return final

# ---------------------------------------------------------------------
# QUESTION 2
# ---------------------------------------------------------------------


def count_frequency(login):
    copy = login.copy(deep=True)
    copy['Time'] = copy['Time'].apply(change_time)
    total_logs = copy.groupby('Login Id').count()
    today = pd.Timestamp('2024-01-31 23:59:00')
    first_login_date = copy.groupby('Login Id')['Time'].min().dt.date
    days_on_site = (today.date() - first_login_date).dt.days 


    freq = total_logs['Time'] / days_on_site
    return freq


# ---------------------------------------------------------------------
# QUESTION 3
# ---------------------------------------------------------------------


def cookies_null_hypothesis():
    return [1, 2]
                         
def cookies_p_value(N):

    # Simulate drawing samples from a binomial distribution
    null_samples = np.random.binomial(n=N, p=0.96, size=N) / N
    
    extreme_samples = np.sum(null_samples < 0.96)
    p_value = extreme_samples / N
    
    return p_value

# ---------------------------------------------------------------------
# QUESTION 4
# ---------------------------------------------------------------------


# valid null hypothesis will prove what the dealership said
# claims that their tires are so good, they will bring a Toyota Highlander from 60 MPH to a complete stop in under 106 feet, 95% percent of the time.
# valid null hypothesis would be 1 and 4?

# valid alt hypothesis would be 2 6? 

def car_null_hypothesis():
    return [1, 4]

def car_alt_hypothesis():
    return [2, 6]

def car_test_statistic():
    return [1, 4]

def car_p_value():
    return 3


# ---------------------------------------------------------------------
# QUESTION 5
# ---------------------------------------------------------------------


def superheroes_test_statistic():
    return [1, 4]
    
def bhbe_col(heroes):
    copy = heroes.copy(deep=True)
    copy['Hair color'] = heroes['Hair color'].str.lower()
    copy['Eye color'] = heroes['Eye color'].str.lower()
    blond = copy['Hair color'].str.contains('blond')
    blue = copy['Eye color'].str.contains('blue')
    blond_blue = blond & blue
    return blond_blue

def superheroes_observed_statistic(heroes):
    # test stat is the proportion of good heroes among blond and blue eyed heroes
    return test_stat(heroes)
    
#     return (bhbe_col(heroes).sum()) / bhbe_col(heroes).shape[0]


# helper to calculate test stat given bhbe_col series
def test_stat(df):
    indices = np.where(bhbe_col(df))
    bb = df.iloc[indices] # finds rows of blond hair, blue eyes
    good_b = (bb['Alignment'] == 'good').sum() # num of good, blond hair, blue eyes
    num_bb = bhbe_col(df).sum()
    if good_b == 0 or num_bb == 0:
        stat = 0
    else:
        stat = good_b / num_bb # proportion of good / total blond hair, blue eyes
    return stat


# helper for samples
def create_samples(df, N):
#     # Repeat the DataFrame N times using broadcasting
#     repeated_df = np.repeat(df.values[None, :, :], N, axis=0)
    
#     # Shuffle the rows within each sample
#     np.apply_along_axis(np.random.shuffle, axis=1, arr=repeated_df)
    
#     # Convert the shuffled arrays back to DataFrames
#     repeated_dfs = map(lambda sample: pd.DataFrame(sample, columns=df.columns), repeated_df)
    
#     return list(repeated_dfs)
    # Generate random indices for sampling with replacement
    sample_indices = np.random.choice(len(df), size=(N, len(df)), replace=True)
    
    # Use advanced indexing to create the sample array
    sample_array = df.values[sample_indices]
    
    # Convert each array slice into a DataFrame
    sample_dfs = map(lambda sample: pd.DataFrame(sample, columns=df.columns), sample_array)
    
    return list(sample_dfs)

def simulate_bhbe_null(heroes, N):
       
#     test_stats = np.broadcast_to(samples(heroes, N), (N, N))

#     return test_stats
    test_statistics = [samples(heroes) for i in range(N)]
    return np.array(test_statistics)

#     samples = create_samples(heroes, N)
#     test_statistics = np.array(list(map(test_stat, samples)))

   
def samples(heroes):
    sample_indices = np.random.choice(len(heroes), size=(len(heroes)), replace=True)
    sample_array = heroes.iloc[sample_indices]
    test_statistics = test_stat(sample_array)
    return test_statistics

def superheroes_p_value(heroes):
    observed = superheroes_observed_statistic(heroes)
    simulation = simulate_bhbe_null(heroes, 1000)
    p_value = np.mean(simulation >= observed)
    return [p_value, 'Fail to reject']


# ---------------------------------------------------------------------
# QUESTION 6
# ---------------------------------------------------------------------


def diff_of_means(data, col='orange'):
    copy = data.copy(deep=True)
    avg = copy.groupby('Factory')[col].mean()
    abs_diff = abs(avg[0] - avg[1])
    return abs_diff


def simulate_null(data, col='orange'):
    shuffled_skittles = data.copy(deep=True)
    shuffled_skittles['Factory'] = np.random.permutation(shuffled_skittles['Factory'])
    return diff_of_means(shuffled_skittles, col)



def color_p_value(data, col='orange'):
    num_tests = 1000
    observed = simulate_null(data)
    tests = np.array([simulate_null(data) or 0 for i in range(num_tests)])
    p_val = (np.abs(tests) >= np.abs(observed)).mean()
    return p_val


# ---------------------------------------------------------------------
# QUESTION 7
# ---------------------------------------------------------------------


def ordered_colors():
    return [('yellow', 0.000), ('red', 0.181), ('green', 0.407), ('purple', 0.451), ('orange', 0.786)]


# ---------------------------------------------------------------------
# QUESTION 8
# ---------------------------------------------------------------------


    
def same_color_distribution():
    return (0.02, 'Fail to Reject')


# ---------------------------------------------------------------------
# QUESTION 9
# ---------------------------------------------------------------------


def perm_vs_hyp():
    return ['P', 'P', 'H', 'P', 'H']
