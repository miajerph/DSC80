# lab.py


from pathlib import Path
import pandas as pd
import numpy as np
from scipy import stats


# ---------------------------------------------------------------------
# QUESTION 1
# ---------------------------------------------------------------------


def after_purchase():
    return ['NMAR', 'MD', 'MAR', 'MAR', 'MAR']


# ---------------------------------------------------------------------
# QUESTION 2
# ---------------------------------------------------------------------


def multiple_choice():
    return ['MAR', 'MAR', 'MD', 'NMAR', 'MCAR']


# ---------------------------------------------------------------------
# QUESTION 3
# ---------------------------------------------------------------------



def first_round():
    return [0.143, 'NR']


def second_round():
    return [0.022, 'R', 'D']


# ---------------------------------------------------------------------
# QUESTION 4
# ---------------------------------------------------------------------

from scipy.stats import ks_2samp

def verify_child(heights):
    p_vals = pd.Series(index=heights.columns[2:], dtype=float)
    
    for child_column in heights.columns[2:]:
        child_missing = heights[heights[child_column].isna()]['father']
        child_not_missing = heights[~heights[child_column].isna()]['father']
        
        ks_stat, p_val = ks_2samp(child_missing, child_not_missing)
        p_vals[child_column] = p_val

    return p_vals


# ---------------------------------------------------------------------
# QUESTION 5
# ---------------------------------------------------------------------


def cond_single_imputation(new_heights):
    quartiles = pd.qcut(new_heights['father'], q=4)
    mean_child_by_quartile = new_heights.groupby(quartiles)['child'].mean()
    imputed_child = new_heights['child'].fillna(new_heights.groupby(quartiles)['child'].transform('mean'))
    return imputed_child


# ---------------------------------------------------------------------
# QUESTION 6
# ---------------------------------------------------------------------


def quantitative_distribution(child, N):
    non_missing_child = child.dropna()
    freq, bin_edges = np.histogram(non_missing_child, density=True)
    choice_distr = []
    for i in range(len(freq)):
        prob = (bin_edges[i+1]-bin_edges[i]) * freq[i]
        choice_distr.append(prob)
    imp_vals = []
    for i in range(N):
        bin_index = np.random.choice(range(10), p=choice_distr)
        imp_val = np.random.uniform(bin_edges[bin_index], bin_edges[bin_index+1])
        imp_vals.append(imp_val)
    imp_vals = np.array(imp_vals)
    return imp_vals


def impute_height_quant(child):
    num_missing = child.isna().sum()
    full_child = child.copy()
    full_child[full_child.isna()] = quantitative_distribution(child, num_missing)
    return full_child


# ---------------------------------------------------------------------
# QUESTION 7
# ---------------------------------------------------------------------


def answers():
    return [1, 2, 2, 1], ['soundcloud.com/robots.txt', 'instagram.com/robots.txt']
