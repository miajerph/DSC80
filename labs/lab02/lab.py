# lab.py


import os
import io
from pathlib import Path
import pandas as pd
import numpy as np


# ---------------------------------------------------------------------
# QUESTION 1
# ---------------------------------------------------------------------


def trick_me():
    return 1


def trick_bool():
    return [2, 2, 13]


# ---------------------------------------------------------------------
# QUESTION 2
# ---------------------------------------------------------------------


def population_stats(df):
    num_nonnull = df.notna().sum()
    prop_nonnull = num_nonnull / len(df)
    num_distinct = df.nunique(dropna=True)
    prop_distinct = num_distinct / num_nonnull
    
    stats = pd.DataFrame({
        'num_nonnull': num_nonnull,
        'prop_nonnull': prop_nonnull,
        'num_distinct': num_distinct,
        'prop_distinct': prop_distinct
    })
    
    return stats


# ---------------------------------------------------------------------
# QUESTION 3
# ---------------------------------------------------------------------


def most_common(df, N=10):
    result_df = pd.DataFrame(index=range(N))

    for col in df.columns:
        top_N = df[col].value_counts().head(N)
        result_df[col+'_values'] = top_N.index.tolist() + [np.NaN] * (N-len(top_N))
        result_df[col+'_counts'] = top_N.tolist() + [np.NaN] * (N-len(top_N))

    return result_df


# ---------------------------------------------------------------------
# QUESTION 4
# ---------------------------------------------------------------------


def super_hero_powers(powers):
    powers_count = powers.drop('hero_names', axis=1).sum(axis=1)
    max_powers_hero = powers.loc[powers_count.idxmax(), 'hero_names']
    
    fliers = powers[powers['Flight'] == True]
    fliers_count = fliers.drop('hero_names', axis=1).sum()
    fliers_count = fliers_count.drop('Flight')  
    second_most_common = fliers_count.idxmax()
    
    single_power_heroes = powers[powers_count == 1]
    single_power_heroes_count = single_power_heroes.drop('hero_names',axis=1).sum()
    most_common_single_power = single_power_heroes_count.idxmax()
    
    return [max_powers_hero, second_most_common, most_common_single_power]



# ---------------------------------------------------------------------
# QUESTION 5
# ---------------------------------------------------------------------


def clean_heroes(heroes):
    return heroes.replace(['-', -99.0], np.NaN)


# ---------------------------------------------------------------------
# QUESTION 6
# ---------------------------------------------------------------------


def super_hero_stats():
    return [
        "Onslaught",
        "George Lucas",
        "bad",
        "Marvel Comics",
        "NBC - Heroes",
        "Groot"
    ]


# ---------------------------------------------------------------------
# QUESTION 7
# ---------------------------------------------------------------------


def clean_universities(df):
    cleaned_df = df.copy()
    cleaned_df['institution'] = cleaned_df['institution'].str.replace('\n', ', ')
    cleaned_df['broad_impact'] = cleaned_df['broad_impact'].astype(int)
    cleaned_df[['nation', 'national_rank_cleaned']] = cleaned_df['national_rank'].str.split(', ', expand=True)
    cleaned_df = cleaned_df.drop('national_rank', axis=1)
    cleaned_df['nation'] = cleaned_df['nation'].replace({'Czechia': 'Czech Republic', 'USA': 'United States', 'UK': 'United Kingdom'})
    cleaned_df['national_rank_cleaned'] = cleaned_df['national_rank_cleaned'].astype(int)
    cleaned_df['is_r1_public'] = ~(cleaned_df['control'].isnull() | cleaned_df['city'].isnull() | cleaned_df['state'].isnull())
    return cleaned_df

def university_info(cleaned):
    results = []
    
    state_means = cleaned.groupby('state').filter(lambda x: len(x) >= 3).groupby('state')['score'].mean()
    lowest_mean_state = state_means.idxmin()
    results.append(lowest_mean_state)

    results.append(len(cleaned[(cleaned['quality_of_faculty'] <= 100) & (cleaned['world_rank'])])/100)
    
    state_private_proportions = cleaned.groupby('state')['is_r1_public'].mean()
    state_50 = (state_private_proportions < 0.5).sum()
    results.append(state_50)

    nations_top = cleaned[(cleaned['national_rank_cleaned'] == 1)]
    nations_top = nations_top.sort_values(by='world_rank', ascending=False)
    lowest_institution = nations_top['institution'].iloc[0]
    results.append(lowest_institution)

    return results

