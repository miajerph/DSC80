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
    cols = ['Name', 'Name', 'Age']
    data = [['Abby', 'Frondoza', 19], ['Jessica', 'Chen', 18], ['Jennifer', 'Pei', 18], ['Katie', 'Truong', 18], ['Kyana', 'Early', 20]]
   
    tricky_1 = pd.DataFrame(data, columns=cols)
    tricky_1.to_csv('tricky_1.csv', index=False)
    tricky_2 = pd.read_csv('tricky_1.csv')
    return 3

def trick_bool():
    return [4, 10, 13]
    


# ---------------------------------------------------------------------
# QUESTION 2
# ---------------------------------------------------------------------


def population_stats(df):
    measure = ['num_nonnull', 'prop_nonnull', 'num_distinct', 'prop_distinct']
    stats = pd.DataFrame(index = df.columns, columns = measure)
    
    nonnull = df.notnull().sum()
    stats['num_nonnull'] = nonnull
    
    prop_nonnull = df.notnull().sum() / df.shape[0]
    stats['prop_nonnull'] = prop_nonnull
    
    num_distinct = df.apply(lambda x: x.nunique(dropna=True))
    stats['num_distinct'] = num_distinct
    
    prop_distinct = num_distinct / df.shape[0]
    stats['prop_distinct'] = prop_distinct
    
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

#     # Find the N most common values across all columns
#     series = df.unstack()
#     y = series.value_counts().head(N).index

#     # Create new column names for the result DataFrame
#     df_new_cols = []
#     for col in df.columns:  
#         df_new_cols.append(col + '_values')
#         df_new_cols.append(col + '_counts')

#     # Create a DataFrame to store the results
#     result_df = pd.DataFrame(index=range(N), columns=df_new_cols)

#     # Filter the original DataFrame to include only rows containing common values
#     filtered = df[df.apply(lambda row: any(val in row.values for val in y), axis=1)]

#     # Iterate over each column and calculate counts for common values
#     for col in df.columns:
#         counts = filtered[col].value_counts()
#         if len(counts) < N:
#             counts = counts.reindex(y, fill_value=np.nan)
#         else:
#             counts = counts.head(N)
#         result_df[col + '_values'] = y
#         result_df[col + '_counts'] = counts.values

#     return result_df

   
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

    # return a list with three strings
    # string 1: name of hero with the most powers
    # string 2: second most common superpower among heros who can fly --> 'Flight'
    # string 3: most common power among heros with only one power
    
    
#     # initialize array 
#     stat_list = []

#     # find hero with the most powers
#     most_powers = powers.iloc[powers.sum(numeric_only=True, axis=1).sort_values().idxmax()]['hero_names']
# #     stat_list.append(most_powers)
    
#     # second most common given they can fly
#     second_after_flight = powers.query('Flight == True').sum(numeric_only=True, axis=0).sort_values(ascending=False).index[1]
    
#     # most common power among heros w/ one power
#     common_one_power = powers[powers.sum(numeric_only=True, axis=1) == 1].sum(numeric_only=True, axis=0).sort_values(ascending=False).idxmax()
    
    
#     stat_list.extend((most_powers, second_after_flight, common_one_power))
#     return stat_list


# # ---------------------------------------------------------------------
# # QUESTION 5
# # ---------------------------------------------------------------------


def clean_heroes(heroes):
    heroes = heroes.replace('-', np.NaN)
    heroes = heroes.apply(lambda x: x.mask(x < 0) if np.issubdtype(x.dtype, np.number) else x)
    return heroes


# # ---------------------------------------------------------------------
# # QUESTION 6
# # ---------------------------------------------------------------------


def super_hero_stats():
    # 0: name of tallest mutant with 'No Hair'
    # 1: publishers w/ more than 5 characters, which publisher has the largest
    # prop of human characters --> 'Race' == 'Human'
    # 2: are good or bad characters taller on average
    # 3: publisher with greater prop of bad characters --> 'Marvel Comics' or 'DC Comics'
    # 4: publisher besides marvel and dc w/ the most characters, only drop row if 
    # publisher is null
    # 5: name of character who is more than 1 SD above mean in height and more than 1
    # SD below mean in weight
    
    
    return ['Onslaught', 'George Lucas', 'bad', 'Marvel Comics', 'NBC - Heroes', 'Groot']


# # ---------------------------------------------------------------------
# # QUESTION 7
# # ---------------------------------------------------------------------


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
    
    info = []
    top_100 = 100
    min_institutions = 3
    
    # lowest mean score of states with at least 3 universities 
    states = cleaned[cleaned['state'].isna() == False]
    state_counts = states.groupby('state').size()
    d = state_counts[state_counts > min_institutions]
    p = cleaned[cleaned['state'].isin(d.index)]
    state_low_mean = p.groupby('state')['score'].mean().sort_values(ascending=True).idxmin()
    info.append(state_low_mean)
    
    
    # proportion of institutions in top 100 that are also in top 100 for quality of faculty
    prop_quality = cleaned[(cleaned['world_rank'] <= 100) & (cleaned['quality_of_faculty'] <= 100)].shape[0] / top_100
    info.append(prop_quality)
    
    # number of states where at least half of their schools are private
    # is_r1_public will be False
    states = cleaned[cleaned['state'].notna()]
    state_counts = states.groupby('state').size()
    half_private_states = int((states.groupby('state')['is_r1_public'].mean() < 0.5).sum())
    info.append(half_private_states)

    # lowest ranking institution based on world rank but they have to be
    # rank 1 in their country
    lowest_1st = cleaned.iloc[cleaned[cleaned['national_rank_cleaned'] == 1]['world_rank'].idxmax()]['institution'] 
    info.append(lowest_1st)
    
    # info should be str float int str
    return info

    
def swap_sum(A, B):
    # they dont have to be the same length
    # stop > start
#     A2 = A
#     B2 = B
    
    a_len = len(A)
    b_len = len(B)
    if a_len > b_len:
        short = B
        long = A
    else:
        short = A
        long = B
        
    for i in range(len(long)):
        for j in range(len(short)):
            A2 = A[:]  # Make a copy of A
            B2 = B[:]  # Make a copy of B
            A2[i], B2[j] = B2[j], A2[i]  # Swap elements
            if sum(A2) + 10 == sum(B2):
                return i, j
    
    return None
