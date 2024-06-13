# lab.py


import os
import io
from pathlib import Path
import pandas as pd
import numpy as np


# ---------------------------------------------------------------------
# QUESTION 1
# ---------------------------------------------------------------------


def read_linkedin_survey(dirname):
    
    target_cols  = ['first name', 'last name', 'current company', 'job title', 'email', 'university']    
    surveys = []

    # edge case for if the path doesn't exist within the directory
    if not Path(dirname).is_dir():
        print(True)
        return
    
    # search through the directory and add files to list
    files = []
    for file_path in dirname.iterdir():
        files.append(file_path)
    for file_path in files:
        survey = pd.read_csv(file_path)
        survey.columns = survey.columns.str.lower().str.replace('_', ' ').str.strip()
        survey = survey[target_cols]
        surveys.append(survey)
        
    concat_surveys = pd.concat([file for file in surveys], axis=0)
    
    concat_surveys.reset_index(drop=True, inplace=True)
    
    return concat_surveys


def com_stats(df):

    df = df.fillna('')
    
    # ohio programmers
    ohio_programmers = df[(df['university'].str.contains('Ohio', case=False, na=False)) & 
                                 (df['job title'].str.contains('Programmer', case=False, na=False))].shape[0]
    ohio = df[df['university'].str.contains('Ohio', case=False, na=False)].shape[0]
    ohio_programmers = ohio_programmers / ohio if ohio_programmers > 0 else 0

    # engineers
    engineer_titles = df[df['job title'].str.endswith('Engineer', na=False)].shape[0]

    # longest job name
    longest_title = df.loc[df['job title'].str.len().idxmax(), 'job title']

    # number of jobs with manager in the name
    managers = df[df['job title'].str.contains('manager', case=False, na=False)].shape[0]

    return [ohio_programmers, engineer_titles, longest_title, managers]


# ---------------------------------------------------------------------
# QUESTION 2
# ---------------------------------------------------------------------

from functools import reduce

def read_student_surveys(dirname):
    if not Path(dirname).is_dir():
        print(True)
        return
    files = []
    favs = []
    for file in dirname.iterdir():
        files.append(file)
#     files = files[1:]
    for fav in files:
        read = pd.read_csv(fav)
        favs.append(read)

    combined = reduce(lambda left, right: pd.merge(left,right,on=['id'], how='outer'), favs)
    combined.index = combined['id']
    combined = combined.drop(columns=['id'])
    return combined


def check_credit(df):
    names = pd.read_csv('data/extra-credit-surveys/favorite1.csv')

    df = df.replace('(no genres listed)', np.NaN)

    # ec without participation
#     df_clean = df.drop(columns=['id'])
    ec = (df.isna()).sum(axis=1)
    ser = pd.Series([5 if x >= 3 else 0 for x in ec])
    
    # find ec for participation 
    class_ec = df.drop(columns=['name']).count() / len(df)
    num_points = (class_ec >= 0.9).sum()
    if num_points > 2:
        num_points = 2

    # combine the ec and class_ec into new df
    result = pd.DataFrame({'name': names['name'], 'ec': (ser + num_points)})
    max_val = 7
    result.loc[result['ec'] > max_val, 'ec'] = max_val
    result.index = df.index
    
    return result
# ---------------------------------------------------------------------
# QUESTION 3
# ---------------------------------------------------------------------


def most_popular_procedure(pets, procedure_history):
    pet_pro = pets.merge(procedure_history, left_on='PetID', right_on='PetID')
    popular = pet_pro['ProcedureType'].value_counts().idxmax()
    return popular 

def pet_aggregation(x):
    if x.empty:
        return ''  
    elif len(x) == 1:
        return x.iloc[0]  
    else:
        return x.tolist()
    
def pet_name_by_owner(owners, pets):
    pets.rename(columns={'Name': 'PetName'}, inplace=True)
    owner_df = owners.merge(pets, left_on='OwnerID', right_on='OwnerID')
    owner_df = owner_df.sort_values('OwnerID') # idk if this will make it wrong
    series = owner_df.groupby('OwnerID')['PetName'].agg(pet_aggregation)
    
    # use the series to get corresponding names based on the IDs
    # im not really sure how to change the index to a different column entirely 
    owner_series = owner_df['OwnerID'].unique()
    name = owner_df.drop_duplicates(subset='OwnerID').loc[owner_df['OwnerID'].isin(owner_series), 'Name'].values

    series.index = name
    return series


def total_cost_per_city(owners, pets, procedure_history, procedure_detail):
    city_owners = owners[['OwnerID', 'City']]
    city_pets = city_owners.merge(pets, left_on='OwnerID', right_on='OwnerID')
    city_history = city_pets.merge(procedure_history, left_on='PetID', right_on='PetID')
    city_detail = pd.merge(city_history, procedure_detail, on=['ProcedureType', 'ProcedureSubCode'], how='inner')
    return city_detail.groupby('City')['Price'].sum()

# ---------------------------------------------------------------------
# QUESTION 4
# ---------------------------------------------------------------------


def average_seller(sales):
    avg_seller = sales.pivot_table(index='Name', values='Total', aggfunc='mean')
    avg_seller.rename(columns={'Total': 'Average Sales'}, inplace=True)
    return avg_seller

def product_name(sales):
    pivot_sales = sales.pivot_table('Total', index='Name', columns='Product')
    return pivot_sales

def count_product(sales):
    count_items = pd.pivot_table(sales, values='Total', index=['Product', 'Name'], columns='Date', aggfunc='count', fill_value=0)
    return count_items

def total_by_month(sales):
    sales['Date'] = pd.to_datetime(sales['Date'])
    sales['Month'] = sales['Date'].dt.strftime('%B')
    pivot_month = sales.pivot_table(index=['Name', 'Product'], columns='Month', values='Total', aggfunc='sum', fill_value=0)
    return pivot_month
