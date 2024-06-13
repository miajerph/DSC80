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
    surveys = []
    target_cols = ['first name', 'last name', 'current company', 'job title', 'email', 'university']

    if not Path(dirname).is_dir():
        print(True)
        return
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
    ohio_programmers = df[(df['university'].str.contains('Ohio', case=False, na=False)) & 
                                 (df['job title'].str.contains('Programmer', case=False, na=False))].shape[0]
    ohio = df[df['university'].str.contains('Ohio', case=False, na=False)].shape[0]
    ohio_programmers = ohio_programmers / ohio if ohio_programmers > 0 else 0

    engineer_titles = df[df['job title'].str.endswith('Engineer', na=False)].shape[0]

    longest_title = df.loc[df['job title'].str.len().idxmax(), 'job title']

    managers = df[df['job title'].str.contains('manager', case=False, na=False)].shape[0]

    return [ohio_programmers, engineer_titles, longest_title, managers]




# ---------------------------------------------------------------------
# QUESTION 2
# ---------------------------------------------------------------------


def read_student_surveys(dirname):
    from functools import reduce
    if not Path(dirname).is_dir():
        print(True)
        return
    files = []
    for file_path in dirname.iterdir():
        files.append(file_path)
    favorites = []
    for file in files:
        fav = pd.read_csv(file)
        favorites.append(fav)
    combin = reduce(lambda left, right: pd.merge(left, right, on=['id'], how='outer'), favorites)
    combin.set_index('id', drop=True, inplace=True)
    return combin


def check_credit(df):
    df_copy = df.copy()
    df_copy = df.replace('(no genres listed)', np.NaN)
    names = df_copy['name']
    df_copy = df.drop(columns=['name'])
    ec = (df_copy.isna()==False)
    ec = ec.sum(axis=1)
    class_ec = df_copy.count() / len(df_copy)
    num_points = (class_ec >= 0.9).sum()
    result = pd.DataFrame({'name': names, 'ec': (ec+num_points)})
    max_val = 7
    result.loc[result['ec'] > max_val, 'ec']=max_val
    return result


# ---------------------------------------------------------------------
# QUESTION 3
# ---------------------------------------------------------------------


def most_popular_procedure(pets, procedure_history):
    merged_df = pd.merge(pets, procedure_history, on= 'PetID', how='left')
    procedure_counts = merged_df['ProcedureType'].value_counts()
    most_popular_procedure_type = procedure_counts.idxmax()
    return most_popular_procedure_type

def pet_name_by_owner(owners, pets):
    pets_copy = pets.copy()
    pets_copy.rename(columns={'Name':'PetName'}, inplace=True)
    merged_df = owners.merge(pets_copy, left_on='OwnerID', right_on='OwnerID')

    def pet_agg(x):
        if x.empty:
            return ''
        elif len(x)==1:
            return x.iloc[0]
        else:
            return x.tolist()

    pet_names_by_owner = merged_df.groupby('OwnerID')['PetName'].agg(pet_agg)
    
    unique_owners = owners['OwnerID'].unique()
    name = owners.drop_duplicates(subset='OwnerID').loc[owners['OwnerID'].isin(unique_owners), 'Name'].values

    pet_names_by_owner.index = name
    return pet_names_by_owner


def total_cost_per_city(owners, pets, procedure_history, procedure_detail):
    # Merge owners, pets, and procedure_history DataFrames to get pet procedures and owner cities
    merged_df = pd.merge(pd.merge(owners, pets, on='OwnerID', how='left'), procedure_history, on='PetID', how='left')
    
    # Merge with procedure_detail DataFrame to get procedure costs
    merged_df = pd.merge(merged_df, procedure_detail, on='ProcedureType', how='left')
    
    # Calculate total cost per procedure
    #merged_df['TotalCost'] = merged_df['Price'] * merged_df['Quantity']
    
    # Group by city and sum total costs
    total_cost_per_city = merged_df.groupby('City')['Price'].sum()
    
    return total_cost_per_city


# ---------------------------------------------------------------------
# QUESTION 4
# ---------------------------------------------------------------------


def average_seller(sales):
    average_sales = sales.groupby('Name')['Total'].mean()    
    average_sales = average_sales.to_frame(name='Average Sales')
    return average_sales

def product_name(sales):
    product_sales = pd.pivot_table(sales, values='Total', index='Name', columns='Product', aggfunc='sum')
    return product_sales

def count_product(sales):
    count_items = pd.pivot_table(sales, values='Total', index=['Product', 'Name'], columns='Date', aggfunc='count', fill_value=0)
    return count_items

def total_by_month(sales):
    sales['Month'] = pd.to_datetime(sales['Date']).dt.month_name()
    total_monthly_sales = pd.pivot_table(sales, values='Total', index=['Name', 'Product'], columns='Month', aggfunc='sum', fill_value=0)
    return total_monthly_sales
