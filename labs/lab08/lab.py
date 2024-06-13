# lab.py


import pandas as pd
import numpy as np
import plotly.express as px
import statsmodels.api as sm
from pathlib import Path
from sklearn.preprocessing import Binarizer, QuantileTransformer, FunctionTransformer

import warnings
warnings.filterwarnings('ignore')


# ---------------------------------------------------------------------
# QUESTION 1
# ---------------------------------------------------------------------


def best_transformation():
    return 1


# ---------------------------------------------------------------------
# QUESTION 2
# ---------------------------------------------------------------------



def create_ordinal(df):
    data = df.copy()
    data = df[['cut', 'color', 'clarity']]
    cuts = ['Fair', 'Good', 'Very Good', 'Premium', 'Ideal']
    cut_dict = dict(zip(cuts, np.arange(5)))
    colors = ['J', 'I', 'H', 'G', 'F', 'E', 'D']
    color_dict = dict(zip(colors, np.arange(7)))
    clarities = ['I1', 'SI2', 'SI1', 'VS2', 'VS1', 'VVS2', 'VVS1', 'IF']
    clarity_dict = dict(zip(clarities, np.arange(8)))

    data['ordinal_cut'] = data['cut'].apply(lambda x: cut_dict[x])
    data['ordinal_color'] = data['color'].apply(lambda x: color_dict[x])
    data['ordinal_clarity'] = data['clarity'].apply(lambda x: clarity_dict[x])

    data = data.drop(columns = ['cut', 'color', 'clarity'])

    return data


# ---------------------------------------------------------------------
# QUESTION 3
# ---------------------------------------------------------------------



def one_hot_encode_column(df, col):
    unique_values = df[col].unique()
    
    new_columns = {}
    
    for val in unique_values:
        new_col_name = f'one_hot_{col}_{val}'
        new_columns[new_col_name] = (df[col] == val).astype(int)
    
    return pd.DataFrame(new_columns)

def create_one_hot(df):
    one_hot_df = pd.DataFrame()
    
    for col in ['cut', 'color', 'clarity']:
        one_hot_col_df = one_hot_encode_column(df, col)
        one_hot_df = pd.concat([one_hot_df, one_hot_col_df], axis=1)
    
    return one_hot_df

def create_proportions(df):
    proportion_df = pd.DataFrame()
    
    for col in ['cut', 'color', 'clarity']:
        value_counts = df[col].value_counts(normalize=True)
        proportion_col = df[col].map(value_counts)
        proportion_df[f'proportion_{col}'] = proportion_col
    
    return proportion_df
    
# ---------------------------------------------------------------------
# QUESTION 4
# ---------------------------------------------------------------------


import itertools

def create_quadratics(df):
    quantitative_columns = ['carat', 'depth', 'table', 'x', 'y', 'z']
    
    quadratics_df = pd.DataFrame()
    
    column_pairs = itertools.combinations(quantitative_columns, 2)
    
    for col1, col2 in column_pairs:
        new_col_name = f'{col1} * {col2}'
        quadratics_df[new_col_name] = df[col1] * df[col2]
    
    return quadratics_df


# ---------------------------------------------------------------------
# QUESTION 5
# ---------------------------------------------------------------------

def comparing_performance():
    return [0.8493305264354858, 1548.5331930613174, 'x', 'carat * x', 'ordinal_color', 1434.8400089047332]



# ---------------------------------------------------------------------
# QUESTION 6
# ---------------------------------------------------------------------


class TransformDiamonds(object):
    
    def __init__(self, diamonds):
        self.data = diamonds
        
    # Question 6.1
    def transform_carat(self, data):
        binarizer = Binarizer(threshold=1.0)
        return binarizer.fit_transform(data[['carat']])
    
    # Question 6.2
    def transform_to_quantile(self, data):
        quantile_transformer = QuantileTransformer(n_quantiles=100, output_distribution='uniform')
        quantile_transformer.fit(self.data[['carat']])
        return quantile_transformer.transform(data[['carat']])
    
    # Question 6.3
    def transform_to_depth_pct(self, data):
        def calculate_depth_percentage(X):
            X = np.asarray(X)
            x, y, z = X[:, 0], X[:, 1], X[:, 2]
            with np.errstate(divide='ignore', invalid='ignore'):
                depth_percentage = (2 * z) / (x + y) * 100
            return depth_percentage
        
        depth_transformer = FunctionTransformer(calculate_depth_percentage)
        return depth_transformer.transform(data[['x', 'y', 'z']]).flatten()