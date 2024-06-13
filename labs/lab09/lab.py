# lab.py


import pandas as pd
import numpy as np
from pathlib import Path
import plotly.express as px

from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import FunctionTransformer
from sklearn.preprocessing import OneHotEncoder
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer


# ---------------------------------------------------------------------
# QUESTION 1
# ---------------------------------------------------------------------


def simple_pipeline(data):
    X = data[['c2']]
    y = data['y']
    
    log_transformer = FunctionTransformer(np.log, validate=True)
    
    pipeline = Pipeline(steps=[
        ('log_transformer', log_transformer),
        ('regressor', LinearRegression())
    ])
    
    pipeline.fit(X, y)
    
    predictions = pipeline.predict(X)
    
    return (pipeline, predictions)


# ---------------------------------------------------------------------
# QUESTION 2
# ---------------------------------------------------------------------


from sklearn.preprocessing import FunctionTransformer, OneHotEncoder
from sklearn.linear_model import LinearRegression

def multi_type_pipeline(data):
    X = data[['group', 'c1', 'c2']]
    y = data['y']
    
    log_transformer = FunctionTransformer(np.log, validate=True)
    
    preprocessor = ColumnTransformer(
        transformers=[
            ('num', 'passthrough', ['c1']),           # Use 'c1' as is
            ('log', log_transformer, ['c2']),         # Log-scale 'c2'
            ('onehot', OneHotEncoder(), ['group'])    # One-hot encode 'group'
        ]
    )
    
    pipeline = Pipeline(steps=[
        ('preprocessor', preprocessor),
        ('regressor', LinearRegression())
    ])
    
    pipeline.fit(X, y)
    
    predictions = pipeline.predict(X)
    
    return (pipeline, predictions)


# ---------------------------------------------------------------------
# QUESTION 3
# ---------------------------------------------------------------------


# Imports
from sklearn.base import BaseEstimator, TransformerMixin

class StdScalerByGroup(BaseEstimator, TransformerMixin):

    def __init__(self):
        pass

    def fit(self, X, y=None):
        df = pd.DataFrame(X)
        group_col = df.columns[0]
        
        self.grps_ = df.groupby(group_col).agg(['mean', 'std']).to_dict()
        
        return self

    def transform(self, X, y=None):

        try:
            getattr(self, "grps_")
        except AttributeError:
            raise RuntimeError("You must fit the transformer before transforming the data!")
        
        def standardize(row):
            group = row[0]
            return [(row[i] - self.grps_[col, 'mean'][group]) / self.grps_[col, 'std'][group] for i, col in enumerate(df.columns[1:], start=1)]
        
        df = pd.DataFrame(X)
        group_col = df.columns[0]
        
        standardized_data = df.apply(lambda row: standardize(row), axis=1, result_type='expand')
        standardized_data.columns = df.columns[1:]
        
        return standardized_data
        
# ---------------------------------------------------------------------
# QUESTION 4
# ---------------------------------------------------------------------

def eval_toy_model():
    return [(2.7551086974518104, 0.39558507345910776), (2.3148336164355268, 0.573324931567333), (2.315733947782385, 0.5729929650348397)]


# ---------------------------------------------------------------------
# QUESTION 5
# ---------------------------------------------------------------------

from sklearn.metrics import mean_squared_error, r2_score

# rmse helper
def rmse(y_true, y_pred):
    return np.sqrt(mean_squared_error(y_true, y_pred))

def tree_reg_perf(galton):
    # Add your imports here
    from sklearn.tree import DecisionTreeRegressor
    from sklearn.model_selection import train_test_split
    
    X = galton.drop(columns=['childHeight'])
    y = galton['childHeight']
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=42)
    
    # make lists toi store error vals
    train_err = []
    test_err = []

    
    for depth in range(1, 21):
        # fit
        tree_reg = DecisionTreeRegressor(max_depth=depth, random_state=42)
        tree_reg.fit(X_train, y_train)
        
        # predict
        y_train_pred = tree_reg.predict(X_train)
        y_test_pred = tree_reg.predict(X_test)
        
        # find rmse for training data and test data
        train_rmse = rmse(y_train, y_train_pred)
        test_rmse = rmse(y_test, y_test_pred)
    
        # append error vals
        train_err.append(train_rmse)
        test_err.append(test_rmse)
    
    results = pd.DataFrame({'train_err': train_err, 'test_err': test_err}, index=range(1, 21))
    return results
    
    
def knn_reg_perf(galton):
    # Add your imports here
    from sklearn.neighbors import KNeighborsRegressor
    from sklearn.model_selection import train_test_split

    # set x and y
    X = galton.drop(columns=['childHeight'])
    y = galton['childHeight']
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=42)
    
    # make lists for error vals
    train_err = []
    test_err = []
    
    
    for k in range(1, 21):
        # fit X and y
        knn_reg = KNeighborsRegressor(n_neighbors=k)
        knn_reg.fit(X_train, y_train)
        
        # train w regressor
        y_train_pred = knn_reg.predict(X_train)
        y_test_pred = knn_reg.predict(X_test)
        
        # find rmse
        train_rmse = rmse(y_train, y_train_pred)
        test_rmse = rmse(y_test, y_test_pred)
        
        # append errors vals to lists
        train_err.append(train_rmse)
        test_err.append(test_rmse)
    
    results = pd.DataFrame({'train_err': train_err, 'test_err': test_err}, index=range(1, 21))
    return results

# ---------------------------------------------------------------------
# QUESTION 6
# ---------------------------------------------------------------------

from sklearn.base import BaseEstimator, TransformerMixin

class TitleExtractor(BaseEstimator, TransformerMixin):
    def fit(self, X, y=None):
        return self
    
    def transform(self, X, y=None):
        X = X.copy()
        X['Title'] = X['Name'].apply(lambda name: re.search(r'\b([A-Za-z]+)\.', name).group(1) if re.search(r'\b([A-Za-z]+)\.', name) else "")
        return X

def standardize_age(df):
    df['Age'] = df.groupby('Pclass')['Age'].transform(lambda x: (x - x.mean()) / x.std())
    return df

def titanic_model(titanic):
    # Add your import(s) here
    import re 
    from sklearn.preprocessing import StandardScaler, OneHotEncoder
    from sklearn.compose import ColumnTransformer
    from sklearn.ensemble import RandomForestClassifier


    df = titanic.copy(deep=True)
    df['Title'] = df['Name'].apply(extract_title)
    df = standardize_age(df)


    # set features and instantiate pipeline
    # separated by numeric and categorical features
    numeric_features = ['Age', 'Fare', 'Siblings/Spouses Aboard', 'Parents/Children Aboard']
    numeric_transformer = Pipeline(steps=[
        ('scaler', StandardScaler())
    ])
    
    categorical_features = ['Sex', 'Pclass', 'Title']
    categorical_transformer = Pipeline(steps=[
        ('onehot', OneHotEncoder(handle_unknown='ignore'))
    ])
        
    preprocessor = ColumnTransformer(
        transformers=[
            ('num', numeric_transformer, numeric_features),
            ('cat', categorical_transformer, categorical_features)
        ])
    
    pipeline = Pipeline(steps=[
        ('title_extractor', TitleExtractor()),  # Custom transformer to add 'Title' column
        ('preprocessor', preprocessor),
        ('classifier', RandomForestClassifier())
    ])
    
    
    X = df.drop(columns=['Survived'], axis=1)
    y = df['Survived']
    
    pipeline.fit(X, y)
    return pipeline

