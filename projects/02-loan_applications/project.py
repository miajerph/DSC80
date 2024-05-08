# project.py


import pandas as pd
import numpy as np
from pathlib import Path

import plotly.express as px
import plotly.graph_objects as go
import plotly.io as pio
pd.options.plotting.backend = 'plotly'
from pandas.tseries.offsets import DateOffset

from IPython.display import display

# DSC 80 preferred styles
pio.templates["dsc80"] = go.layout.Template(
    layout=dict(
        margin=dict(l=30, r=30, t=30, b=30),
        autosize=True,
        width=600,
        height=400,
        xaxis=dict(showgrid=True),
        yaxis=dict(showgrid=True),
        title=dict(x=0.5, xanchor="center"),
    )
)
pio.templates.default = "simple_white+dsc80"
import warnings
warnings.filterwarnings("ignore")


# ---------------------------------------------------------------------
# QUESTION 1
# ---------------------------------------------------------------------

# helper for part 3

def clean_title(x):
    return x.lower().strip()

# helper for part 4
def offset(x):
    from pandas.tseries.offsets import DateOffset
    return DateOffset(months = x)
    
def clean_loans(loans):
    copy = loans.copy(deep=True)
    copy['issue_d'] = copy['issue_d'].apply(lambda x: pd.Timestamp(x))
    copy['term'] = copy['term'].apply(lambda x: x.strip().split(' ', 1)[0])
    copy['term'] = copy['term'].apply(lambda x: int(x))

    copy['emp_title'] = copy['emp_title'].apply(clean_title)
    copy.head()
    copy.loc[copy['emp_title'] == 'rn', 'emp_title'] = 'registered nurse'
    
    offsets = copy['term'].apply(offset)
    copy['term_end'] = copy['issue_d'] + offsets
    
    return copy
    
# ---------------------------------------------------------------------
# QUESTION 2
# ---------------------------------------------------------------------

# helper for indices
def new_idx(pairs):
    indices = []
    for i in range(len(pairs)):
        idx = 'r_' + pairs[i][0] + '_' + pairs[i][1]
        indices.append(idx)
    return indices

def correlations(df, pairs):
    pearsons = []
    for tpl in pairs:
        tpl_df = df[list(tpl)]
        tpl_df = tpl_df.corr(method='pearson').iloc[0][1]
        pearsons.append(tpl_df)
    pearsons = pd.Series(pearsons)
    pearsons.index = new_idx(pairs)
    return pearsons


# ---------------------------------------------------------------------
# QUESTION 3
# ---------------------------------------------------------------------


def create_boxplot(loans):
    df = loans.copy()
    # Define the credit score bins
    score_bins = [580, 670, 740, 800, 850]
    score_labels = ['[580, 670)', '[670, 740)', '[740, 800)', '[800, 850)']
    
    # Bin the credit scores
    df['credit_score_bin'] = pd.cut(df['fico_range_low'], bins=score_bins, labels=score_labels, right=False)
    
    # Convert the credit score bin to string
    df['credit_score_bin'] = df['credit_score_bin'].astype(str)

    #term_colors = {'36':'deeppink', '60':'deepskyblue'}
    
    # Create the boxplot
    fig = px.box(df, x='credit_score_bin', y='int_rate', color='term',
                 category_orders={'credit_score_bin': score_labels, 'term': ['36', '60']},
                 color_discrete_sequence=['deeppink', 'deepskyblue'], labels={'credit_score_bin': 'Credit Score Range', 'int_rate': 'Interest Rate (%)', 'term': 'Loan Length (Months)'},
                 title='Interest Rate vs. Credit Score')
    
    return fig


# ---------------------------------------------------------------------
# QUESTION 4
# ---------------------------------------------------------------------


def paradox_example(loans):
    return {
    'loans': loans,
    'keywords': ['lawyer', 'teacher'],
    'quantitative_column': 'dti',
    'categorical_column': 'addr_state'
    }
def tax_owed(income, brackets):
    taxes = []
    for i in range(len(brackets)-1):
        taxable = brackets[i][0] * (brackets[i+1][1]-brackets[i][1])
        taxes.append(taxable)
    j=0
    while (income >= brackets[j][1]) & (j<=len(brackets)-2):
        j += 1
    tax = np.sum(taxes[0:j-1])
    tax += brackets[j-1][0] * (income-brackets[j-1][1])
    return tax
alyssa — Today at 6:48 PM
def diff_means(df, N):
    stats = []
    copy = df.copy(deep=True)
    for i in range(N):
        copy['desc'] = np.random.permutation(copy['desc'])
        y_statement = copy.loc[copy['desc'].isna() ==  False]['int_rate'].mean()
        n_statement = copy.loc[copy['desc'].isna() == True]['int_rate'].mean()
        t_stat = y_statement - n_statement
        stats.append(t_stat)
    return stats

def ps_test(loans, N):
    copy = loans.copy(deep=True)
    y_statement = copy.loc[copy['desc'].isna() ==  False]['int_rate'].mean()
    n_statement = copy.loc[copy['desc'].isna() == True]['int_rate'].mean()
    observed_diff = y_statement - n_statement
    t_stats = diff_means(copy, N)
    p_val = (np.array(np.abs((t_stats))) >= np.abs(observed_diff)).mean()
    return p_val

def missingness_mechanism():
    return 2

def argument_for_nmar():
    '''
    Put your justification here in this multi-line string.
    Make sure to return your string!
    '''
    return 'The p-value is less the signifance level, so our permutation test suggests that there is a statistically significant difference between interest rates for borrowers with personal statements, versus borrowers without personal statements, so we can reject the null hypothesis and assume that the data is MAR and dependent on interest rate.'

# ---------------------------------------------------------------------
# QUESTION 5
# ---------------------------------------------------------------------


def tax_owed(income, brackets):
    taxes = []
    for i in range(len(brackets)-1):
        taxable = brackets[i][0] * (brackets[i+1][1]-brackets[i][1])
        taxes.append(taxable)
    j=0
    while (income >= brackets[j][1]) & (j<=len(brackets)-2):
        j += 1
    tax = np.sum(taxes[0:j-1])
    tax += brackets[j-1][0] * (income-brackets[j-1][1])
    return tax


# ---------------------------------------------------------------------
# QUESTION 6
# ---------------------------------------------------------------------


def clean_state_taxes(state_taxes_raw): 
    clean= state_taxes_raw.dropna(how='all')
    
    clean['State'] = clean['State'].where(clean['State'].str.match(r'^[A-Za-z\s.]+$'))
    clean['State'].fillna(method='ffill', inplace=True)
    
    clean['Rate'] = clean['Rate'].str.replace('%', '').replace(',', '')
    clean.loc[clean['Rate']=='none', 'Rate'] = 0
    clean['Rate'] = clean['Rate'].astype(float)
    clean['Rate'] = (clean['Rate'] / 100).round(2) 

    clean['Lower Limit'] = clean['Lower Limit'].str.replace('$', '').str.replace(',', '').fillna(0).astype(int)

    return clean


# ---------------------------------------------------------------------
# QUESTION 7
# ---------------------------------------------------------------------


def state_brackets(state_taxes):
    df = state_taxes.copy()
    df['bracket_list'] = tuple(zip(df['Rate'], df['Lower Limit']))
    series = df.groupby('State')['bracket_list'].apply(list)
    bracket_df = pd.DataFrame(series)
    return bracket_df
    
def combine_loans_and_state_taxes(loans, state_taxes):
    import json
    state_mapping_path = Path('data') / 'state_mapping.json'
    with open(state_mapping_path, 'r') as f:
        state_mapping = json.load(f)

    state_copy = state_taxes.copy()
    state_copy['State'] = [state_mapping[state] for state in state_copy['State']]
    brackets = state_brackets(state_copy)
    combined = loans.merge(brackets, how='left', left_on='addr_state', right_on='State')
    combined.rename(columns={'addr_state': 'State'}, inplace=True)
    return combined

# ---------------------------------------------------------------------
# QUESTION 8
# ---------------------------------------------------------------------


def find_disposable_income(loans_with_state_taxes):
    FEDERAL_BRACKETS = [
     (0.1, 0), 
     (0.12, 11000), 
     (0.22, 44725), 
     (0.24, 95375), 
     (0.32, 182100),
     (0.35, 231251),
     (0.37, 578125)
    ]
    df = loans_with_state_taxes.copy()
    df['federal_tax_owed'] = df['annual_inc'].apply(lambda x: tax_owed(x, FEDERAL_BRACKETS))
    df['state_tax_owed'] = df.apply(lambda x: tax_owed(x['annual_inc'], x['bracket_list']), axis=1)
    df['disposable_income'] = df['federal_tax_owed'] - df['state_tax_owed']
    return df


# ---------------------------------------------------------------------
# QUESTION 9
# ---------------------------------------------------------------------


def aggregate_and_combine(loans, keywords, quantitative_column, categorical_column):
    def col_names(keywords, quantitative_column):
        names = []
        for key in keywords:
            name = f'{key}_mean_{quantitative_column}'
            names.append(name)
        return names
    result = pd.DataFrame(index=np.unique(loans[categorical_column]), columns=col_names(keywords, quantitative_column))
    result.index.name = categorical_column

    unique_cat = np.unique(loans[categorical_column])
    for i in range(len(np.unique(loans[categorical_column]))):
        for j in range(len(keywords)):
            # assign row by row values
            # first is the value in the first column, 
            # second is the value in the second column
            first = loans.loc[(loans['emp_title'].apply(lambda x: keywords[0] in x)) & (loans[categorical_column] == unique_cat[i])][quantitative_column].mean()
            second = loans.loc[(loans['emp_title'].apply(lambda x: keywords[1] in x)) & (loans[categorical_column] == unique_cat[i])][quantitative_column].mean()
            result.iloc[i] = [first, second]
    result = result.dropna(axis = 0, how = 'all')
    mean_1 = loans.loc[loans['emp_title'].apply(lambda x: keywords[0] in x)]['loan_amnt'].mean()
    mean_2 = loans.loc[loans['emp_title'].apply(lambda x: keywords[1] in x)]['loan_amnt'].mean()
    result.loc['Overall'] = [mean_1, mean_2]
    return result


# ---------------------------------------------------------------------
# QUESTION 10
# ---------------------------------------------------------------------


def exists_paradox(loans, keywords, quantitative_column, categorical_column):
    df = aggregate_and_combine(loans, keywords, quantitative_column, categorical_column)
    simp1 = df.loc[df.iloc[:, 0] > df.iloc[:, 1]]
    simp1 = len(simp1)
    simp2 = (df.iloc[-1, 0] < df.iloc[-1, 1])
    if (simp1==len(df)-1) & (simp2==True):
         return True
    elif simp1==0 & (not simp2):
        return True
    else:
        return False
    
    
def paradox_example(loans):
    return {
    'loans': loans,
    'keywords': ['lawyer', 'teacher'],
    'quantitative_column': 'dti',
    'categorical_column': 'addr_state'
    }
