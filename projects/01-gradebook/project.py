# project.py


import pandas as pd
import numpy as np
from pathlib import Path

import plotly.express as px


# ---------------------------------------------------------------------
# QUESTION 1
# ---------------------------------------------------------------------


def get_assignment_names(grades):
    assignment_names = {
        'lab': [],
        'project': [],
        'midterm': [],
        'final': [],
        'disc': [],
        'checkpoint': []
    }
    
    for col in grades.columns.to_list():
        if ('lab' in col.lower()) & (len(col)==5):
            assignment_names['lab'].append(col)
        elif ('project' in col.lower()) & (len(col)==9):
            assignment_names['project'].append(col)
        elif ('midterm' in col.lower()) & (len(col)==6):
            assignment_names['midterm'].append(col)
        elif ('final' in col.lower()) & (len(col)==5):
            assignment_names['final'].append(col)
        elif ('disc' in col.lower()) & (len(col)==12):
            assignment_names['disc'].append(col)
        elif ('checkpoint' in col.lower()) & (len(col)==22):
            assignment_names['checkpoint'].append(col)
    
    return assignment_names


# ---------------------------------------------------------------------
# QUESTION 2
# ---------------------------------------------------------------------


def projects_total(grades):
    grades.fillna(0, inplace=True)
    project_avg = 0.20 * (\
    ((grades['project01'] + grades['project01_free_response']) / (grades['project01 - Max Points'] + grades['project01_free_response - Max Points'])) + \
    ((grades['project02'] + grades['project02_free_response']) / (grades['project02 - Max Points'] + grades['project02_free_response - Max Points'])) + \
    (grades['project03'] / grades['project03 - Max Points']) + \
    (grades['project04'] / grades['project04 - Max Points']) + \
    ((grades['project05'] + grades['project05_free_response']) / (grades['project05 - Max Points'] + grades['project05_free_response - Max Points'])))
    return project_avg


# ---------------------------------------------------------------------
# QUESTION 3
# ---------------------------------------------------------------------


def lateness_penalty(col):
    import datetime
    def to_multiplier(datestring):
        hours, minutes, seconds = map(int, datestring.split(':'))
        duration = datetime.timedelta(hours=hours, minutes=minutes, seconds=seconds)
        if duration <= datetime.timedelta(days=0, hours=2, minutes=0):
            return 1.0
        elif duration <= datetime.timedelta(days=7, hours=0, minutes=0):
            return 0.9
        elif duration <= datetime.timedelta(days=14, hours=0, minutes=0):
            return 0.7
        else: 
            return 0.4
        
    return col.apply(to_multiplier)


# ---------------------------------------------------------------------
# QUESTION 4
# ---------------------------------------------------------------------


def process_labs(grades):
    grades.fillna(0, inplace=True)
    df = pd.DataFrame(columns=(get_assignment_names(grades)['lab']))
    df['lab01'] = (grades['lab01'] / grades['lab01 - Max Points']) * lateness_penalty(grades['lab01 - Lateness (H:M:S)'])
    df['lab02'] = (grades['lab02'] / grades['lab02 - Max Points']) * lateness_penalty(grades['lab02 - Lateness (H:M:S)'])
    df['lab03'] = (grades['lab03'] / grades['lab03 - Max Points']) * lateness_penalty(grades['lab03 - Lateness (H:M:S)'])
    df['lab04'] = (grades['lab04'] / grades['lab04 - Max Points']) * lateness_penalty(grades['lab04 - Lateness (H:M:S)'])
    df['lab05'] = (grades['lab05'] / grades['lab05 - Max Points']) * lateness_penalty(grades['lab05 - Lateness (H:M:S)'])
    df['lab06'] = (grades['lab06'] / grades['lab06 - Max Points']) * lateness_penalty(grades['lab06 - Lateness (H:M:S)'])
    df['lab07'] = (grades['lab07'] / grades['lab07 - Max Points']) * lateness_penalty(grades['lab07 - Lateness (H:M:S)'])
    df['lab08'] = (grades['lab08'] / grades['lab08 - Max Points']) * lateness_penalty(grades['lab08 - Lateness (H:M:S)'])
    df['lab09'] = (grades['lab09'] / grades['lab09 - Max Points']) * lateness_penalty(grades['lab09 - Lateness (H:M:S)'])
    return df


# ---------------------------------------------------------------------
# QUESTION 5
# ---------------------------------------------------------------------


def lab_total(processed):
    num_labs = len(processed.columns)
    processed['sum'] = processed.sum(axis=1)
    processed['min_values'] = processed.min(axis=1)
    processed['avgs'] = (processed['sum'] - processed['min_values']) / (num_labs-1)
    return processed['avgs']


# ---------------------------------------------------------------------
# QUESTION 6
# ---------------------------------------------------------------------


def total_points(grades):
    df = pd.DataFrame()
    df['projects'] = 0.3 * projects_total(grades)
    df['labs'] = 0.2 * lab_total(process_labs(grades))
    df['midterm'] = 0.15 *( grades['Midterm'] / grades['Midterm - Max Points'])
    checkpoints_df = pd.DataFrame(columns=get_assignment_names(grades)['checkpoint'])
    for checkpoint in get_assignment_names(grades)['checkpoint']:
        checkpoints_df[checkpoint] = grades[checkpoint] / grades[checkpoint + ' - Max Points']        
    df['checkpoints'] = 0.025 * checkpoints_df.mean(axis=1)
    discussions_df = pd.DataFrame(columns=get_assignment_names(grades)['disc'])
    for discussion in get_assignment_names(grades)['disc']:
        discussions_df[discussion] = grades[discussion] / grades[discussion + ' - Max Points']
    df['discussions'] = 0.025 * discussions_df.mean(axis=1)
    df['final'] = 0.3 * grades['Final'] / grades['Final - Max Points']
    df['total_points'] = df.sum(axis=1)
    return df['total_points']


# ---------------------------------------------------------------------
# QUESTION 7
# ---------------------------------------------------------------------


def final_grades(total):
    letter_grades = pd.cut(total, bins=[-float("inf"), 0.6, 0.7, 0.8, 0.9, float("inf")], labels=['F', 'D', 'C', 'B', 'A'], right=False)
    return letter_grades

def letter_proportions(total):
    final_grades = final_grades(total)
    letter_counts = final_grades.value_counts(normalize=True)
    return letter_counts


# ---------------------------------------------------------------------
# QUESTION 8
# ---------------------------------------------------------------------


def raw_redemption(final_breakdown, question_numbers):
    ...
    
def combine_grades(grades, raw_redemption_scores):
    ...


# ---------------------------------------------------------------------
# QUESTION 9
# ---------------------------------------------------------------------


def z_score(ser):
    ...
    
def add_post_redemption(grades_combined):
    ...


# ---------------------------------------------------------------------
# QUESTION 10
# ---------------------------------------------------------------------


def total_points_post_redemption(grades_combined):
    ...
        
def proportion_improved(grades_combined):
    ...


# ---------------------------------------------------------------------
# QUESTION 11
# ---------------------------------------------------------------------


def section_most_improved(grades_analysis):
    ...
    
def top_sections(grades_analysis, t, n):
    ...


# ---------------------------------------------------------------------
# QUESTION 12
# ---------------------------------------------------------------------


def rank_by_section(grades_analysis):
    ...


# ---------------------------------------------------------------------
# QUESTION 13
# ---------------------------------------------------------------------


def letter_grade_heat_map(grades_analysis):
    ...
