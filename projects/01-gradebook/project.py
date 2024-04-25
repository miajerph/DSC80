# project.py


import pandas as pd
import numpy as np
from pathlib import Path
import datetime

import plotly.express as px
import plotly.io as pio

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
    grades_copy = grades.copy()
    grades_copy.fillna(0, inplace=True)
    project_avg = 0.20 * (\
    ((grades_copy['project01'] + grades_copy['project01_free_response']) / (grades_copy['project01 - Max Points'] + grades_copy['project01_free_response - Max Points'])) + \
    ((grades_copy['project02'] + grades_copy['project02_free_response']) / (grades_copy['project02 - Max Points'] + grades_copy['project02_free_response - Max Points'])) + \
    (grades_copy['project03'] / grades_copy['project03 - Max Points']) + \
    (grades_copy['project04'] / grades_copy['project04 - Max Points']) + \
    ((grades_copy['project05'] + grades_copy['project05_free_response']) / (grades_copy['project05 - Max Points'] + grades_copy['project05_free_response - Max Points'])))
    return project_avg


# ---------------------------------------------------------------------
# QUESTION 3
# ---------------------------------------------------------------------


def lateness_penalty(col):
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
    grades_copy = grades.copy()
    grades_copy.fillna(0, inplace=True)
    df = pd.DataFrame(columns=(get_assignment_names(grades_copy)['lab']))
    df['lab01'] = (grades_copy['lab01'] / grades_copy['lab01 - Max Points']) * lateness_penalty(grades_copy['lab01 - Lateness (H:M:S)'])
    df['lab02'] = (grades_copy['lab02'] / grades_copy['lab02 - Max Points']) * lateness_penalty(grades_copy['lab02 - Lateness (H:M:S)'])
    df['lab03'] = (grades_copy['lab03'] / grades_copy['lab03 - Max Points']) * lateness_penalty(grades_copy['lab03 - Lateness (H:M:S)'])
    df['lab04'] = (grades_copy['lab04'] / grades_copy['lab04 - Max Points']) * lateness_penalty(grades_copy['lab04 - Lateness (H:M:S)'])
    df['lab05'] = (grades_copy['lab05'] / grades_copy['lab05 - Max Points']) * lateness_penalty(grades_copy['lab05 - Lateness (H:M:S)'])
    df['lab06'] = (grades_copy['lab06'] / grades_copy['lab06 - Max Points']) * lateness_penalty(grades_copy['lab06 - Lateness (H:M:S)'])
    df['lab07'] = (grades_copy['lab07'] / grades_copy['lab07 - Max Points']) * lateness_penalty(grades_copy['lab07 - Lateness (H:M:S)'])
    df['lab08'] = (grades_copy['lab08'] / grades_copy['lab08 - Max Points']) * lateness_penalty(grades_copy['lab08 - Lateness (H:M:S)'])
    df['lab09'] = (grades_copy['lab09'] / grades_copy['lab09 - Max Points']) * lateness_penalty(grades_copy['lab09 - Lateness (H:M:S)'])
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
    grades_copy = grades.copy()
    grades_copy.fillna(0, inplace=True)
    df = pd.DataFrame()
    df['projects'] = 0.3 * projects_total(grades_copy)
    df['labs'] = 0.2 * lab_total(process_labs(grades_copy))
    df['midterm'] = 0.15 * (grades_copy['Midterm'] / grades_copy['Midterm - Max Points'])
    checkpoints_df = pd.DataFrame(columns=get_assignment_names(grades_copy)['checkpoint'])
    for checkpoint in get_assignment_names(grades_copy)['checkpoint']:
        checkpoints_df[checkpoint] = grades_copy[checkpoint] / grades_copy[checkpoint + ' - Max Points']        
    df['checkpoints'] = 0.025 * checkpoints_df.mean(axis=1)
    discussions_df = pd.DataFrame(columns=get_assignment_names(grades_copy)['disc'])
    for discussion in get_assignment_names(grades_copy)['disc']:
        discussions_df[discussion] = grades_copy[discussion] / grades_copy[discussion + ' - Max Points']
    df['discussions'] = 0.025 * discussions_df.mean(axis=1)
    df['final'] = 0.3 * grades_copy['Final'] / grades_copy['Final - Max Points']
    df['total_points'] = df.sum(axis=1)
    return df['total_points']


# ---------------------------------------------------------------------
# QUESTION 7
# ---------------------------------------------------------------------


def final_grades(total):
    letter_grades = pd.cut(total, bins=[-float("inf"), 0.6, 0.7, 0.8, 0.9, float("inf")], labels=['F', 'D', 'C', 'B', 'A'], right=False)
    return letter_grades

def letter_proportions(total):
    fin_grades = final_grades(total)
    letter_counts = fin_grades.value_counts(normalize=True)
    return letter_counts


# ---------------------------------------------------------------------
# QUESTION 8
# ---------------------------------------------------------------------


def raw_redemption(final_breakdown, question_numbers):
    question_scores = final_breakdown.iloc[:, question_numbers]
    total_redemption_pts = question_scores.sum(axis=1)
    max_redemption_pts = question_scores.sum(axis=1).max()
    raw_redemption_score = total_redemption_pts / max_redemption_pts
    question_scores.fillna(0, inplace=True)
    scores = pd.DataFrame({'PID': final_breakdown['PID'], 'Raw Redemption Score': raw_redemption_score})
    return scores
    
def combine_grades(grades, raw_redemption_scores):
    combined_grades = pd.merge(grades, raw_redemption_scores, on='PID', how='left')
    return combined_grades


# ---------------------------------------------------------------------
# QUESTION 9
# ---------------------------------------------------------------------


def z_score(ser):
    mean = ser.mean()
    std = ser.std(ddof=0)
    z_scores = (ser - mean) / std
    return z_scores
    
def add_post_redemption(grades_combined):
    gc = grades_combined.copy()
    gc.fillna(0, inplace=True)
    gc['Midterm Score Pre-Redemption'] = gc['Midterm'].div(gc['Midterm - Max Points'])
    gc['Midterm Z Score Pre-Redemption'] = z_score(gc['Midterm Score Pre-Redemption'])
    gc['Redemption Z Score'] = z_score(gc['Raw Redemption Score'])
    gc['Midterm Score Post-Redemption'] = gc['Midterm Score Pre-Redemption']
    mean = gc['Midterm Score Pre-Redemption'].mean()
    std = gc['Midterm Score Pre-Redemption'].std(ddof=0)
    gc.loc[gc['Redemption Z Score'] > gc['Midterm Z Score Pre-Redemption'], 'Midterm Score Post-Redemption'] = (gc['Redemption Z Score'] * std) + mean 
    gc.drop(columns=['Redemption Z Score', 'Midterm Z Score Pre-Redemption'], inplace=True)
    return gc


# ---------------------------------------------------------------------
# QUESTION 10
# ---------------------------------------------------------------------


def total_points_post_redemption(grades_combined):
    tp = total_points(grades_combined)
    gc = add_post_redemption(grades_combined)
    tp -= (0.15 * gc['Midterm Score Pre-Redemption'])
    tp += (0.15 * gc['Midterm Score Post-Redemption'])
    return tp
        
def proportion_improved(grades_combined):
    gc = grades_combined.copy()
    gc['tp_before'] = total_points(grades_combined)
    gc['tp_after'] = total_points_post_redemption(grades_combined)
    gc['improved'] = gc['tp_after'] > gc['tp_before']
    return gc['improved'].mean() 



# ---------------------------------------------------------------------
# QUESTION 11
# ---------------------------------------------------------------------


def section_most_improved(grades_analysis):
    ga = grades_analysis.copy()
    ga['Improvement'] = ga['Letter Grade Post-Redemption'] < ga['Letter Grade Pre-Redemption']
    ga = ga.groupby('Section')['Improvement'].mean().idxmax()
    return ga
    
def top_sections(grades_analysis, t, n):
    top_students = grades_analysis[grades_analysis['Final'] / grades_analysis['Final - Max Points'] >= t]
    section_counts = top_students.groupby('Section').size()
    qualified_sections = section_counts[section_counts >= n]
    return np.array(sorted(qualified_sections.index))


# ---------------------------------------------------------------------
# QUESTION 12
# ---------------------------------------------------------------------


def rank_by_section(grades_analysis):
    section_groups = grades_analysis.groupby('Section')
    largest_section_size = section_groups.size().max()
    ranks_df = pd.DataFrame()
    
    for section, group in section_groups:
        sorted_group = group.sort_values(by='Total Points Post-Redemption', ascending=False)
        section_ranks = sorted_group['PID'].values.tolist()
        section_ranks += [''] * (largest_section_size - len(section_ranks))
        ranks_df[section] = section_ranks

    ranks_df.index = ranks_df.index + 1
    ranks_df.index.name = 'Section Rank'

    return ranks_df


# ---------------------------------------------------------------------
# QUESTION 13
# ---------------------------------------------------------------------


def letter_grade_heat_map(grades_analysis):
    pio.renderers.default = "plotly_mimetype+notebook"
    grouped = grades_analysis.groupby(['Letter Grade Post-Redemption', 'Section']).size().unstack(fill_value=0)
    proportions = grouped.div(grouped.sum(axis=1), axis=0)
    proportions = proportions.reindex(['A', 'B', 'C', 'D', 'F'], fill_value=0)
    sections = ['A{:02d}'.format(i) for i in range(1, 31)]
    proportions = proportions.reindex(columns=sections, fill_value=0)
    fig = px.imshow(
        proportions,
        labels=dict(x="Section", y="Letter Grade Post-Redemption"),
        x=proportions.columns,
        y=proportions.index,
        color_continuous_scale='rainbow', 
        title='Distribution of Letter Grades by Section',
    )
    fig.update_layout(font=dict(family="Arial, sans-serif"))
    return fig
