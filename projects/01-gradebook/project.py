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

# helpers for question 1

def set_keys(d, titles):
    for i in titles:
        d[i] = None
    

def set_values(d, titles, df):
    for i in titles:
        assignment = i.lower()
        if i == 'lab':
            d[i] = [col for col in df.columns if assignment in col.lower() and col[-2:].isdigit()]
        if i == 'project':
            d[i] = [col for col in df.columns if assignment in col.lower() and len(col) == 9]
        if i == 'midterm':
            d[i] = [col for col in df.columns if assignment in col.lower() and len(col) == 7]
        if i == 'final':
            d[i] = [col for col in df.columns if assignment in col.lower() and len(col) == 5]
        if i == 'disc':
            d[i] = [col for col in df.columns if assignment in col.lower() and col[-2:].isdigit()]
        if i == 'checkpoint':
            d[i] = [col for col in df.columns if assignment in col.lower() and col[-2:].isdigit()]
        
    return d
    
def get_assignment_names(grades):
    names = {}
    categories = ['lab', 'project', 'midterm', 'final', 'disc', 'checkpoint']
    set_keys(names, categories)
    final = set_values(names, categories, grades)
    return final
    


# ---------------------------------------------------------------------
# QUESTION 2
# ---------------------------------------------------------------------


def projects_total(grades):
    proj_cols = [col for col in grades.columns if col.startswith('project') and 'Max Points' not in col and "checkpoint" not in col and "Lateness" not in col]
    scores = grades[proj_cols].sum(numeric_only=True, axis=1)
    maxes = [col for col in grades.columns if "Max Points" in col and "project" in col and "checkpoint" not in col]
    possible_scores = grades[maxes].sum(numeric_only=True, axis=1)
    return scores / possible_scores

# ---------------------------------------------------------------------
# QUESTION 3
# ---------------------------------------------------------------------


def lateness_penalty(col):
    # given the column name of the lateness 
    # 2 hr grace period means you can submit 2 hrs after og deadline and get full credit
    # onwards from 2 hrs and up to/including 1 week, 10% penalty so 0.9 multiplier
    # more than 1 week and up to/including 2 weeks, 30% penalty so 0.7 multiplier
    # more than 2 weeks, 60% penalty so 0.4 multiplier 
    
    
    # hours : minutes: seconds
    penalties = []
    for time in col:
        # conditionals for each penalty cutoff
        if time == '00:00:00':
            penalties.append(1)
            
        # 2 hour grace period
        elif int(time.split(':')[0]) < 2:
            penalties.append(1)
        elif int(time.split(':')[0]) == 2:
            if int(time.split(':')[1]) > 0 or int(time.split(':')[2]) > 0:
                penalties.append(0.9)
                   
        # 1 week penalty, 30%
        elif int(time.split(':')[0]) < 168:
            penalties.append(0.9)
        elif int(time.split(':')[0]) == 168:
            if int(time.split(':')[1]) > 0 or int(time.split(':')[2]) > 0:
                   penalties.append(0.7)
                   
        # 2 week or more penalty, 60%
        elif int(time.split(':')[0]) < 336:
               penalties.append(0.7)
        elif int(time.split(':')[0]) == 336:
            if int(time.split(':')[1]) > 0 or int(time.split(':')[2]) > 0:
                penalties.append(0.4)
        elif int(time.split(':')[0]) > 336:
                 penalties.append(0.4)

    return pd.Series(penalties)

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
    # processed is a dataframe returned by process_labs
    # goal of this function is to drop the lowest lab then find the 
    # lab average for each student in the form of a series
    
    # steps --> if each row is a student then i have to subtract the min from each row
    # then i have to divide by (n - 1) labs (in this case 8 labs instead of 9)
    
    num_labs = len(processed.columns)
    vals = []
    
    for i in range(processed.shape[0]):
        row = (processed.iloc[i]).tolist()
        row.remove(min(row))
        drop_lowest = sum(row) / (num_labs - 1)
        vals.append(drop_lowest)
    return pd.Series(vals)
    


# ---------------------------------------------------------------------
# QUESTION 6
# ---------------------------------------------------------------------


def total_points(grades):
    # labs 20%
    # project 30%
    # checkpoints 2.5%
    # discussions 2.5%
    # midterm exam 15%
    # final 30%
    # formula for this is like
    # score/max x weight for each category 
    # add them all together and that's it i think?
    # only the labs has to be multiplied for lateness but i alr have a function for it
    
    categories = ['lab', 'project', 'midterm', 'final', 'disc', 'checkpoint']
    # put weights in the same order as the assignments
    weights = [0.2, 0.3, 0.15, 0.3, 0.025, 0.025]
    # maybe do a dictionary 
    
    overall = get_assignment_names(grades)
    sums_of_grades = np.zeros(grades.shape[0])
    grades = grades.fillna(0)
    
    for i in range(len(categories)):
        
        if categories[i] == 'lab':
            sums_of_grades += np.array((lab_total(process_labs(grades)) * weights[i]))
            
        if categories[i] == 'project':
            sums_of_grades += np.array(projects_total(grades) * weights[i])
                                   
        if categories[i] == 'midterm':
            # query this and return series of scores then multiply by weight[i]
            # midterm / midterm max points
            # return series of midterm scores
            sums_of_grades += np.array((grades['Midterm'].values / grades['Midterm - Max Points']) * weights[i])
  
        if categories[i] == 'final':
            sums_of_grades += np.array((grades['Final'].values / grades['Final - Max Points'].values) * weights[i])
                                   
        if categories[i] == 'disc':
            # get discussion columns and max points columns
            
            disc_grades = get_assignment_names(grades)['disc']
            disc_grades_df = grades[disc_grades].sum(numeric_only=True, axis=1)
            
            max_disc = [col for col in grades.columns if "Max Points" in col and "discussion" in col]
            max_disc_df = grades[max_disc].sum(numeric_only=True, axis=1)
            
            disc_result = disc_grades_df.values / max_disc_df.values
            sums_of_grades += np.array(disc_result * weights[i])
           
        if categories[i] == 'checkpoint':    
            
            # maybe my query is like contains checkpoint and does not contain lateness
            # get the checkpoints in one df and then the maxes in another df
            checkpt = get_assignment_names(grades)['checkpoint']
            checkpt_df = grades[checkpt].sum(numeric_only=True, axis=1)
            checkpt_max = [col for col in grades.columns if "Max Points" in col and "checkpoint" in col]
            checkpt_max_df = grades[checkpt_max].sum(numeric_only=True, axis=1)
            checkpt_result = checkpt_df.values / checkpt_max_df.values
            sums_of_grades += np.array(checkpt_result * weights[i])
                                   
    return pd.Series(sums_of_grades)


# ---------------------------------------------------------------------
# QUESTION 7
# ---------------------------------------------------------------------


def final_grades(total):
    letter_grades = pd.cut(total, bins=[-float("inf"), 0.6, 0.7, 0.8, 0.9, float("inf")], labels=['F', 'D', 'C', 'B', 'A'], right=False)
    return letter_grades

def letter_proportions(total):
    fin_grades = final_grades(total)
    letter_counts = fin_grades.value_counts(normalize=True)
    letter_counts = letter_counts.sort_values(ascending=False)
    return letter_counts

# ---------------------------------------------------------------------
# QUESTION 8
# ---------------------------------------------------------------------

def index_columns(df, indices):
    cols = df.iloc[:, indices]
    return cols

def raw_redemption(final_breakdown, question_numbers):
    redemption_df = index_columns(final_breakdown, question_numbers)

    numerator = redemption_df.sum(numeric_only=True, axis=1)
    
    denominator = index_columns(final_breakdown, question_numbers)
    denominator = redemption_df.max().sum()
    
    result_df = pd.DataFrame(final_breakdown['PID'])
    result_df['Raw Redemption Score'] = numerator / denominator
    result_df['Raw Redemption Score'].fillna(0, inplace=True)
    
    return result_df
    
def combine_grades(grades, raw_redemption_scores):
    grades_copy = grades.copy(deep=True)
#     x = grades_copy.merge(raw_redemption_scores, left_on='PID', right_on='PID', how='inner')
    return grades_copy.merge(raw_redemption_scores, left_on='PID', right_on='PID', how='inner')


# ---------------------------------------------------------------------
# QUESTION 9
# ---------------------------------------------------------------------


def z_score(ser):
    # formula is (x - mean of x) / sd of x
    sample_sd = np.std(ser, ddof=0)
    sample_mean = np.mean(ser)

    return (ser - sample_mean) / sample_sd


# helper for post redemption

def compare_scores(df):
    # make a dataframe containing the z-scores of the final and pre-redemption scores
    # iterate through this dataframe to determine which z score is higher
    # if the final z-score is higher, plug the mean and sd to find the new score
    # else keep pre-redemption score
    # append all values to an array and convert to series
    a = z_score(df['Midterm Score Pre-Redemption'])
    b = z_score(df['Raw Redemption Score'])
    data = {'Raw Redemption Score': b, 'Midterm Score Pre-Redemption': a}
    c = pd.DataFrame(data, index = range(535))

    final_redemption = []
    midterm_mean = np.mean(df['Midterm Score Pre-Redemption'])
    midterm_sd = np.std(df['Midterm Score Pre-Redemption'], ddof=0)
    for i in range(df.shape[0]):
        
        if (c['Raw Redemption Score'].iloc[i] == 0) and (c['Raw Redemption Score'].iloc[i] == 0):
            final_redemption.append(0)
            
        if c['Raw Redemption Score'].iloc[i] > c['Midterm Score Pre-Redemption'].iloc[i]:
            new_score = (c['Raw Redemption Score'].iloc[i] * midterm_sd) + midterm_mean
            final_redemption.append(new_score)
            
        else:
            final_redemption.append(df['Midterm Score Pre-Redemption'].iloc[i])
            
    return pd.Series(final_redemption)

def add_post_redemption(grades_combined):
    g = grades_combined.copy(deep=True)
    g['Midterm'].fillna(0, inplace=True)
    g['Midterm Score Pre-Redemption'] = g['Midterm'] / g['Midterm - Max Points']
    
    g['Midterm Score Post-Redemption'] = compare_scores(g)
    
    g.loc[g['Midterm Score Post-Redemption'] > 1, 'Midterm Score Post-Redemption'] = 1

    return g

# ---------------------------------------------------------------------
# QUESTION 10
# ---------------------------------------------------------------------


def total_points_post_redemption(grades_combined):
#     old_grades = total_points(grades_combined) # returns series of grades
    g = grades_combined.copy(deep=True) # make copy of df
    post_r = add_post_redemption(g)
    new_grades = redeem_total_grades(post_r)
    return new_grades
#     diffs = post_r['Midterm Score Post-Redemption'] - post_r['Midterm Score Pre-Redemption']
#     diffs = diffs * 0.15
#     return old_grades + diffs
        
def proportion_improved(grades_combined):
    g = grades_combined.copy(deep=True)
    post_r = total_points_post_redemption(g)
    pre_r = total_points(g)
    pre = final_grades(pre_r)
    post = final_grades(post_r)
    
    grade_order = ['A', 'B', 'C', 'D', 'F']

    grade_values = {'A': 5, 'B': 4, 'C': 3, 'D': 2, 'F': 1}

    pre = pre.map(grade_values)
    post = post.map(grade_values)
    improved = (post > pre).sum()
    total = len(pre)
    prop = improved / total
    return prop

def redeem_total_grades(grades):
    # labs 20%
    # project 30%
    # checkpoints 2.5%
    # discussions 2.5%
    # midterm exam 15%
    # final 30%
    # formula for this is like
    # score/max x weight for each category 
    # add them all together and that's it i think?
    # only the labs has to be multiplied for lateness but i alr have a function for it
    
    categories = ['lab', 'project', 'midterm', 'final', 'disc', 'checkpoint']
    # put weights in the same order as the assignments
    weights = [0.2, 0.3, 0.15, 0.3, 0.025, 0.025]
    # maybe do a dictionary 
    
    overall = get_assignment_names(grades)
    sums_of_grades = np.zeros(grades.shape[0])
    grades = grades.fillna(0)
    
    for i in range(len(categories)):
        
        if categories[i] == 'lab':
            sums_of_grades += np.array((lab_total(process_labs(grades)) * weights[i]))
            
        if categories[i] == 'project':
            sums_of_grades += np.array(projects_total(grades) * weights[i])
                                   
        if categories[i] == 'midterm':
            # query this and return series of scores then multiply by weight[i]
            # midterm / midterm max points
            # return series of midterm scores
            sums_of_grades += np.array(grades['Midterm Score Post-Redemption'].values * weights[i])
#                 (grades['Midterm'].values / grades['Midterm - Max Points']) * weights[i])
  
        if categories[i] == 'final':
            sums_of_grades += np.array((grades['Final'].values / grades['Final - Max Points'].values) * weights[i])
                                   
        if categories[i] == 'disc':
            # get discussion columns and max points columns
            
            disc_grades = get_assignment_names(grades)['disc']
            disc_grades_df = grades[disc_grades].sum(numeric_only=True, axis=1)
            
            max_disc = [col for col in grades.columns if "Max Points" in col and "discussion" in col]
            max_disc_df = grades[max_disc].sum(numeric_only=True, axis=1)
            
            disc_result = disc_grades_df.values / max_disc_df.values
            sums_of_grades += np.array(disc_result * weights[i])
           
        if categories[i] == 'checkpoint':    
            
            # maybe my query is like contains checkpoint and does not contain lateness
            # get the checkpoints in one df and then the maxes in another df
            checkpt = get_assignment_names(grades)['checkpoint']
            checkpt_df = grades[checkpt].sum(numeric_only=True, axis=1)
            checkpt_max = [col for col in grades.columns if "Max Points" in col and "checkpoint" in col]
            checkpt_max_df = grades[checkpt_max].sum(numeric_only=True, axis=1)
            checkpt_result = checkpt_df.values / checkpt_max_df.values
            sums_of_grades += np.array(checkpt_result * weights[i])
                                   
    return pd.Series(sums_of_grades)
# ---------------------------------------------------------------------
# QUESTION 11
# ---------------------------------------------------------------------


def section_most_improved(grades_analysis):
    grade_order = ['A', 'B', 'C', 'D', 'F']
    grade_values = {'A': 5, 'B': 4, 'C': 3, 'D': 2, 'F': 1}
    proportions = []
    num_sections = 30
    section_names = np.array([f"{i:02d}" for i in range(1, 31)])
    
    for i in range(num_sections):
        section = grades_analysis[grades_analysis['Section'] == ('A' + section_names[i])]
        grades_pre = section['Letter Grade Pre-Redemption']
        grades_post = section['Letter Grade Post-Redemption']
        grades_pre = pd.Series(grades_pre.values)
        grades_post = pd.Series(grades_post.values)
        pre = grades_pre.map(grade_values)
        post = grades_post.map(grade_values)
        total = len(pre)
        improved = (post > pre).sum()
        proportion = improved / total
        proportions.append(proportion)
    return 'A' + section_names[proportions.index(max(proportions))]
    
def top_sections(grades_analysis, t, n):
    # t is a float between 0 and 1
    # t represents the score on the final exam
    # n represents students
    # at least n students 
    # return array in alphanumeric order of sections
    # where at least n students scored at least t on the final
    
    num_sections = len(grades_analysis['Section'].unique())
    section_names = np.array([f"{i:02d}" for i in range(1, num_sections + 1)])
    section_names_letters = ['A' + s for s in section_names]
    meet_criteria = []
    
    for i in range(30):
        section = grades_analysis[grades_analysis['Section'] == ('A' + section_names[i])]
        series = section['Final'] / section['Final - Max Points']
        min_t = (series >= t).sum()
        if min_t >= n:
            meet_criteria.append(True)
        else:
            meet_criteria.append(False)
        
    result = np.array([val for val, flag in zip(section_names_letters, meet_criteria) if flag])
    return result


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
