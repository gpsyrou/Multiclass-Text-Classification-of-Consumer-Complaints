"""
Consumer Complaints
Author: Georgios Spyrou
Date Last Updated: 15/07/2020
"""

import os
import pandas as pd

import matplotlib.pyplot as plt
import seaborn as sns


project_dir = 'C:\\Users\\george\\Desktop\\GitHub\\Projects\\Consumer_Complaints'
os.chdir(project_dir)

complaints_df = pd.read_csv(
        os.path.join(project_dir, 'Data\cons_complaints_dataset.csv'))



complaints_df.dtypes
# The only column that is not text is the ID of the complaint

# Have a look at what columns the dataset contains
complaints_df.columns


# Identify how many missing values we have per column
complaints_df.isnull().sum(axis=0)


# Part 1. Exploratory Data Analysis

# Main purpose of this project is to use the 'Consumer complaint narrative'
# column, in order to predict to which category (defined by 'Product' column)
# the complaint belongs to.

# Identify the amount of times that each Category (Product) is present in our
# dataset. We have no missing values for this column, so no imputation method
# is necessary.
from Functions import consumer_complaints_functions as ccf

ccf.plotNumberOfObservationsPerCategory(complaints_df)

# Find states that most complaints have been submitted to
most_complaints_by_state = complaints_df[['Complaint ID',
                    'State']].groupby(['State']).agg(['count'])

most_complaints_by_state = most_complaints_by_state.sort_values(
        by=[('Complaint ID','count')], ascending=False)

# Plot the results
top_n = 10 

plt.figure(figsize=(10,8))
sns.barplot(x=most_complaints_by_state.index[0:top_n],
            y=('Complaint ID','count'),
            data = most_complaints_by_state[0:top_n])

plt.ylabel('Number of complaints')
plt.title('States with the most number of complaints', fontweight="bold")

# Find companies that received the most complaints from their consumers
most_complaints_by_company = complaints_df[['Complaint ID',
                    'Company']].groupby(['Company']).agg(['count'])

most_complaints_by_company = most_complaints_by_company.sort_values(
        by=[('Complaint ID','count')], ascending=False)

top_n = 5 

plt.figure(figsize=(12,10))
sns.barplot(x=most_complaints_by_company.index[0:top_n],
            y=('Complaint ID','count'),
            data = most_complaints_by_company[0:top_n])

# Filter the dataset to retain only the rows for which the
# 'Consumer complaint narrative' column is populated (i.e. we have input
# from the consumer regarding the complaint that they are submitting)
compl_w_text = complaints_df[complaints_df
                                ['Consumer complaint narrative'].notnull()]

ccf.plotNumberOfObservationsPerCategory(compl_w_text)
# Its interesting to see that the category with the most complaints is now
# the 'Credit reporting, credit repair services, or other personal consumer
# reports' instead of 'Mortgage' that was first when we were using the whole
# dataset.

