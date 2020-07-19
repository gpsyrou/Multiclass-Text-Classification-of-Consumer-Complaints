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

compl_full = pd.read_csv(
        os.path.join(project_dir, 'Data\cons_complaints_dataset.csv'))


compl_full.dtypes
# The only column that is not text is the ID of the complaint

# Have a look at what columns the dataset contains
compl_full.columns


# Identify how many missing values we have per column
compl_full.isnull().sum(axis=0)

# Get the year that the complaint took place as a separate column
compl_full['Year'] = compl_full['Date received'].apply(lambda x: int(x[-4:]))

# Part 1. Exploratory Data Analysis (EDA)

# Main purpose of this project is to use the 'Consumer complaint narrative'
# column, in order to predict to which category (defined by 'Product' column)
# the complaint belongs to.

# Identify the amount of times that each Category (Product) is present in our
# dataset. We have no missing values for this column, so no imputation method
# is necessary.
from Functions import consumer_complaints_functions as ccf

ccf.plotNumberOfObservationsPerCategory(compl_full)

# Find states that most complaints have been submitted to
ccf.plotTopComplaints(compl_full,
                      agg_col='State', top_n=10, bottom=False)

# Find companies that received the most complaints from their consumers
ccf.plotTopComplaints(compl_full,
                      agg_col='Company', top_n=10, bottom=False)



# Filter the dataset to retain only the rows for which the
# 'Consumer complaint narrative' column is populated (i.e. we have input
# from the consumer regarding the complaint that they are submitting)
compl_w_text = compl_full[compl_full['Consumer complaint narrative'].notnull()]

ccf.plotNumberOfObservationsPerCategory(compl_w_text)
# Its interesting to see that the category with the most complaints is now
# the 'Credit reporting, credit repair services, or other personal consumer
# reports' instead of 'Mortgage' that was first when we were using the whole
# dataset.

# Part 2. Preprocessing

# In order to build our classification model, we will need only the
# 'Consumer complaint narrative' column as the predictor  variable
#  and 'Product' as the target variable