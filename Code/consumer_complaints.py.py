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

cons_complaints_df = pd.read_csv(
        os.path.join(project_dir, 'Data\cons_complaints_dataset.csv'))



cons_complaints_df.dtypes
# The only column that is not text is the ID of the complaint

# Have a look at what columns the dataset contains
cons_complaints_df.columns


# Identify how many missing values we have per column
cons_complaints_df.isnull().sum(axis=0)

small = cons_complaints_df[0:10000]

# Main purpose of this project is to use the 'Consumer complaint narrative'
# column, in order to predict to which category (defined by 'Product' column)
# the complaint belongs to.

# Identify the amount of times that each Category (Product) is present in our
# dataset. We have no missing values for this column, so no imputation method
# is necessary.

plt.figure(figsize=(8,10))
sns.countplot(y=cons_complaints_df['Product'],
              order = cons_complaints_df['Product'].value_counts().index)
plt.title('Number of Observations per Product Category', fontweight="bold")

# Find states that most complaints have been submitted to
most_complaints = cons_complaints_df[['Complaint ID',
                    'State']].groupby(['State']).agg(['count'])

most_complaints = most_complaints.sort_values(by=[('Complaint ID','count')],
                                                  ascending=False)
# Plot the results
top_n = 10 
plt.figure(figsize=(10,8))
sns.barplot(x=most_complaints.index[0:top_n], y=('Complaint ID','count'),
            data = most_complaints[0:top_n])
plt.ylabel('Number of complaints')
plt.title('States with the most number of complaints')