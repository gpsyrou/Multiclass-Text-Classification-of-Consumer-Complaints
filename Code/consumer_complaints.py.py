"""
Consumer Complaints
Data source: 
    https://www.kaggle.com/selener/consumer-complaint-database
    https://catalog.data.gov/dataset/consumer-complaint-database

Context
These are real world complaints received about financial products and services.
 Each complaint has been labeled with a specific product; therefore, 
 this is a supervised text classification problem. With the aim to classify 
 future complaints based on its content, we used different machine learning
 algorithms can make more accurate predictions (i.e., classify the complaint 
 in one of the product categories)

Content
The dataset contains different information of complaints that customers
 have made about a multiple products and services in the financial sector,
 such us Credit Reports, Student Loans, Money Transfer, etc.
The date of each complaint ranges from November 2011 to May 2019.

Acknowledgements
This work is considered a U.S. Government Work. 
The dataset is public dataset and it was downloaded from
https://catalog.data.gov/dataset/consumer-complaint-database
on 2019, May 13.


"""

import os
import pandas as pd

import matplotlib.pyplot as plt
import seaborn as sns

project_dir = 'C:\\Users\\george\\Desktop\\GitHub\\Projects\\Consumer_Complaints'
os.chdir(project_dir)

cons_complaints_df = pd.read_csv(
        os.path.join(project_dir, 'Data\cons_complaints_dataset.csv'))



cons_complaints_df.dtypes()
# The only column that is not text is the ID of the complaint

# Have a look at what columns the dataset contains
cons_complaints_df.columns


# Identify how many missing values we have per column
cons_complaints_df.isnull().sum(axis=0)


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
