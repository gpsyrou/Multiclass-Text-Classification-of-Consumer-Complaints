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

project_dir = 'C:\\Users\\george\\Desktop\\GitHub\\Projects\\Consumer_Complaints'
os.chdir(project_dir)

data = pd.read_csv(os.path.join(project_dir,
                                'Data\cons_complaints_dataset.csv'))
small = data.iloc[0:1000]