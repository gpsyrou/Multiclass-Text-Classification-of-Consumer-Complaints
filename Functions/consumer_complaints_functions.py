"""
Consumer Complaints
Author: Georgios Spyrou
Date Last Updated: 16/07/2020

File: Contains the functions used for the Consumer Complaints project
"""

import pandas as pd

import matplotlib.pyplot as plt
import seaborn as sns

def plotNumberOfObservationsPerCategory(input_df):
    
    plt.figure(figsize=(8,10))
    sns.countplot(y=input_df['Product'],
                  order = input_df['Product'].value_counts().index)
    plt.title('Number of Observations per Product Category', fontweight="bold")
    

def plotTopComplaints(input_df: pd.core.frame.DataFrame,
                      agg_col: str, top_n: int, bottom=False, figsize=(10,8)):
    """
    Aggregate a dataframe based on column of interest and calcualte the number
    of observations per aggregated group.
    
    The function returns a barplot object showing the results of the above
    calculation.
    
    Args:
    ----
    agg_col: Name of the column that we want to base the aggregation
    top_n: Amount of observations to be included in the plot
    bottom: Plot the top-n from the top (highest) or from the bottom (lowest)
    """
    try:
        most_cmplts = input_df[['Complaint ID',
                            agg_col]].groupby([agg_col]).agg(['count'])
    
        most_cmplts = most_cmplts.sort_values(
                by=[('Complaint ID','count')], ascending=bottom)
            
        plt.figure(figsize=figsize)
        sns.barplot(x=most_cmplts.index[0:top_n], y=('Complaint ID','count'),
                    data = most_cmplts[0:top_n])
        
        plt.ylabel('Number of complaints')
        plt.title(f'{agg_col} with the most number of complaints',
                  fontweight="bold")
    except KeyError:
        print('agg_col does not correspond to a column that exists')