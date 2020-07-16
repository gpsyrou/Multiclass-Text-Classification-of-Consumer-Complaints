"""
Consumer Complaints
Author: Georgios Spyrou
Date Last Updated: 16/07/2020

File: Contains the functions used for the Consumer Complaints project
"""

import matplotlib.pyplot as plt
import seaborn as sns

def plotNumberOfObservationsPerCategory(input_df):
    
    plt.figure(figsize=(8,10))
    sns.countplot(y=input_df['Product'],
                  order = input_df['Product'].value_counts().index)
    plt.title('Number of Observations per Product Category', fontweight="bold")