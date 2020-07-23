"""
Consumer Complaints
Author: Georgios Spyrou
Date Last Updated: 16/07/2020

File: Contains the functions used for the Consumer Complaints project
"""

import pandas as pd
import re

import matplotlib.pyplot as plt
import seaborn as sns

import string 
from nltk import word_tokenize
from nltk.corpus import stopwords
from nltk import WordNetLemmatizer

stop_words = set(stopwords.words('english')) 


def plotNumberOfObservationsPerCategory(input_df: pd.core.frame.DataFrame,
                                        col: str):

    plt.figure(figsize=(8,10))
    sns.countplot(y=input_df[col],
                  order = input_df[col].value_counts().index)
    plt.title('Number of Observations per Product Category', fontweight="bold")
    plt.show()


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
        plt.show()
    except KeyError:
        print('agg_col does not correspond to a column that exists')


# Remove Stopwords
def tokenize_sentence(sentence: str, rm_stopwords=True,
                      rm_punctuation=True) -> list:
    """
    Tokenize a given string, and return the words as a list.
    The function offers functionality to exclude the words that are either
    a stopword or punctuation.
    """
    tokenized = word_tokenize(sentence)
    
    if rm_stopwords is True:
        tokenized = [x for x in tokenized if x not in stop_words]
     
    if rm_punctuation is True:
        tokenized = [x for x in tokenized if x not in string.punctuation]
    
    return tokenized


def lemmatize_sentence(sentence, return_form = 'string'):
    """
    Lemmatize a given string . 
    
    Input:
    ------
        sentence: 
            Sentence that we want to lemmatize each word. The input can be
            of the form of tokens (list) or the complete sentence (string).
        return_form: 
            Format of the return function. Can be either a string
            with the concatenated lemmatized words or a list of the 
            lemmatized words.
    Returns:
    -------
        If join_string = True then the function returns the
        lemmatized words as a sentence. Else it returns the words as a list.
    """
    # Handle the case where the input is the string without being tokenized
    if type(sentence) != list:
        sentence = re.findall(r"[\w']+|[.,!?;]", sentence)

    lemmatizer = WordNetLemmatizer()
    if return_form == 'string':
        return ' '.join([lemmatizer.lemmatize(word) for word in sentence])
    else:
        return [lemmatizer.lemmatize(word) for word in sentence]
