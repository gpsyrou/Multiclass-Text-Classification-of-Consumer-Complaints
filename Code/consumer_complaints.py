"""
Multitext Classification of Consumer Complaints
Author: Georgios Spyrou
Date Last Updated: 10/08/2020

File: Contains the main analysis of the project like the EDA, data cleaning and
      preprocessing, model creation, model evaluation

"""

# Import dependencies
import os
import numpy as np
import pandas as pd

import re

# Visualization
import matplotlib.pyplot as plt
import seaborn as sns

# Part 1 - Set up project directory and read dataset
project_dir = r'C:\Users\george\Desktop\GitHub\Projects\Consumer_Complaints'
os.chdir(project_dir)

# Read the whole dataset into a dataframe
complaints_df = pd.read_csv(os.path.join(project_dir,
                                           'Data', 'complaints.csv'))

complaints_df.dtypes
# The only column that is not text is the ID of the complaint

# Have a look at what columns the dataset contains
complaints_df.columns

# Part 2 - Exploratory Data Analysis & Data Cleaning

# From this dataset the only two columns that are in scope are the compaints
# - which we can find in the 'Consumer complaint narrative' column -  and the
# 'Product' column which is the category of the complaint (and which is going
# to be our target variable). Thus, based on the text that we see in the former
# we will attempt to classify the text into one of the categories in 'Product'

# Renaming the predictor column for ease of use
complaints_df.rename(columns={'Consumer complaint narrative': 'Complaint'},
                  inplace=True)

# Identify how many missing values we have per column 
complaints_df.isnull().sum(axis=0)

# Get the year that the complaint took place as a separate column
complaints_df['Year'] = complaints_df['Date received'].apply(lambda x: int(
        re.findall('[0-9]{4}', x)[0]))
    
complaints_df.Year.value_counts()

# Identify the amount of times that each Category (Product) is present in our
# dataset. We have no missing values for this column, so no imputation method
# is necessary.
from Functions import consumer_complaints_functions as ccf

ccf.plotNumberOfObservationsPerCategory(complaints_df, col='Product')
# We can observe that the initial dataset has some imbalance in terms of the
# categories of the complaints. Appears that the most regular complaints are
# related to credit reporting, mortage, debt collection and credit reporting.

# Find states that most complaints have been submitted to
ccf.plotTopComplaints(complaints_df, agg_col='State', top_n=10, bottom=False)

# Find companies that received the most complaints from their consumers
ccf.plotTopComplaints(complaints_df, agg_col='Company', top_n=10, bottom=False,
                      figsize=(14,12))

# Filter the dataset to retain only the rows for which the
# 'Consumer complaint narrative' column is populated (i.e. we have input
# from the consumer regarding the complaint that they are submitting)
complaints_df = complaints_df[complaints_df['Complaint'].notnull()]
complaints_df.shape

# In order to build our classification model, we will need only the
# 'Consumer complaint narrative' column as the predictor  variable
#  and 'Product' as the target variable
relevant_cols = ['Complaint', 'Product']
complaints_processed = complaints_df[relevant_cols]

complaints_processed.shape
complaints_processed.isnull().sum(axis=0)

del complaints_df

# We can see that some of the categories appear to overlap with each other
# For example 'credit reporting' and 'credit reporting,credit repair....'
# For such cases we will combine them into one category

prod_category_map = {'Credit reporting, credit repair services, or other personal consumer reports': 'Credit reporting',
                     'Credit card': 'Credit card or prepaid card',
                     'Payday loan': 'Payday loan, title loan, or personal loan',
                     'Prepaid card': 'Credit card or prepaid card',
                     'Money transfers': 'Money transfer, virtual currency, or money service',
                     'Virtual currency': 'Money transfer, virtual currency, or money service'}

complaints_processed['Product'].replace(prod_category_map, inplace=True)

ccf.plotNumberOfObservationsPerCategory(complaints_processed, col='Product')

# We are going to transform the Product from text into numerical values
from sklearn.preprocessing import LabelEncoder

label_encoder = LabelEncoder()
complaints_processed['Product_Id'] = label_encoder.fit_transform(
        complaints_processed['Product'])

ccf.plotNumberOfObservationsPerCategory(complaints_processed, col='Product_Id')

# Also its good to have the categories as a dictionary
product_map = complaints_processed.set_index('Product_Id').to_dict()['Product']

# 1. Split each of the rows (corresponding to a complaint) into tokens
# and remove stopwords

# Tokenize
complaints_processed['Complaint_Tokenized'] = complaints_processed.apply(lambda
                    x: ccf.tokenize_sentence(x['Complaint'], rm_stopwords=True,
                                             rm_punctuation=True,
                                             rm_numbers=True,
                                             rm_classified=True), axis=1)

# 2. Lemmatize each of the above
complaints_processed['Complaint_Clean'] = complaints_processed.apply(lambda x:
    ccf.lemmatize_sentence(x['Complaint_Tokenized'],
                           return_form='string'), axis=1)

# 3. Split the data to train and test sets
from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV

X = complaints_processed['Complaint_Clean']
y = complaints_processed['Product_Id']

# Use the stratify parameter in order to split the target variabe (categories)
# evenly among train and test sets.
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2,
                                                    random_state=42,
                                                    stratify=y)


# What is the distribution of the categories (target) in the training set ?
    
# 4. Create the pipeline
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.pipeline import Pipeline
from sklearn.naive_bayes import MultinomialNB

pipeline_mnb = Pipeline(steps = [('TfIdf', TfidfVectorizer()),
                              ('MultinomialNB', MultinomialNB())])


# 5. Create the parameter Grid
param_grid = {
 'TfIdf__max_features' : [1000, 2000, 3000],
 'TfIdf__min_df': [5, 10, 20],
 'TfIdf__ngram_range' : [(1,1),(1,2)],
 'TfIdf__use_idf' : [True, False],
 'MultinomialNB__alpha' : [0.1, 0.5, 1]
}  


# 6. Fit the model and evalute the scores
grid_search_mnb = GridSearchCV(pipeline_mnb, param_grid, cv=5,
                               verbose=1, n_jobs=-1)

grid_search_mnb.fit(X_train, y_train)

# Check the score on the training and test sets
grid_search_mnb.score(X_train, y_train)

grid_search_mnb.score(X_test, y_test)

# Observe which were the best parameters for the model
grid_search_mnb.best_params_

predicted = grid_search_mnb.predict(X)
complaints_processed['Predicted_Category'] = predicted


# 7. Review performance
from sklearn.metrics import confusion_matrix
predicted = grid_search_mnb.predict(X_test)
conf_matrix = confusion_matrix(y_test, predicted)
conf_matrix

plt.figure(figsize=(14,14))
sns.heatmap(conf_matrix, annot=True)

plt.figure(figsize=(14,14))

sns.heatmap(conf_matrix/np.sum(conf_matrix), annot=True, 
            fmt='.2%', cmap='Blues')
plt.ylabel('True')
plt.xlabel('Predicted')

# We can observe that our dataset it's highly imbalanced regarding the
# distribution of the product categories. Most of the product are falling under
# the credit reporting, debit collection and mortgage categories. Imbalanced 
# datasets can usually be a major issue as they can lead to misleading results.
# This is because the algorithm will seem to be predicting 'correctly' and 
# achieving high accuracy scores, while this will be due to the fact that its
# only predicting correctly the majority class.

# Preprocessing - Predictor variable

# Most of the algorithms do not work well when they have to deal with text data
# in their raw form. To solve this issue we will work with some techniques such
# like the bag of words and TF-IDF.

# Bag of words refers to the process of creating a vector of word counts of
# a document (which in our case refers to the complaints made by the consumers)
# In simple words, its the number that each word appears in a document. Please
# have in mind that Bag of Words does not take into consideration the order of
# the words neither any grammatical rules.

# TF-IDF is being used to get an understanding of how relevant a word is on a 
# document. Words that appear many times in one document are getting higher
# significance for that document if they do not appear in other documents. On
# the other hand, common words like 'and' and 'the' are usually common in all
# documents and therefore they do not provide much information.

'''
from sklearn.feature_extraction.text import TfidfVectorizer

tfidf = TfidfVectorizer(analyzer='word', encoding='utf-8', min_df=10,
                        norm='l2', stop_words='english')

feat = tfidf.fit_transform(main_df.Complaint) # feat.data gives me the array
'''

