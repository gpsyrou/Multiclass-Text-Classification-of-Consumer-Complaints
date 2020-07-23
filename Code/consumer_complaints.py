"""
Consumer Complaints
Author: Georgios Spyrou
Date Last Updated: 15/07/2020
"""

import os
import pandas as pd

import matplotlib.pyplot as plt
import seaborn as sns

import re


project_dir = 'C:\\Users\\george\\Desktop\\GitHub\\Projects\\Consumer_Complaints'
os.chdir(project_dir)

compl_full = pd.read_csv(
        os.path.join(project_dir, 'Data\cons_complaints_dataset.csv'))


compl_full.dtypes
# The only column that is not text is the ID of the complaint

# Have a look at what columns the dataset contains
compl_full.columns

# Renaming the predictor column for ease of use
compl_full.rename(columns={'Consumer complaint narrative':'Complaint'},
                  inplace=True)

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

ccf.plotNumberOfObservationsPerCategory(compl_full, col='Product')

# Find states that most complaints have been submitted to
ccf.plotTopComplaints(compl_full,
                      agg_col='State', top_n=10, bottom=False)

# Find companies that received the most complaints from their consumers
ccf.plotTopComplaints(compl_full,
                      agg_col='Company', top_n=10, bottom=False)



# Filter the dataset to retain only the rows for which the
# 'Consumer complaint narrative' column is populated (i.e. we have input
# from the consumer regarding the complaint that they are submitting)
compl_w_text = compl_full[compl_full['Complaint'].notnull()]

ccf.plotNumberOfObservationsPerCategory(compl_w_text, col='Product')
# Its interesting to see that the category with the most complaints is now
# the 'Credit reporting, credit repair services, or other personal consumer
# reports' instead of 'Mortgage' that was first when we were using the whole
# dataset.

# Part 2. Preprocessing

# Preprocessing - Target Variable

# In order to build our classification model, we will need only the
# 'Consumer complaint narrative' column as the predictor  variable
#  and 'Product' as the target variable
relevant_cols = ['Complaint', 'Product']
main_df = compl_w_text[relevant_cols]

main_df.shape

print(f'There are {main_df.shape[0]} instances of complaints distributed among'
                   f' {len(main_df.Product.unique())} different categories')

# We are going to transform the Product from text into numerical values
from sklearn.preprocessing import LabelEncoder, OneHotEncoder

lab_enc = LabelEncoder()
main_df['Category'] = lab_enc.fit_transform(main_df['Product'])

ccf.plotNumberOfObservationsPerCategory(main_df, col='Category')



# 1. Split each of the rows (corresponding to a complaint) into tokens (tokenizing)
# and remove stopwords

# Tokenize
from nltk import word_tokenize

main_df['Complaint_Clean'] = main_df['Complaint'].apply(
        lambda x: word_tokenize(x))

# Remove Stopwords



# 2. Lemmatize each of the above
from nltk import WordNetLemmatizer

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

# 3. Split the data to train and test sets
# 4. Create the pipeline
# 5. Create the parameter Grid
# 6. Fit the model
# 7. Review performance







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
# document. Word that appear many times in one document are getting higher
# significance for that document if they do not appear in other documents. On
# the other hand, common words like 'and' and 'the' are usually common in all
# documents and therefore they do not provide much information.

from sklearn.feature_extraction.text import TfidfVectorizer

tfidf = TfidfVectorizer(analyzer='word', encoding='utf-8', min_df=10,
                        norm='l2', stop_words='english')

feat = tfidf.fit_transform(main_df.Complaint) # feat.data gives me the array
