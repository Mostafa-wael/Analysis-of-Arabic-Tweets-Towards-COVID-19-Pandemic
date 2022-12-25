# For Data
import numpy as np
import pandas as pd
import re


# For NLP
import nltk
from nltk.corpus import stopwords
from nltk.stem.isri import ISRIStemmer
from spellchecker import SpellChecker

# Downloading periphrals
nltk.download('vader_lexicon')
nltk.download('stopwords')




def CleanTweets(df, clearData =False):
    data = df.copy() # Copying the dataset
    ##### Related to the tweets #####
    # Remove twitter handlers
    data.text = data.text.apply(lambda x:re.sub('@[^\s]+','',x))
    # Remove digits
    data.text = data.text.apply(lambda x:re.sub(r'\d+','',x))
    # Remove all the (special characters, punctuations, and emojis)
    data.text = data.text.apply(lambda x:' '.join(re.findall(r'\w+', x)))
    # Remove all english alphabets
    data.text = data.text.apply(lambda x:re.sub(r'[a-zA-Z]', '', x))
    # Substituting multiple spaces with single space
    data.text = data.text.apply(lambda x:re.sub(r'\s+', ' ', x, flags=re.I))
    # Remove all the empty spaces
    data.text = data.text.apply(lambda x: x.strip())
    # Remove all the stopwords
    data.text = data.text.apply(lambda x: ' '.join([word for word in x.split() if word not in (stopwords.words('arabic'))]))
    ##### Related to the dataset ##### 
    if clearData:
        # Remove all the empty rows
        data = data[data.text != '']
        # Removing the duplicated rows
        data = data.drop_duplicates()
        # Removing the duplicated tweets
        data = data.drop_duplicates(subset=['text'])
        # Removing the tweets with less than 10 characters
        data = data[data.text.str.len() > 10]
        # Removing the tweets with less than 4 words
        data = data[data.text.str.split().str.len() > 3]
        # Resetting the index, why? because we removed some rows
        data = data.reset_index(drop=True)
    return data

def cleanData(df, name, clean = False, clearData=False):  # If you want to clean the data, set it to True. Default is False to save time
    if clean:
        data = CleanTweets(df, clearData)
        data.to_csv('out/' + name + '_data_cleaned.csv', index=False) # print the df in a csv file
        data.head() # Displaying the dataset
    else:
        # Read the cleaned data
        data = pd.read_csv('out/' + name + '_data_cleaned.csv')
    return data




def processing(data):
    # Apply Lemmatization to the tweets
    st = ISRIStemmer()
    data['Lemmatization'] = data.text.apply(lambda x: ''.join([st.stem(word) for word in x.split()]))
    # Extract Sentiment Values for each tweet 
    data['sentiment'] = data['stance'].apply(lambda x: 
                                                'positive' if x == 1 
                                                else ('negative' if x == -1 
                                                else 'neutral' )); # Extracting the overall sentiment

    return data