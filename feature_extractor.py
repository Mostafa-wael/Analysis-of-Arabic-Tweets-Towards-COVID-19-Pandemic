import numpy as np
from gensim.models import Word2Vec
from sklearn.feature_extraction.text import CountVectorizer
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
import pickle

# Extract word embeddings for each tweet
def extractWordEmbeddings(data):
    model = Word2Vec(data, min_count=1, window=5, sg=0, vector_size=5000)
    #save the model
    pickle.dump(model, open('out/models/features/w2v_model.pkl', 'wb'))
    return model
    
# use the model to extract word embeddings
def getWordEmbeddings(model, word):
    if word in model.wv:
        return model.wv[word]
    return np.zeros(5000)

# get the word embeddings for each tweet
def getTweetsEmbeddings(model, tweets):
    return tweets.apply(lambda x: np.mean([getWordEmbeddings(model, word) for word in x], axis=0))


# BOW
def train_count_vectorizer(processed_train_corpus):
    vectorizer = None
    vectorizer = CountVectorizer(token_pattern=r'[^\s]+')
    vectorizer.fit(processed_train_corpus)
    #save the vectorizer
    pickle.dump(vectorizer, open('out/models/features/bow_model.pkl', 'wb'))
    
    return vectorizer

#tf-idf
def TFIDF(corpus):
    vectorizer = TfidfVectorizer()
    vectors = vectorizer.fit(corpus)
    pickle.dump(vectorizer, open('out/models/features/tfidf_model.pkl', 'wb'))
    return vectors


#---------------------------------main---------------------------------------

def get_feature_models(train_data):
    w2v_model = extractWordEmbeddings(train_data)
    train_data_strings = [' '.join(ele) for ele in train_data]
    bow_model = train_count_vectorizer(train_data_strings)
    tfidf_model = TFIDF(train_data_strings)


    return w2v_model, bow_model, tfidf_model

def get_features(w2v_model, bow_model, tfidf_model, train_data, include_w2v = True, include_bow = True, include_tfidf = True):
    w2v_features = getTweetsEmbeddings(w2v_model, train_data)
    train_data = [' '.join(ele) for ele in train_data]
    bow_features = bow_model.transform(train_data)
    bow_features = bow_features.toarray()
    tfidf_features = tfidf_model.transform(train_data)
    tfidf_features = tfidf_features.toarray()

    res = []
    for idx in range(len(w2v_features)):
        #temp = w2v_features[idx]
        temp = np.append(w2v_features[idx], tfidf_features[idx])
        #temp = np.append(temp, bow_features[idx])

        res.append(temp)
    
    if include_w2v and include_bow and include_tfidf:
        return res
    elif include_w2v:
        return w2v_features
    elif include_bow:
        return bow_features.tolist()
    elif include_tfidf:
        return tfidf_features.tolist()
    return None