import numpy as np
from gensim.models import Word2Vec

# Extract word embeddings for each tweet
def extractWordEmbeddings(data):
    model = Word2Vec(data, min_count=1, window=5, sg=0)
    model.save('out/models/word2vec.model')
    return model
    
# use the model to extract word embeddings
def getWordEmbeddings(model, word):
    return model.wv[word]

# get the word embeddings for each tweet
def getTweetsEmbeddings(model, tweets):
    return tweets.apply(lambda x: np.mean([getWordEmbeddings(model, word) for word in x], axis=0))