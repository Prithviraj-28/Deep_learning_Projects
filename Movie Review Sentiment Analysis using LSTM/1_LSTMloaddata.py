#-------------------------------------------------------------------------------------
# Step 1 . Import Required ibraries
#-------------------------------------------------------------------------------------

from tensorflow.keras.datasets import imdb
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Embedding, LSTM , Dense
from tensorflow.keras.preprocessing.sequence import pad_sequences

#-------------------------------------------------------------------------------------
# Step 2 . Configuration of values 
#-------------------------------------------------------------------------------------

VOCAB_SIZE = 10000  #cONSIDER MOST FREQUENT 10000 UNIQUE WORDS
MAX_LENGTH = 200    #CONSIDER MAXIMUM 200 WORDS IN REVIEW 


#-------------------------------------------------------------------------------------
# Step 3 . Load the IMDB (International Movies DataBase)dataset  
#-------------------------------------------------------------------------------------
print("#"*40)
print("Movie Review Sentiment Analysis Using LSTM ")
print("#"*40)

print("Loading Dataset............")

(X_train , Y_train),(X_test,Y_test) = imdb.load_data(num_words = VOCAB_SIZE)

print("IMDB Dataset loded sucessfully")

print("Number of traning reviwes : ",len(X_train))
print("Number of testing reviwes : ",len(X_test))

#-------------------------------------------------------------------------------------
#   X_train : Reviwes used for traning 
#   Y_train : Actual sentiment of traning
#   X_test  : Reviwes used for testing
#   Y_test  : Actual sentiments of testing
# 
# Sentiments:
# 0 -> Negative Sentiment 
# 1 -> Positive sentiment 
#-------------------------------------------------------------------------------------

#-------------------------------------------------------------------------------------
# Step 4 . Load the Word Dictonary  
#-------------------------------------------------------------------------------------

word_index = imdb.get_word_index()

#Dictonary contains mapping of words and its corresponding Number 
#Drishyam is a good Movie -> (20 56 78 43)
#20 -> drishyam
#56 -> is
#78 -> good
#43 -> movie

#-------------------------------------------------------------------------------------
# Step 5 . Creat reverse dictonary 
#-------------------------------------------------------------------------------------

reverse_word_index = {}

for word,index in word_index.items():
    reverse_word_index[index+3] = word