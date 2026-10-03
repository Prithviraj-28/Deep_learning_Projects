"""
Movie Review Sentiment Analysis using LSTM
==========================================
Trains a Long Short-Term Memory (LSTM) neural network on the IMDB
movie-reviews dataset to classify reviews as POSITIVE or NEGATIVE.

Model Pipeline:
    Raw Review - Embedding - LSTM - Dense (Sigmoid) - Positive / Negative

Author  : Marvellous
License : MIT
"""

# Step 1 - Import Required Libraries
from tensorflow.keras.datasets import imdb
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Embedding, LSTM, Dense
from tensorflow.keras.preprocessing.sequence import pad_sequences


# Step 2 - Hyper-parameters / Configuration
VOCAB_SIZE    = 10_000  # Keep the 10,000 most-frequent words
MAX_LENGTH    = 200     # Truncate / pad every review to 200 tokens
EMBEDDING_DIM = 32      # Each word is represented by 32 numbers
LSTM_UNITS    = 64      # Size of the LSTM hidden state
EPOCHS        = 3       # Number of full passes through the training data
BATCH_SIZE    = 64      # Reviews processed per gradient-update step
VAL_SPLIT     = 0.2     # Fraction of training data used for validation
