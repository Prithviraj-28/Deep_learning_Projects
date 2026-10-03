"""
Movie Review Sentiment Analysis using LSTM
==========================================
Author  : Marvellous
License : MIT
"""

# Step 1 - Import Required Libraries
from tensorflow.keras.datasets import imdb
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Embedding, LSTM, Dense
from tensorflow.keras.preprocessing.sequence import pad_sequences

# Step 2 - Hyper-parameters / Configuration
VOCAB_SIZE    = 10_000
MAX_LENGTH    = 200
EMBEDDING_DIM = 32
LSTM_UNITS    = 64
EPOCHS        = 3
BATCH_SIZE    = 64
VAL_SPLIT     = 0.2


# Step 3 - Load the IMDB Dataset
def load_dataset(vocab_size: int):
    """Download and return the IMDB train/test split."""
    print("Loading IMDB dataset ...")
    (X_train, Y_train), (X_test, Y_test) = imdb.load_data(num_words=vocab_size)
    print("Dataset loaded successfully.")
    print(f"   Training reviews : {len(X_train):,}")
    print(f"   Testing  reviews : {len(X_test):,}")
    return (X_train, Y_train), (X_test, Y_test)


# Step 4 & 5 - Build Word Index and Reverse Word Index
def build_reverse_index() -> dict:
    """Fetch the IMDB word-to-integer mapping and invert it."""
    word_index    = imdb.get_word_index()
    reverse_index = {(idx + 3): word for word, idx in word_index.items()}
    return reverse_index


# Step 6 - Decode an Encoded Review Back to Plain Text
def decode_review(encoded_review: list, reverse_index: dict) -> str:
    """Convert a list of integers back into a readable sentence."""
    words = [reverse_index.get(idx, "?") for idx in encoded_review if idx >= 3]
    return " ".join(words)
