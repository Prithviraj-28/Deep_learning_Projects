"""
Movie Review Sentiment Analysis using LSTM
==========================================
Author  : Marvellous
License : MIT
"""

from tensorflow.keras.datasets import imdb
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Embedding, LSTM, Dense
from tensorflow.keras.preprocessing.sequence import pad_sequences

VOCAB_SIZE    = 10_000
MAX_LENGTH    = 200
EMBEDDING_DIM = 32
LSTM_UNITS    = 64
EPOCHS        = 3
BATCH_SIZE    = 64
VAL_SPLIT     = 0.2


def load_dataset(vocab_size: int):
    """Download and return the IMDB train/test split."""
    print("Loading IMDB dataset ...")
    (X_train, Y_train), (X_test, Y_test) = imdb.load_data(num_words=vocab_size)
    print("Dataset loaded successfully.")
    print(f"   Training reviews : {len(X_train):,}")
    print(f"   Testing  reviews : {len(X_test):,}")
    return (X_train, Y_train), (X_test, Y_test)


def build_reverse_index() -> dict:
    """Fetch the IMDB word-to-integer mapping and invert it."""
    word_index    = imdb.get_word_index()
    reverse_index = {(idx + 3): word for word, idx in word_index.items()}
    return reverse_index


def decode_review(encoded_review: list, reverse_index: dict) -> str:
    """Convert a list of integers back into a readable sentence."""
    words = [reverse_index.get(idx, "?") for idx in encoded_review if idx >= 3]
    return " ".join(words)


def display_sample_reviews(X_train, Y_train, reverse_index: dict,
                            start: int = 3, end: int = 7) -> None:
    """Print decoded training reviews with their true sentiment."""
    print("\n" + "-" * 60)
    print("  SAMPLE REVIEWS")
    print("-" * 60)
    for i in range(start, end):
        sentiment   = "Positive" if Y_train[i] == 1 else "Negative"
        review_text = decode_review(X_train[i], reverse_index)
        print(f"\n  Review #{i + 1}  |  Sentiment: {sentiment}")
        print(f"  {review_text[:150]} ...")
        print("  " + "." * 56)


def pad_data(X_train, X_test, max_length: int):
    """Pad or truncate all reviews to the same fixed length."""
    X_train_padded = pad_sequences(X_train, maxlen=max_length)
    X_test_padded  = pad_sequences(X_test,  maxlen=max_length)
    print(f"\nPadding complete.")
    print(f"   Training shape : {X_train_padded.shape}")
    print(f"   Testing  shape : {X_test_padded.shape}")
    return X_train_padded, X_test_padded


# Step 9 - Build the LSTM Model
def build_model(vocab_size: int, embedding_dim: int, lstm_units: int) -> Sequential:
    """
    Construct the LSTM sentiment-classification model.
    Architecture: Embedding -> LSTM -> Dense (sigmoid)
    """
    model = Sequential([
        # Maps each word-index to a dense embedding vector
        Embedding(input_dim=vocab_size, output_dim=embedding_dim),
        # Captures sequential context across the review
        LSTM(units=lstm_units),
        # Single sigmoid output: probability of Positive sentiment
        Dense(units=1, activation="sigmoid"),
    ])
    return model


# Step 10 - Compile the Model
def compile_model(model: Sequential) -> None:
    """Compile with Adam optimizer and binary cross-entropy loss."""
    model.compile(
        optimizer="adam",
        loss="binary_crossentropy",
        metrics=["accuracy"],
    )
    print("\nModel compiled successfully.")
    model.summary()
