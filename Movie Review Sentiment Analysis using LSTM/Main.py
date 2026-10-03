"""
Movie Review Sentiment Analysis using LSTM
==========================================
Trains a Long Short-Term Memory (LSTM) neural network on the IMDB
movie-reviews dataset to classify reviews as POSITIVE or NEGATIVE.

Model Pipeline:
    Raw Review → Embedding → LSTM → Dense (Sigmoid) → Positive / Negative

Author  : Marvellous
License : MIT
"""

# ─────────────────────────────────────────────────────────────────────────────
# Step 1 · Import Required Libraries
# ─────────────────────────────────────────────────────────────────────────────
from tensorflow.keras.datasets import imdb
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Embedding, LSTM, Dense
from tensorflow.keras.preprocessing.sequence import pad_sequences


# ─────────────────────────────────────────────────────────────────────────────
# Step 2 · Hyper-parameters / Configuration
# ─────────────────────────────────────────────────────────────────────────────
VOCAB_SIZE    = 10_000  # Keep the 10,000 most-frequent words
MAX_LENGTH    = 200     # Truncate / pad every review to 200 tokens
EMBEDDING_DIM = 32      # Each word is represented by 32 numbers
LSTM_UNITS    = 64      # Size of the LSTM hidden state
EPOCHS        = 3       # Number of full passes through the training data
BATCH_SIZE    = 64      # Reviews processed per gradient-update step
VAL_SPLIT     = 0.2     # Fraction of training data used for validation


# ─────────────────────────────────────────────────────────────────────────────
# Step 3 · Load the IMDB Dataset
# ─────────────────────────────────────────────────────────────────────────────
def load_dataset(vocab_size: int):
    """
    Download and return the IMDB train / test split.

    Args:
        vocab_size (int): Number of most-frequent words to keep.

    Returns:
        tuple: (X_train, Y_train), (X_test, Y_test)
            X_* — Lists of integer-encoded reviews.
            Y_* — Binary sentiment labels (0 = Negative, 1 = Positive).
    """
    print("Loading IMDB dataset …")
    (X_train, Y_train), (X_test, Y_test) = imdb.load_data(num_words=vocab_size)
    print("✔  Dataset loaded successfully.")
    print(f"   Training reviews : {len(X_train):,}")
    print(f"   Testing  reviews : {len(X_test):,}")
    return (X_train, Y_train), (X_test, Y_test)


# ─────────────────────────────────────────────────────────────────────────────
# Step 4 & 5 · Build Word Index and Reverse Word Index
# ─────────────────────────────────────────────────────────────────────────────
def build_reverse_index() -> dict:
    """
    Fetch the IMDB word-to-integer mapping and invert it.

    IMDB reserves indices 0-2 for padding / start / unknown tokens,
    so every real word index is offset by +3 when the data is loaded.

    Returns:
        dict: Maps integer index → word string.

    Example:
        "Drishyam is a good Movie" might encode as (20, 56, 78, 43)
        reverse_index[20] → 'drishyam'
    """
    word_index    = imdb.get_word_index()
    reverse_index = {(idx + 3): word for word, idx in word_index.items()}
    return reverse_index


# ─────────────────────────────────────────────────────────────────────────────
# Step 6 · Decode an Encoded Review Back to Plain Text
# ─────────────────────────────────────────────────────────────────────────────
def decode_review(encoded_review: list, reverse_index: dict) -> str:
    """
    Convert a list of integers back into a readable sentence.

    Args:
        encoded_review (list[int]): Integer-encoded movie review.
        reverse_index  (dict)     : Mapping from integer → word.

    Returns:
        str: Human-readable review text.
    """
    # Indices 0, 1, 2 are reserved tokens — skip them
    words = [reverse_index.get(idx, "?") for idx in encoded_review if idx >= 3]
    return " ".join(words)


# ─────────────────────────────────────────────────────────────────────────────
# Step 7 · Display Sample Reviews
# ─────────────────────────────────────────────────────────────────────────────
def display_sample_reviews(X_train, Y_train, reverse_index: dict,
                            start: int = 3, end: int = 7) -> None:
    """
    Print a few decoded training reviews with their true sentiment.

    Args:
        X_train       : Encoded training reviews.
        Y_train       : Training sentiment labels.
        reverse_index : Integer-to-word mapping.
        start (int)   : First review index to display (inclusive).
        end   (int)   : Last  review index to display (exclusive).
    """
    print("\n" + "─" * 60)
    print("  SAMPLE REVIEWS")
    print("─" * 60)

    for i in range(start, end):
        sentiment   = "Positive ✔" if Y_train[i] == 1 else "Negative ✘"
        review_text = decode_review(X_train[i], reverse_index)

        print(f"\n  Review #{i + 1}")
        print(f"  Sentiment : {sentiment}")
        print(f"  Text      : {review_text[:150]} …")
        print("  " + "·" * 56)


# ─────────────────────────────────────────────────────────────────────────────
# Step 8 · Pad Sequences to a Fixed Length
# ─────────────────────────────────────────────────────────────────────────────
def pad_data(X_train, X_test, max_length: int):
    """
    Pad (or truncate) all reviews to the same length so they can be
    stacked into a matrix for batch training.

    Args:
        X_train    : Raw training review sequences.
        X_test     : Raw testing  review sequences.
        max_length : Target sequence length.

    Returns:
        tuple: (X_train_padded, X_test_padded) as NumPy arrays.
    """
    X_train_padded = pad_sequences(X_train, maxlen=max_length)
    X_test_padded  = pad_sequences(X_test,  maxlen=max_length)

    print(f"\n✔  Padding complete.")
    print(f"   Training shape : {X_train_padded.shape}")
    print(f"   Testing  shape : {X_test_padded.shape}")

    return X_train_padded, X_test_padded


# ─────────────────────────────────────────────────────────────────────────────
# Step 9 · Build the LSTM Model
# ─────────────────────────────────────────────────────────────────────────────
def build_model(vocab_size: int, embedding_dim: int, lstm_units: int) -> Sequential:
    """
    Construct and return the LSTM sentiment-classification model.

    Architecture:
        Embedding → LSTM → Dense (sigmoid)

    Args:
        vocab_size    (int): Number of unique words (vocabulary size).
        embedding_dim (int): Size of each word-embedding vector.
        lstm_units    (int): Number of hidden units in the LSTM layer.

    Returns:
        keras.Sequential: Un-compiled model.
    """
    model = Sequential([
        # Embedding: maps each word-index to a dense vector
        Embedding(input_dim=vocab_size, output_dim=embedding_dim),

        # LSTM: captures sequential / contextual patterns in the review
        LSTM(units=lstm_units),

        # Output: single neuron with sigmoid → probability of Positive sentiment
        Dense(units=1, activation="sigmoid"),
    ])

    return model


# ─────────────────────────────────────────────────────────────────────────────
# Step 10 · Compile the Model
# ─────────────────────────────────────────────────────────────────────────────
def compile_model(model: Sequential) -> None:
    """
    Compile the model with Adam optimizer and binary cross-entropy loss.

    Args:
        model (Sequential): Keras model to compile (modified in-place).
    """
    model.compile(
        optimizer="adam",             # Adaptive gradient-descent algorithm
        loss="binary_crossentropy",   # Standard loss for binary classification
        metrics=["accuracy"],         # Track accuracy during training
    )
    print("\n✔  Model compiled successfully.")
    model.summary()


# ─────────────────────────────────────────────────────────────────────────────
# Step 11 · Train the Model
# ─────────────────────────────────────────────────────────────────────────────
def train_model(model: Sequential, X_train, Y_train,
                epochs: int, batch_size: int, val_split: float) -> None:
    """
    Fit the model on the training data.

    Args:
        model      (Sequential): Compiled Keras model.
        X_train    : Padded training sequences.
        Y_train    : Training sentiment labels.
        epochs     (int)  : Number of training epochs.
        batch_size (int)  : Batch size per gradient update.
        val_split  (float): Fraction of training data used for validation.
    """
    print("\n⏳  Training model …")
    model.fit(
        X_train, Y_train,
        epochs=epochs,
        batch_size=batch_size,
        validation_split=val_split,
    )
    print("✔  Training complete.")


# ─────────────────────────────────────────────────────────────────────────────
# Step 12 · Evaluate the Model on Test Data
# ─────────────────────────────────────────────────────────────────────────────
def evaluate_model(model: Sequential, X_test, Y_test) -> None:
    """
    Evaluate and print the model's performance on the held-out test set.

    Args:
        model  (Sequential): Trained Keras model.
        X_test : Padded test sequences.
        Y_test : True test labels.
    """
    loss, accuracy = model.evaluate(X_test, Y_test, verbose=0)
    print(f"\n📊  Test Results")
    print(f"   Loss     : {loss:.4f}")
    print(f"   Accuracy : {accuracy * 100:.2f} %")


# ─────────────────────────────────────────────────────────────────────────────
# Steps 13-15 · Predict Sentiment for a Single Review
# ─────────────────────────────────────────────────────────────────────────────
def predict_review(model: Sequential, X_test, X_test_padded,
                   Y_test, reverse_index: dict, review_index: int = 0) -> None:
    """
    Decode, predict, and compare the sentiment of one test review.

    Args:
        model          (Sequential): Trained Keras model.
        X_test         : Raw (unpadded) test sequences.
        X_test_padded  : Padded test sequences fed to the model.
        Y_test         : True test labels.
        reverse_index  (dict): Integer-to-word mapping.
        review_index   (int) : Which test review to predict (default 0).
    """
    # Decode the raw text
    decoded_text = decode_review(X_test[review_index], reverse_index)

    # True sentiment
    actual_sentiment = "POSITIVE" if Y_test[review_index] == 1 else "NEGATIVE"

    # Model prediction — shape must be (1, MAX_LENGTH)
    sample      = X_test_padded[review_index : review_index + 1]
    probability = model.predict(sample, verbose=0)[0][0]
    predicted_sentiment = "POSITIVE" if probability >= 0.5 else "NEGATIVE"

    # Display results
    print("\n" + "═" * 60)
    print("  PREDICTION RESULT")
    print("═" * 60)
    print(f"\n  Review Text         : {decoded_text[:200]} …")
    print(f"  Confidence          : {probability:.4f}")
    print(f"  Actual Sentiment    : {actual_sentiment}")
    print(f"  Predicted Sentiment : {predicted_sentiment}")

    verdict = "✔  Correct!" if actual_sentiment == predicted_sentiment else "✘  Incorrect"
    print(f"\n  {verdict}")
    print("═" * 60)


# ─────────────────────────────────────────────────────────────────────────────
# Main Entry Point
# ─────────────────────────────────────────────────────────────────────────────
def main():
    print("=" * 60)
    print("  Movie Review Sentiment Analysis — LSTM Model")
    print("=" * 60)

    # 1. Load data
    (X_train, Y_train), (X_test, Y_test) = load_dataset(VOCAB_SIZE)

    # 2. Build vocabulary lookup
    reverse_index = build_reverse_index()

    # 3. Show sample reviews
    display_sample_reviews(X_train, Y_train, reverse_index)

    # 4. Pad sequences to uniform length
    X_train_padded, X_test_padded = pad_data(X_train, X_test, MAX_LENGTH)

    # 5. Build, compile, and train the model
    model = build_model(VOCAB_SIZE, EMBEDDING_DIM, LSTM_UNITS)
    compile_model(model)
    train_model(model, X_train_padded, Y_train, EPOCHS, BATCH_SIZE, VAL_SPLIT)

    # 6. Evaluate on the test set
    evaluate_model(model, X_test_padded, Y_test)

    # 7. Predict a single review
    predict_review(model, X_test, X_test_padded, Y_test, reverse_index, review_index=0)


if __name__ == "__main__":
    main()