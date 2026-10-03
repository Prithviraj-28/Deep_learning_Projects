# 🎬 Movie Review Sentiment Analysis — LSTM

A deep learning project that uses a **Long Short-Term Memory (LSTM)** neural network to classify IMDB movie reviews as **Positive** or **Negative**.

---

## 📌 Project Overview

| Item | Detail |
|---|---|
| **Dataset** | [IMDB Movie Reviews](https://ai.stanford.edu/~amaas/data/sentiment/) (via Keras) |
| **Task** | Binary Sentiment Classification |
| **Model** | Embedding → LSTM → Dense (Sigmoid) |
| **Framework** | TensorFlow / Keras |

### Model Pipeline

```
Raw Review  →  Embedding  →  LSTM  →  Dense (Sigmoid)  →  Positive / Negative
```

---

## 📂 Project Structure

```
marvellous lstm/
├── 1_LSTMloaddata.py      # Step-by-step data loading walkthrough
├── 2_LSTMDecode.py        # Word encoding / decoding exploration
├── 3_LSTModel.py          # Full LSTM model — train, evaluate, predict
├── requirements.txt       # Python dependencies
├── .gitignore             # Files excluded from version control
├── LICENSE                # MIT License
└── README.md              # This file
```

---

## ⚙️ Configuration

All hyper-parameters live at the top of `3_LSTModel.py` and are easy to tweak:

| Parameter | Default | Description |
|---|---|---|
| `VOCAB_SIZE` | 10,000 | Most-frequent words to keep |
| `MAX_LENGTH` | 200 | Max tokens per review (pad / truncate) |
| `EMBEDDING_DIM` | 32 | Size of each word-embedding vector |
| `LSTM_UNITS` | 64 | LSTM hidden-state size |
| `EPOCHS` | 3 | Training passes over the full dataset |
| `BATCH_SIZE` | 64 | Reviews per gradient-update step |
| `VAL_SPLIT` | 0.2 | Fraction of training data for validation |

---

## 🚀 Getting Started

### 1. Clone the repository

```bash
git clone https://github.com/your-username/marvellous-lstm.git
cd marvellous-lstm
```

### 2. Create and activate a virtual environment *(recommended)*

```bash
python -m venv .venv

# Windows
.venv\Scripts\activate

# macOS / Linux
source .venv/bin/activate
```

### 3. Install dependencies

```bash
pip install -r requirements.txt
```

### 4. Run the model

```bash
python 3_LSTModel.py
```

The script will:
1. Download the IMDB dataset automatically (first run only)
2. Display sample decoded reviews
3. Build and compile the LSTM model
4. Train for 3 epochs with 20 % validation split
5. Report test accuracy
6. Predict and compare sentiment on a test review

---

## 📊 Expected Output

```
============================================================
  Movie Review Sentiment Analysis — LSTM Model
============================================================
Loading IMDB dataset …
✔  Dataset loaded successfully.
   Training reviews : 25,000
   Testing  reviews : 25,000
...
📊  Test Results
   Loss     : 0.3142
   Accuracy : 86.52 %
...
  Actual Sentiment    : POSITIVE
  Predicted Sentiment : POSITIVE
  ✔  Correct!
```

> Accuracy will vary slightly between runs due to random weight initialization.

---

## 🧠 How It Works

1. **Embedding Layer** — Maps each word index to a 32-dimensional dense vector so the model can learn word meaning from context.
2. **LSTM Layer** — Processes the sequence of word vectors, capturing long-range dependencies (e.g., negation like "not bad").
3. **Dense + Sigmoid** — Outputs a probability between 0 and 1; ≥ 0.5 is classified as **Positive**, < 0.5 as **Negative**.

---

## 📋 Requirements

See [`requirements.txt`](requirements.txt) for the full list. Core packages:

- `tensorflow >= 2.13`
- `numpy`
- `keras` (bundled with TensorFlow)

---

## 📄 License

This project is licensed under the **MIT License** — see [`LICENSE`](LICENSE) for details.

---

## 👤 Author

**Marvellous**  
Deep Learning Practitioner

---

> 💡 *This project is designed for learning purposes. Feel free to experiment with the hyper-parameters, add dropout layers, or swap LSTM for GRU to compare performance.*
