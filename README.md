 # LSTM Next-Word Predictor

A Streamlit web app that predicts and generates the next word(s) in a sentence using a trained LSTM (Recurrent Neural Network) language model.

Type a starting phrase, and the model predicts what comes next — one word at a time — using an embedding layer, an LSTM, and a softmax output over the vocabulary.

## Features

- 🔮 **Next-word generation** — generate up to 50 words from any seed phrase
- 🎛️ **Adjustable creativity** — a temperature slider controls how deterministic vs. varied the output is
- ⚡ **Cached model loading** — fast repeat predictions via Streamlit's resource caching
- 🧠 **Simple, inspectable pipeline** — tokenizer → padding → model → softmax → next word

## Demo
Input:  "i am feeling"

Output: "i am feeling really good about this decision"
## Tech Stack

| Component        | Tool                          |
|-------------------|-------------------------------|
| Frontend           | [Streamlit](https://streamlit.io) |
| Model              | LSTM (Keras / TensorFlow)     |
| Tokenization       | Keras `Tokenizer`              |
| Language           | Python 3.11                    |

## Model Details

- **Architecture:** Embedding → LSTM(128) → Dense(softmax)
- **Vocabulary size:** 10,000 words
- **Max input sequence length:** 745 tokens
- **Output:** probability distribution over the vocabulary, decoded to the most likely (or sampled) next word

## Project Structure
.

├── app.py              # Streamlit app

├── lstm_model.h5        # Trained Keras LSTM model

├── tokenizer.pkl         # Fitted Keras Tokenizer

├── max_len.pkl           # Max sequence length used during training

└── requirements.txt      # Python dependencies

## Getting Started

### Prerequisites

- Python **3.11** (TensorFlow does not yet support 3.13+/3.14 — using a newer Python version will cause `ModuleNotFoundError: No module named 'tensorflow'`)

### Installation

```bash
# Clone the repo
git clone https://github.com/f20221354/lstm-next-word-predictor.git
cd lstm-next-word-predictor

# Create and activate a virtual environment (use Python 3.11 specifically)
python3.11 -m venv venv
source venv/bin/activate      # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

### Run the app

```bash
streamlit run app.py
```

The app will open automatically at `http://localhost:8501`.

## Usage

1. Enter a starting phrase in the text box
2. Choose how many words to generate
3. Adjust the temperature slider:
   - **0.0** — always picks the most likely next word (deterministic, can repeat/loop)
   - **0.5–1.0** — samples more freely for varied, creative output
4. Click **Generate**

## Known Limitations

- As a word-level LSTM trained on a fixed vocabulary, it cannot predict out-of-vocabulary words
- At low temperature, output can fall into repetitive loops — this is expected model behavior, not an app bug
- Prediction quality depends entirely on the size and quality of the original training corpus

## License

This project is licensed under the MIT License.
