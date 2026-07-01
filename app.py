import pickle
import numpy as np
import streamlit as st
from tensorflow import keras
from tensorflow.keras.preprocessing.sequence import pad_sequences

# ---------- Page setup ----------
st.set_page_config(page_title="Next-Word Predictor", page_icon="✍️", layout="centered")

# ---------- Cached loaders ----------
@st.cache_resource
def load_artifacts():
    model = keras.models.load_model("lstm_model.h5")
    with open("tokenizer.pkl", "rb") as f:
        tokenizer = pickle.load(f)
    with open("max_len.pkl", "rb") as f:
        max_len = pickle.load(f)
    index_word = {idx: word for word, idx in tokenizer.word_index.items()}
    return model, tokenizer, max_len, index_word


model, tokenizer, max_len, index_word = load_artifacts()


# ---------- Generation logic ----------
def sample_with_temperature(preds, temperature):
    if temperature <= 0:
        return int(np.argmax(preds))
    preds = np.asarray(preds).astype("float64")
    preds = np.log(preds + 1e-9) / temperature
    exp_preds = np.exp(preds)
    preds = exp_preds / np.sum(exp_preds)
    probas = np.random.multinomial(1, preds, 1)
    return int(np.argmax(probas))


def generate_text(seed_text, num_words, temperature):
    text = seed_text
    generated_words = []
    for _ in range(num_words):
        seq = tokenizer.texts_to_sequences([text])[0]
        seq = pad_sequences([seq], maxlen=max_len, padding="pre")
        preds = model.predict(seq, verbose=0)[0]
        next_idx = sample_with_temperature(preds, temperature)
        next_word = index_word.get(next_idx, "")
        if not next_word:
            break
        text += " " + next_word
        generated_words.append(next_word)
    return text, generated_words


# ---------- UI ----------
st.title("✍️ Next-Word Predictor")
st.caption("LSTM text-generation model · trained vocab size: "
           f"{len(tokenizer.word_index):,} · max sequence length: {max_len}")

seed_text = st.text_area(
    "Enter a starting phrase",
    value="i am feeling",
    height=100,
    placeholder="Type a few words to start...",
)

col1, col2 = st.columns(2)
with col1:
    num_words = st.slider("Words to generate", min_value=1, max_value=50, value=10)
with col2:
    temperature = st.slider(
        "Creativity (temperature)",
        min_value=0.0, max_value=1.5, value=0.0, step=0.1,
        help="0 = always pick the most likely word (deterministic, can repeat/loop). "
             "Higher values sample more freely and add variety.",
    )

generate_clicked = st.button("Generate", type="primary", use_container_width=True)

if generate_clicked:
    if not seed_text.strip():
        st.warning("Please enter some starting text.")
    else:
        with st.spinner("Generating..."):
            full_text, words = generate_text(seed_text.strip(), num_words, temperature)
        st.subheader("Result")
        st.write(full_text)
        st.caption("Generated words: " + ", ".join(words) if words else "No words generated.")

with st.expander("About this app"):
    st.markdown(
        """
This app loads a pre-trained **LSTM next-word prediction** model
(`lstm_model.h5`) along with its matching **tokenizer** (`tokenizer.pkl`)
and **max sequence length** (`max_len.pkl`).

Given a starting phrase, it repeatedly:
1. Converts the current text to token IDs using the tokenizer
2. Pads the sequence to the model's expected input length
3. Predicts a probability distribution over the vocabulary for the next word
4. Picks the next word (greedily at temperature 0, or by sampling at higher temperatures)
5. Appends it and repeats

**Note:** if generated text starts looping or feels repetitive, that's a
property of the underlying model/training data — try raising the
temperature slider for more varied output.
        """
    )