import streamlit as st
import numpy as np
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.sequence import pad_sequences
import pickle


model = load_model("story_generator.h5")

with open("tokenizer.pkl", "rb") as f:
    tokenizer = pickle.load(f)


max_sequence_len = 30  

# Story generation logic
def generate_story(seed_text, next_words=50):
    for _ in range(next_words):
        token_list = tokenizer.texts_to_sequences([seed_text])[0]
        token_list = pad_sequences([token_list], maxlen=max_sequence_len - 1, padding='pre')
        predicted = model.predict(token_list, verbose=0)
        predicted_index = np.argmax(predicted)

        output_word = ''
        for word, index in tokenizer.word_index.items():
            if index == predicted_index:
                output_word = word
                break

        if output_word == '':
            break

        seed_text += ' ' + output_word
    return seed_text

# Streamlit UI
st.title("🦋Fairy Tales Generator Using LSTM")
st.markdown("Enter a **starting phrase**, and let the AI continue the story!")
st.markdown("This app uses a pre-trained LSTM model that was trained on a dataset of fairy tales.")

st.markdown("######  Press Enter after typing your starting line and then click 'Generate' to see the AI's continuation of your story.")
user_input = st.text_input("Your starting line:", "A prince")

word_count = st.slider("Number of words to generate", 10, 30, 15)

if st.button("Generate"):
    story = generate_story(user_input, word_count)
    st.markdown("### ✨ Generated Story:")
    st.write(story)
