import streamlit as st
import pickle

# Load model and vectorizer
model = pickle.load(open("model.pkl", "rb"))
vectorizer = pickle.load(open("vectorizer.pkl", "rb"))

st.set_page_config(page_title="Fake News Detection", layout="centered")

st.title("📰 Fake News Detection App")
st.write("Enter a news article or headline to check whether it is **Real or Fake**.")

news_text = st.text_area("Enter news text here:", height=200)

if st.button("Predict"):
    if news_text.strip() == "":
        st.warning("Please enter some text.")
    else:
        text_vec = vectorizer.transform([news_text])
        prediction = model.predict(text_vec)

        if prediction[0] == 1:
            st.success("✅ This looks like REAL news")
        else:
            st.error("❌ This looks like FAKE news")
