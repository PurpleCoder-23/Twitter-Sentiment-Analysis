# If running in local/Colab, you can install streamlit (not needed in actual app file)
#!pip install streamlit

import streamlit as st
import pickle
import re
from wordcloud import WordCloud
import matplotlib.pyplot as plt
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import PorterStemmer

# Download necessary NLTK resources
nltk.download('stopwords')
nltk.download('punkt')

# Initialize stemmer and stopwords
ps = PorterStemmer()
stop_words = set(stopwords.words('english'))

# Load the model and vectorizer
with open("LR_model.pkl", "rb") as model_file:
    model = pickle.load(model_file)

with open("vectorizer.pkl", "rb") as vectorizer_file:
   vectorizer = pickle.load(vectorizer_file)

# Preprocessing function
def stemming(content):
    stemmed_content = re.sub('[^a-zA-Z]', ' ', content)
    stemmed_content = stemmed_content.lower()
    stemmed_content = stemmed_content.split()
    stemmed_content = [ps.stem(word) for word in stemmed_content if word not in stop_words]
    return ' '.join(stemmed_content)

# Streamlit App UI
st.title("Twitter Sentiment Analysis App")
st.write("Enter text to analyze sentiment and view the WordCloud")

user_input = st.text_area("Enter your text here", height=120)

# Button to trigger prediction
if st.button("Analyze Sentiment"):
    if user_input.strip() == "":
        st.warning("Please enter some text before analyzing.")
    else:
        # Preprocess text
        preprocessed_text = stemming(user_input)

        # Vectorize text
        vectorized_text = vectorizer.transform([preprocessed_text])

        # Make prediction
        prediction = model.predict(vectorized_text)[0]
        confidence = model.predict_proba(vectorized_text).max()

        # Convert prediction to label
        sentiment_label = "Positive" if prediction == 1 else "Negative"

        st.subheader("Sentiment Prediction")
        st.write(f"Sentiment: **{sentiment_label}**")
        st.write(f"Confidence: **{confidence:.2f}**")

        # Generate and show WordCloud
        st.subheader("WordCloud")
        wc = WordCloud(background_color="white", width=500, height=400, max_words=30)

        # Plot wordcloud
        plt.figure(figsize=(10, 6))
        plt.imshow(wc.generate(preprocessed_text), interpolation='bilinear')
        plt.axis("off")
        st.pyplot(plt)
