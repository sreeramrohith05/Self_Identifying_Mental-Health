import streamlit as st
import pickle
from sklearn.feature_extraction.text import TfidfVectorizer
import nltk
from nltk.corpus import stopwords
import string
from nltk.stem.porter import PorterStemmer

# Load the TF-IDF vectorizer and the model
cv = pickle.load(open('vector.pkl', 'rb'))
model = pickle.load(open('model.pkl', 'rb'))

# Download NLTK resources
nltk.download('punkt')
nltk.download('stopwords')

# Initialize Porter Stemmer
ps = PorterStemmer()


# Function to preprocess text
def preprocess_text(text):
    # Tokenize the text
    tokens = nltk.word_tokenize(text)

    # Remove special characters and keep only alphanumeric tokens
    tokens = [token for token in tokens if token.isalnum()]

    # Remove stopwords and punctuation, and perform stemming
    tokens = [ps.stem(token) for token in tokens if
              token not in stopwords.words("english") and token not in string.punctuation]

    # Join the tokens back into a string
    processed_text = " ".join(tokens)

    return processed_text

st.title("Mental Health Detection")
# Input text field
input_text = st.text_input("Enter your message")

if st.button('Predict'):
    # Preprocess the input text
    preprocessed_text = preprocess_text(input_text)

    # Transform the preprocessed text using TF-IDF vectorizer
    vectorized_text = cv.transform([preprocessed_text])

    # Make prediction
    prediction = model.predict(vectorized_text)

    # Display prediction result
    st.header("Prediction:")
    if prediction[0] == 1:
        st.header("The message indicates a mental health issue.")
        st.write("Please consult the doctor")
    else:
        st.header("The message does not indicate a mental health issue.")
        st.write("You are very happy .. Keep Going...!!")



