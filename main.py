import streamlit as st
import pickle
from sklearn.feature_extraction.text import TfidfVectorizer
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
import string
from nltk.stem.porter import PorterStemmer


cv = pickle.load(open('vector.pkl', 'rb'))
model = pickle.load(open('model.pkl', 'rb'))


nltk.download('punkt')
nltk.download('stopwords')


# Initialize necessary tools
stop_words = set(stopwords.words('english'))
stemmer = PorterStemmer()

# Define preprocessing function
def preprocess_text(text):
    # Convert text to lowercase
    text = text.lower()
    
    # Remove punctuation
    text = text.translate(str.maketrans('', '', string.punctuation))
    
    # Remove emojis
   # text = emoji.replace_emoji(text, replace='')
    
    # Tokenize text
    tokens = word_tokenize(text)
    
    # Remove stopwords
    tokens = [word for word in tokens if word not in stop_words]
    
    # Stemming
    tokens = [stemmer.stem(word) for word in tokens]
    
    return ' '.join(tokens)

st.title("SIMH - Self identifying the mental health")
st.write("Here you can discover your inner health")

input_text = st.text_input("Enter your message")

if st.button('Predict'):
    
    preprocessed_text = preprocess_text(input_text)

    vectorized_text = cv.transform([preprocessed_text])

    prediction = model.predict(vectorized_text)

    st.header("Prediction:")
    if prediction[0] == 1:
        st.header("The message indicates a mental health issue.")
        st.write("Please consult the doctor")
    else:
        st.header("The message does not indicate a mental health issue.")
        st.write("You are very happy .. Keep Going...!!")



