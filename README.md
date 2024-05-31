# Self_Identifying_Mental-Health

# Self-Identifying Mental Health Project

## Overview

This project aims to create a tool for identifying potential mental health issues from text data, specifically focusing on identifying messages that may indicate a suicide risk. Using natural language processing (NLP) and machine learning (ML) techniques, the project preprocesses text data, vectorizes it, and trains models to classify the text as either indicative of a mental health issue or not.

## Features

- **Text Preprocessing**: Cleans and processes text data by removing punctuation, stopwords, and emojis, and applies stemming.
- **TF-IDF Vectorization**: Converts processed text data into numerical features using TF-IDF.
- **Machine Learning Models**: Implements Logistic Regression and K-Nearest Neighbors (KNN) models for text classification.
- **Streamlit Integration**: Provides an interactive web interface for users to input text and receive predictions.

## Technologies Used

- **Python**: The primary programming language used.
- **NLTK**: For text preprocessing, including tokenization, stopword removal, and stemming.
- **scikit-learn**: For vectorization and machine learning model implementation.
- **Streamlit**: For creating an interactive web application.
- **pandas**: For data manipulation and analysis.

## Installation

1. **Clone the Repository**:
    - Clone the repository from GitHub and navigate to the project directory.

2. **Install Required Packages**:
    - Install the necessary Python packages using pip.

3. **Download NLTK Data**:
    - Download required NLTK data packages for text preprocessing.

## Usage

1. **Load and Preprocess Data**:
    - Load the dataset and ensure labels are converted to numerical values.
    - Preprocess the text by removing punctuation, stopwords, and emojis, and apply stemming.

2. **Vectorize Text Data**:
    - Convert the processed text data into numerical features using TF-IDF.

3. **Split Data and Train Models**:
    - Split the data into training and testing sets.
    - Train both Logistic Regression and K-Nearest Neighbors (KNN) models.

4. **Evaluate Models**:
    - Evaluate the performance of both models using accuracy and classification reports.

5. **Run Streamlit App**:
    - Use Streamlit to create an interactive web application for predictions.
    - Allow users to input text and receive predictions from both models.

## Challenges Faced

- **Data Quality**: Ensuring a large and accurately labeled dataset.
- **Preprocessing Complexity**: Handling various text preprocessing tasks.
- **Model Performance**: Balancing accuracy and computational efficiency.
- **User Interface**: Creating an intuitive UI with Streamlit.

## Conclusion

This project demonstrates the application of NLP and ML to identify potential mental health issues from text data. By preprocessing text, vectorizing it, and using ML models for classification, the tool aims to provide early warnings for those at risk, encouraging timely intervention and support.
