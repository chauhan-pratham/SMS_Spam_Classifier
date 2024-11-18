import streamlit as st
import pickle
import string
from nltk.corpus import stopwords
import nltk
from nltk.stem.porter import PorterStemmer

# Download the punkt tokenizer and stopwords
nltk.download('punkt', quiet=True)
nltk.download('stopwords', quiet=True)

ps = PorterStemmer()

def transform_text(text):
    text = text.lower()  # Convert text to lowercase
    text = nltk.word_tokenize(text)  # Tokenize the text

    y = []
    for i in text:
        if i.isalnum():  # Keep only alphanumeric characters
            y.append(i)

    text = y[:]  # Copy the filtered list
    y.clear()  # Clear the temporary list

    for i in text:
        if i not in stopwords.words('english') and i not in string.punctuation:
            y.append(i)  # Remove stopwords and punctuation

    text = y[:]  # Copy the filtered list again
    y.clear()  # Clear the temporary list

    for i in text:
        y.append(ps.stem(i))  # Stem the words

    return " ".join(y)  # Join the words back into a single string

# Load the model and vectorizer
tfidf = pickle.load(open('vectorizer.pkl', 'rb'))
model = pickle.load(open('model.pkl', 'rb'))

st.title("Email/SMS Spam Classifier")  # Set the title of the app

input_sms = st.text_area("Enter the message")  # Input area for the user

if st.button('Predict'):
    # 1. Preprocess the input SMS
    transformed_sms = transform_text(input_sms)
    # 2. Vectorize the preprocessed SMS
    vector_input = tfidf.transform([transformed_sms])
    # 3. Predict using the loaded model
    result = model.predict(vector_input)[0]
    # 4. Display the result
    if result == 1:
        st.header("Spam")  # If the result is 1, it's spam
    else:
        st.header("Not Spam")  # Otherwise, it's not spam
