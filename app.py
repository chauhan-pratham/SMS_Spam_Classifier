import streamlit as st
import pickle
import string
from nltk.corpus import stopwords
import nltk
from nltk.stem.porter import PorterStemmer

# Download NLTK data (only needs to be run once)
try:
    stopwords.words('english')
except LookupError:
    nltk.download('stopwords')
try:
    nltk.word_tokenize("test")
except LookupError:
    nltk.download('punkt')
    
ps = PorterStemmer()

def transform_text(text):
    text = text.lower()
    text = nltk.word_tokenize(text)

    y = []
    for i in text:
        if i.isalnum():
            y.append(i)

    text = y[:]
    y.clear()
    for i in text:
        if i not in stopwords.words('english') and i not in string.punctuation:
            y.append(i)

    text = y[:]
    y.clear()
    for i in text:
        y.append(ps.stem(i))

    return " ".join(y)

# Load the trained model and TF-IDF vectorizer
tfidf = pickle.load(open('vectorizer.pkl', 'rb'))
model = pickle.load(open('model.pkl', 'rb'))

st.set_page_config(page_title="SMS Spam Detector", layout="centered")

st.title("SMS Spam Detector")
st.markdown("Enter a message below to check if it's spam or not.")

input_sms = st.text_area("Enter the message")

if st.button("Predict"):
    if input_sms:
        # 1. Preprocess
        transformed_sms = transform_text(input_sms)

        # 2. Vectorize
        vector_input = tfidf.transform([transformed_sms]).toarray()

        # 3. Predict
        result = model.predict(vector_input)[0]

        # 4. Display
        st.subheader("Prediction:")
        if result == 1:
            st.error("This is a SPAM message.")
        else:
            st.success("This is NOT a SPAM message (Ham).")
    else:
        st.warning("Please enter a message to predict.")

st.sidebar.header("About")
st.sidebar.info(
    "This app uses a Multinomial Naive Bayes model trained on the "
    "SMS Spam Collection Dataset to classify messages as spam or ham."
)

st.sidebar.header("How it works:")
st.sidebar.markdown(
    """
    1. **Text Preprocessing:** Converts text to lowercase, tokenizes, removes special characters,
       stopwords, punctuation, and applies stemming.
    2. **Vectorization:** Transforms the preprocessed text into numerical features using TF-IDF.
    3. **Prediction:** The trained Multinomial Naive Bayes model predicts whether the message is spam or ham.
    """

)

