import streamlit as st
import pandas as pd
import re
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.linear_model import LogisticRegression
import nltk
from nltk.corpus import stopwords

# Download required NLTK resources
nltk.download('stopwords')

# Preprocessing function
def preprocess_text(text):
    stop_words = set(stopwords.words('english'))
    text = re.sub(r'[^a-zA-Z\s]', '', text)  # Remove punctuation and special characters
    text = text.lower()  # Convert to lowercase
    words = text.split()
    words = [word for word in words if word not in stop_words]  # Remove stopwords
    return ' '.join(words)

# Load and preprocess the dataset
data_file_path = 'IMDB Dataset.csv'  # Adjust to your data path if needed
st.write("Loading dataset...")

try:
    data = pd.read_csv(data_file_path)
    data.columns = ['Review', 'Sentiment']
    data['Cleaned_Review'] = data['Review'].apply(preprocess_text)
    data['Sentiment'] = data['Sentiment'].map({'positive': 1, 'negative': 0})
except Exception as e:
    st.error("Dataset could not be loaded. Ensure the file path is correct.")
    st.stop()

# Vectorize the cleaned reviews
vectorizer = CountVectorizer(max_features=5000)
X = vectorizer.fit_transform(data['Cleaned_Review']).toarray()
y = data['Sentiment']

# Train Logistic Regression Model
st.write("Training model...")
#log_model = LogisticRegression()
log_model = LogisticRegression(max_iter=200, solver='liblinear')
log_model.fit(X, y)

st.title("IMDB Movie Review Sentiment Predictor")

# User input for new reviews
user_review = st.text_area("Enter a movie review for sentiment prediction:")

if st.button("Predict Sentiment"):
    if user_review:
        cleaned_review = preprocess_text(user_review)
        input_vectorized = vectorizer.transform([cleaned_review]).toarray()
        prediction = log_model.predict(input_vectorized)[0]
        sentiment = "Positive" if prediction == 1 else "Negative"
        st.success(f"Predicted Sentiment: {sentiment}")
    else:
        st.warning("Please enter a review for prediction.")
