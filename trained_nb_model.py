import sqlite3
import re
import string
import numpy as np
import requests
import os
from dotenv import load_dotenv
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
from sklearn.naive_bayes import MultinomialNB

load_dotenv()
hf_api_key = os.getenv("HUGGINGFACE_API_KEY")

HF_API_URL = "https://api-inference.huggingface.co/models/facebook/bart-large-cnn"
HEADERS = {"Authorization": f"Bearer {hf_api_key}"}

def preprocess_text(text):
    text = text.lower()
    text = re.sub(r'\d+', '', text)
    text = text.translate(str.maketrans('', '', string.punctuation))
    text = text.strip()
    return text

def load_legal_data(db_path="legal_data.db"):
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()
    cursor.execute("SELECT question, response FROM legal_faqs")  
    data = cursor.fetchall()
    conn.close()
    return zip(*data) if data else ([], [])

def train_classifier():
    questions, responses = load_legal_data()
    if not questions:
        return None, None, None

    questions = [preprocess_text(q) for q in questions]
    vectorizer = CountVectorizer()
    X = vectorizer.fit_transform(questions)
    tfidf_transformer = TfidfTransformer()
    X_tfidf = tfidf_transformer.fit_transform(X)

    clf = MultinomialNB()
    clf.fit(X_tfidf, responses)

    return clf, vectorizer, tfidf_transformer

def get_legal_response(user_query, clf, vectorizer, tfidf_transformer, threshold=0.7):
    if not clf:
        return "No trained model found."

    processed_query = preprocess_text(user_query)
    query_vectorized = vectorizer.transform([processed_query])
    query_tfidf = tfidf_transformer.transform(query_vectorized)

    predicted_probs = np.max(clf.predict_proba(query_tfidf))
    predicted_response = clf.predict(query_tfidf)[0]

    if predicted_probs >= threshold:
        return predicted_response
    else:
        return get_huggingface_response(user_query)

def get_huggingface_response(user_query):
    payload = {"inputs": user_query}
    response = requests.post(HF_API_URL, headers=HEADERS, json=payload)
    
    if response.status_code == 200:
        return response.json()[0]["generated_text"]
    else:
        return "Sorry, I couldn't process your request at the moment."

clf, vectorizer, tfidf_transformer = train_classifier()

# Example Usage
while True:
    user_input = input("Ask a legal question: ")
    if user_input.lower() == "exit":
        break
    response = get_legal_response(user_input, clf, vectorizer, tfidf_transformer)
    print("\nLegal Advisor:", response, "\n")
