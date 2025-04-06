import re
import joblib
import pandas as pd
import numpy as np
from typing import List, Tuple, Any
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline

def preprocess_text(text: str) -> str:
    text = text.lower()
    text = re.sub(r'[^a-z\s]', '', text)  
    return text

def train_sentiment_model(training_data: List[Tuple[str, str]]) -> Any:
    texts, labels = zip(*training_data)  
    texts = [preprocess_text(text) for text in texts]

    X_train, X_test, y_train, y_test = train_test_split(texts, labels, test_size=0.2, random_state=42)

    model = Pipeline([
        ('vectorizer', CountVectorizer()),
        ('classifier', LogisticRegression())
    ])

    model.fit(X_train, y_train)

    joblib.dump(model, "sentiment_model.pkl")  

    return model

def predict_sentiment(model: Any, new_text: str) -> str:
    new_text = preprocess_text(new_text)
    return model.predict([new_text])[0]

if __name__ == "__main__":
    print("Loading dataset and training model...")

    file_path = "AirlineReviews.csv"
    df = pd.read_csv(file_path)

    df['Sentiment'] = df['OverallScore'].apply(lambda x: 'positive' if x >= 5 else 'negative')
    df = df[['Review', 'Sentiment']].dropna()

    training_data = list(df.itertuples(index=False, name=None))  
    model = train_sentiment_model(training_data)

    print("Model trained and saved as sentiment_model.pkl")

    while True:
        user_input = input("\nEnter feedback (or type 'exit' to quit): ")
        if user_input.lower() == "exit":
            break
        prediction = predict_sentiment(model, user_input)
        print(f"Predicted Sentiment: {prediction}")
