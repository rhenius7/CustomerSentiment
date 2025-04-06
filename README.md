# Binary Classification for Customer Sentiment  

## 📌 Problem Overview  
This project trains a sentiment classification model for airline customer feedback. It classifies reviews as **positive** or **negative** based on the given dataset. The sentiment is determined using the `OverallScore` column:  
- **Positive Sentiment:** If `OverallScore` is **5 or greater**  
- **Negative Sentiment:** If `OverallScore` is **less than 5**  

The model is trained using **Logistic Regression** and uses **text preprocessing** to clean and tokenize the reviews.

---

##  Installation and Running 

1. **Navigate to the project folder in VS Code:**  
   ```sh
   cd submission/problem2

2. **Install depecdencies:**
    pip install -r requirements.txt

3.**Run the project**
    python main.py


**Approach & Multi-Agent Consideration**

Model Training Approach
**Preprocessing:** Converts text to lowercase, removes special characters.

**Vectorization:** Uses CountVectorizer to transform text into numerical features.

**Classification:** Uses LogisticRegression to train a binary classifier.


**Function Responsibilities**

**train_sentiment_model(training_data: List[Tuple[str, str]]):**

Takes customer reviews as input.

Trains the model and saves it as sentiment_model.pkl.

**predict_sentiment(model: Any, new_text: str):**

Accepts the trained model and predicts sentiment for new input.

This approach ensures efficient classification of customer feedback, helping airlines analyze passenger sentiment effectively. 