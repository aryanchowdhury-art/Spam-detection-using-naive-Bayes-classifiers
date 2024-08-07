# Spam-detection-using-naive-Bayes-classifiers
Naive Bayes classifiers are commonly used for spam detection because of their simplicity, effectiveness, and efficiency. Here’s an overview of how you can implement a spam detection system using a Naive Bayes classifier in Python.
Steps to Implement Spam Detection
Collect and Prepare Data: You need a labeled dataset where emails are marked as spam or ham (not spam).

Preprocess the Data: This involves tokenizing the text, converting it to lowercase, removing stop words, and transforming the text into a numerical format the model can understand.

Feature Extraction: Convert text data into numerical data using techniques like Bag of Words or TF-IDF.

Train the Naive Bayes Classifier: Use the preprocessed and feature-extracted data to train the model.

Evaluate the Model: Assess the model's performance using accuracy, precision, recall, and F1-score metrics.

Make Predictions: Use the trained model to predict whether new emails are spam.

Here’s a basic implementation using Python and the sci-kit-learn library:


1. Import Libraries

import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score




2. Load Data
Assume you have a CSV file spam.csv with columns label (spam/ham) and message (the email content).


data = pd.read_csv('spam.csv', encoding='latin-1')
data = data[['label', 'message']]
data['label'] = data['label'].map({'ham': 0, 'spam': 1})


3. Preprocess Data

vectorizer = CountVectorizer(stop_words='english')
X = vectorizer.fit_transform(data['message'])
y = data['label']





4. Split Data

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)




5. Train Naive Bayes Classifier

model = MultinomialNB()
model.fit(X_train, y_train)




6. Evaluate the Model

y_pred = model.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
precision = precision_score(y_test, y_pred)
recall = recall_score(y_test, y_pred)
f1 = f1_score(y_test, y_pred)

print(f'Accuracy: {accuracy}')
print(f'Precision: {precision}')
print(f'Recall: {recall}')
print(f'F1 Score: {f1}')



7. Make Predictions

def predict_spam(message):
    message_transformed = vectorizer.transform([message])
    prediction = model.predict(message_transformed)
    return 'Spam' if prediction[0] == 1 else 'Ham'

print(predict_spam("Free entry in 2 a weekly competition to win FA Cup final tickets"))
