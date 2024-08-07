import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score


data = pd.read_csv('spam.csv', encoding='latin-1')
data = data[['label', 'message']]
data['label'] = data['label'].map({'ham': 0, 'spam': 1})

vectorizer = CountVectorizer(stop_words='english')
X = vectorizer.fit_transform(data['message'])
y = data['label']


X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)


model = MultinomialNB()
model.fit(X_train, y_train)

y_pred = model.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
precision = precision_score(y_test, y_pred)
recall = recall_score(y_test, y_pred)
f1 = f1_score(y_test, y_pred)

print(f'Accuracy: {accuracy}')
print(f'Precision: {precision}')
print(f'Recall: {recall}')
print(f'F1 Score: {f1}')

def predict_spam(message):
    message_transformed = vectorizer.transform([message])
    prediction = model.predict(message_transformed)
    return 'Spam' if prediction[0] == 1 else 'Ham'

print(predict_spam("Free entry in 2 a weekly competition to win FA Cup final tickets"))
