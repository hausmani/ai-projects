import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score

# Sample dataset
data = {
    'email': [
        'Congratulations! You won a prize',
        'Meeting tomorrow at 10am',
        'Free money offer',
        'Project deadline extended',
        'Claim your reward now',
        'Lunch at 1?'
    ],
    'label': [1, 0, 1, 0, 1, 0]  # 1 = spam, 0 = not spam
}

df = pd.DataFrame(data)

# CountVectorizer()	Converts text into numerical vectors so the model can understand words (Bag of Words).
# MultinomialNB()	Naive Bayes classifier, best for text classification and word frequency data.
# accuracy_score()	Tells us what % of predictions were correct.

# Why these?
# Text → needs vectorization → CountVectorizer()
# Classifying → MultinomialNB() is great for discrete word counts
# Naive Bayes works well with spam detection due to word-based patterns

# Vectorize text
vectorizer = CountVectorizer()
X = vectorizer.fit_transform(df['email'])
y = df['label']

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# Train model
model = MultinomialNB()
model.fit(X_train, y_train)

# Evaluate
y_pred = model.predict(X_test)
print("Accuracy:", accuracy_score(y_test, y_pred))

# Test on new email
test_email = ["You've won a free vacation"]
test_vector = vectorizer.transform(test_email)
prediction = model.predict(test_vector)
print("Spam?", "Yes" if prediction[0] == 1 else "No")
