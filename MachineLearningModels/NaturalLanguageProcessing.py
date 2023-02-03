import numpy as np
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB

# Load the data
X = ... # Load the dataset of movie reviews
y = ... # Load the labels (positive/negative)

# Preprocess the data
vectorizer = CountVectorizer()
X = vectorizer.fit_transform(X)

# Train the model
model = MultinomialNB()
model.fit(X, y)

# Predict the sentiment
y_pred = model.predict(X)
