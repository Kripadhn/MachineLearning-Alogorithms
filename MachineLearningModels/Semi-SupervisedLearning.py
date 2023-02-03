import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split

Load the labeled data
labeled_data = pd.read_csv("labeled_reviews.csv")

Split the labeled data into training and test sets
X_labeled = labeled_data.drop("sentiment", axis=1)
y_labeled = labeled_data["sentiment"]
X_train, X_test, y_train, y_test = train_test_split(X_labeled, y_labeled, test_size=0.2)

Load the unlabeled data
unlabeled_data = pd.read_csv("unlabeled_reviews.csv")

Train a logistic regression model on the labeled data
model = LogisticRegression().fit(X_train, y_train)

Make predictions on the unlabeled data
y_pred = model.predict(unlabeled_data)

Combine the labeled and unlabeled data
X_all = pd.concat([X_labeled, unlabeled_data], ignore_index=True)
y_all = np.concatenate([y_labeled, y_pred], axis=0)

Train a new model on the combined data
model = LogisticRegression().fit(X_all, y_all)

Evaluate the model on the test set
print("Accuracy: ", model.score(X_test, y_test))