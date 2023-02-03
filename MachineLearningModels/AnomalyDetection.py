import numpy as np
from sklearn.ensemble import IsolationForest

# Load the data
X = ... # Load the dataset of transaction records

# Train the model
model = IsolationForest(n_estimators=100, contamination=0.1)
model.fit(X)

# Predict the anomalies
y_pred = model.predict(X)

# Filter out the anomalies
X_normal = X[y_pred == 1]
X_anomalies = X[y_pred == -1]
