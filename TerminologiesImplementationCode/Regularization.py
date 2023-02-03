import numpy as np
from sklearn.linear_model import Ridge
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error

# Generate a random dataset
np.random.seed(0)
X = 2 * np.random.rand(100, 1)
y = 4 + 3 * X + np.random.randn(100, 1)

# Split the data into training and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

# Fit a ridge regression model with alpha=0.1
reg = Ridge(alpha=0.1).fit(X_train, y_train)

# Make predictions on the test data
y_pred = reg.predict(X_test)

# Calculate the mean squared error on the test data
mse = mean_squared_error(y_test, y_pred)
print("
