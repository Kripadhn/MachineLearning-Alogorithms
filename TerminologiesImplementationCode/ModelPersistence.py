import pickle
from sklearn.linear_model import LinearRegression

# Generate a random dataset
np.random.seed(0)
X = 2 * np.random.rand(100, 1)
y = 4 + 3 * X + np.random.randn(100, 1)

# Fit a linear regression model
reg = LinearRegression().fit(X, y)

# Save the model to disk
filename = 'lin_reg.pkl'
pickle.dump(reg, open(filename, 'wb'))

# Load the saved model from disk
loaded_reg = pickle.load(open(filename, 'rb'))

# Use the loaded model to make predictions
y_pred = loaded_reg.predict(X)

# Print the mean squared error
print("Mean Squared Error:", np.mean((y - y_pred)**2))
