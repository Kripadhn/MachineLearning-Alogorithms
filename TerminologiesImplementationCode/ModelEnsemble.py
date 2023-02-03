import numpy as np
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestRegressor

# Generate a random dataset
np.random.seed(0)
X = 2 * np.random.rand(100, 1)
y = 4 + 3 * X + np.random.randn(100, 1)

# Fit a random forest regressor
rf = RandomForestRegressor(n_estimators=100, criterion='mse', random_state=0)
rf.fit(X, y.ravel())

# Plot the predicted and actual values
X_test = np.linspace(0, 2, 100).reshape(-1, 1)
y_test = rf.predict(X_test)

plt.plot(X, y, 'o', color='black', label='Observed')
plt.plot(X_test, y_test, '-', color='red', label='Predicted')
plt.legend(loc='best')
plt.show()
