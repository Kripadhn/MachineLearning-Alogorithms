import numpy as np
from sklearn.model_selection import GridSearchCV
from sklearn.linear_model import Ridge

# Generate a random dataset
np.random.seed(0)
X = 2 * np.random.rand(100, 1)
y = 4 + 3 * X + np.random.randn(100, 1)

# Define the hyperparameter search space
param_grid = {'alpha': [0.1, 0.5, 1, 2, 5, 10]}

# Create a ridge regression model
reg = Ridge()

# Perform a grid search to find the best hyperparameters
grid_search = GridSearchCV(reg, param_grid, cv=5)
grid_search.fit(X, y)

# Print the best hyperparameters
print("Best Hyperparameters:", grid_search.best_params_)
