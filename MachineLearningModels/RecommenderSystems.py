import numpy as np
from scipy.sparse import csr_matrix
from scipy.sparse.linalg import svds

# Load the data
R = ... # Load the dataset of movie ratings

# Preprocess the data
R = csr_matrix(R)

# Compute the low-rank matrix factorization
U, s, VT = svds(R, k=50)

# Compute the predicted ratings
R_pred = np.dot(np.dot(U, np.diag(s)), VT)

# Make the recommendations
... # Make recommendations to the users based on the predicted ratings
