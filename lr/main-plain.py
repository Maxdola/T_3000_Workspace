import numpy as np

# Generate sample data
X = np.random.rand(100, 1)
y = 3 + 2*X + np.random.rand(100, 1)
print(X)
print(y)

# Add a column of ones to X for the intercept
X = np.hstack((np.ones((X.shape[0], 1)), X))

# Perform linear regression using matrix multiplication
beta = np.linalg.inv(X.T @ X) @ X.T @ y

# Print coefficients
print(beta[0], beta[1])

# Predict new values
new_X = np.array([[1, 0], [1, 2]])
print(new_X @ beta)
