from sklearn.linear_model import LinearRegression
from sklearn.datasets import make_regression

# Generate sample data
X, y = make_regression(n_samples=100, n_features=1, noise=20)

# Fit the model
model = LinearRegression()
model.fit(X, y)

# Print coefficients
print(model.intercept_)
print(model.coef_)

# Predict new values
new_X = [[0], [2]]
print(model.predict(new_X))
