import time
start_time = time.time()

import pandas as pd
from sklearn.linear_model import LinearRegression

# Load the data from the csv file
data = pd.read_csv("./weatherHistory.csv")
#data = pd.read_csv("./weatherHistory_big.csv")

# Specify the columns to use as independent and dependent variables
X = data[["Temperature (C)"]]
y = data["Humidity"]

# Fit the model
model = LinearRegression()
model.fit(X, y)

# Print coefficients
print(model.intercept_)
print(model.coef_)

# Predict new values
new_X = [[0], [2]]
print(model.predict(new_X))

print("--- %s seconds ---" % (time.time() - start_time))