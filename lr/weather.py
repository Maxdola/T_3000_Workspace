import time

# Record the start time of the program
start_time = time.time()

# Set the reference time
ref_time = time.time()

# Define the timestamp function to measure time elapsed
def timestamp(name):
    global ref_time
    # Print the elapsed time since last update and the name of the step
    print("[%s]:  --- %.2f s ---" % (name, (time.time() - ref_time)))
    # Update the reference time
    ref_time = time.time();

# Import pandas and sklearn libraries
import pandas as pd
from sklearn.linear_model import LinearRegression

# Measure time for imports
timestamp("Imports")

# Load the data from the csv file
#data = pd.read_csv("./data/weatherHistory.csv")
data = pd.read_csv("./data/weatherHistory_big.csv")

# Measure time for reading the file
timestamp("Reading File")

# Specify the columns to use as independent and dependent variables
X = data["Temperature (C)"]
y = data["Humidity"]

# Create empty arrays for the variables
XX = [];
yy = [];

# Append each value of X into a new array as a single element list
for v in X:
    XX.append([v])

# Append each value of y into a new array
for v in y:
    yy.append(v)

# Measure time for data digestion
timestamp("Data Digestion")

# Fit the model using the linear regression algorithm
model = LinearRegression()
model.fit(XX, yy)

# Measure time for training the model
timestamp("Training")

# Specify the independent variable values for prediction
new_X = [[0], [2]]

# Measure time for prediction
timestamp("Prediction")

# Print the intercept and coefficients of the model
print(model.intercept_)
print(model.coef_)

# Print the prediction for the specified independent variable values
print(model.predict(new_X))

# Print the total time elapsed for the program
print("--- %s seconds ---" % (time.time() - start_time))
