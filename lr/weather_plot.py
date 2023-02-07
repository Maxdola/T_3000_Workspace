import time

start_time = time.time()
ref_time = time.time()

def timestamp(name):
  global ref_time
  print("[%s]:  --- %.2f s ---" % (name, (time.time() - ref_time)))
  ref_time = time.time();

import pandas as pd
from sklearn.linear_model import LinearRegression

timestamp("Imports")

# Load the data from the csv file
data = pd.read_csv("./data/weatherHistory.csv")
#data = pd.read_csv("./data/weatherHistory_big.csv")

timestamp("Reading File")

# Specify the columns to use as independent and dependent variables
X = data["Temperature (C)"]
y = data["Humidity"]

XX = [];
yy = [];

for v in X:
  XX.append([v])

for v in y:
  yy.append(v)

timestamp("Data Digestion")

# Fit the model
model = LinearRegression()
model.fit(XX, yy)

timestamp("Training")

# Predict new values
new_X = [[0], [2]]

timestamp("Prediction")

# Print coefficients
print(model.intercept_)
print(model.coef_)
print(model.predict(new_X))


print("--- %s seconds ---" % (time.time() - start_time))

import matplotlib.pyplot as plt

def predictValue(x):
  return model.intercept_ + x * model.coef_;

plt.plot(XX, yy, "ro", markersize=0.1)
plt.plot(XX, predictValue(XX), "ro", markersize=0.1)
plt.xlim([-20, 40])
plt.ylim([0, 1])
plt.ylabel('Temperatur / Humidity')

plt.savefig('my_plot.png')

timestamp("Drawing")

#plt.show()