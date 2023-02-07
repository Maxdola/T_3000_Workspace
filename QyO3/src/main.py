import QyO3
import time

start_time = time.time()
ref_time = time.time()

def timestamp(name):
  global ref_time
  print("[%s]:  --- %.2f s ---" % (name, (time.time() - ref_time)))
  ref_time = time.time();

timestamp("Imports")

result = QyO3.read_csv("./data/weatherHistory_big.csv", ["Temperature (C)", "Humidity"]);

timestamp("Reading File + Data Digestion")

print(len(result[0]))
print(len(result[1]))

model = QyO3.train(result[0], result[1]);

timestamp("Training")

print(model)

print(QyO3.predict(model, 0))
print(QyO3.predict(model, 2))

timestamp("Prediction")

print("--- %s seconds ---" % (time.time() - start_time))