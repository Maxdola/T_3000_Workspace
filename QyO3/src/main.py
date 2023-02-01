import QyO3


p1 = QyO3.Point_new(0.0, 0.0)
p2 = QyO3.Point_new(3.0, 4.0)

# Calculate the distance between the two points
distance = QyO3.Point_distance(p1, p2)
print(distance) # prints 5.0
