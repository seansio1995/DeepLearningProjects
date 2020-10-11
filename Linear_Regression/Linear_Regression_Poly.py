import numpy as np
import matplotlib.pyplot as plt


#Load the data
X = []
Y = []

for line in open("data_poly.csv"):
    x, y = line.split(",")
    x = float(x)
    X.append([1, x, x**2])
    Y.append(float(y))

#turn X and Y into numpy arrays
X = np.array(X)
Y = np.array(Y)

#plot the data
plt.scatter(X[:, 1] , Y)
plt.show()

#solve the weight w
w = np.linalg.solve(np.dot(X.T, X), np.dot(X.T, Y))
Yhat = np.dot(X, w)

#plot it all together
plt.scatter(X[:, 1], Y)
plt.plot(sorted(X[:, 1]), sorted(Yhat), c="red")
plt.show()

#Compute r-squared
delta1 = Y- Yhat
delta2 = Y - Y.mean()
r2 = 1 - delta1.dot(delta1) / delta2.dot(delta2)
print("R squared is :" + str(r2))

#R squared is :0.9991412296366858