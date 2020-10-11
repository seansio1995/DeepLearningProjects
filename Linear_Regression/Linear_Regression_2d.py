import numpy as np
import matplotlib.pyplot as plt


#Load the data
X = []
Y = []

for line in open("data2.csv"):
    x1,x2,y = line.split(",")
    X.append([1, float(x1), float(x2)])
    Y.append(float(y))


#turn X and Y into numpy arrays
X = np.array(X)
Y = np.array(Y)

#plot the data
fig = plt.figure()
ax = fig.add_subplot(111, projection="3d")
ax.scatter(X[:,0], X[:, 1], Y)
plt.show()

#calculate the weights
w = np.linalg.solve(np.dot(X.T, X), np.dot(X.T, Y))
Yhat = np.dot(X, w)

#Compute r-squared
delta1 = Y- Yhat
delta2 = Y - Y.mean()
r2 = 1 - delta1.dot(delta1) / delta2.dot(delta2)
print("R squared is :" + str(r2))

#R squared is :0.9980040612475777