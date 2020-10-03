import numpy as np
from matplotlib import pyplot as plt
N = 100
D = 2

X = np.random.randn(N, D)

X[:50, :] = X[:50, :] - 2* np.ones((50, D))
X[50:, :] = X[50:, :] + 2 * np.ones((50, D))

T= np.array([0] * 50 + [1] * 50)

ones = np.array([[1]] * N)
Xb = np.concatenate((ones, X), axis=1)

#initialize random weights + 1 for residue
w = np.random.randn(D+1)

#calulate model output
z = Xb.dot(w)

#activation func
def sigmoid(z):
    return 1/(1+np.exp(-z))

Y = sigmoid(z)

#cross-entropy error
def cross_entropy(T,Y):
    error = 0
    for i in range(N):
        if T[i] == 1:
            error -= np.log(Y[i])
        else:
            error -= np.log(1-Y[i])
    return error

#random ouput
print("Random Weights Ouput:")
print(cross_entropy(T, Y))

#new weights
#w2 = np.array([0,10,10])
#cross entropy: 2.6669887566307185e-05

w2 = np.array([0,4,4])
#cross entropy: 0.2569457550439963
z2 = Xb.dot(w2)
Y2 = sigmoid(z2)
print("Fixed Weights Ouput:")
print(cross_entropy(T, Y2))

#Visualize the plot
plt.scatter(X[:, 0], X[:, 1], c=T, s=100, alpha=0.5)

x_axis = np.linspace(-6, 6, 100)
y_axis = -x_axis
plt.plot(x_axis, y_axis)
plt.show()