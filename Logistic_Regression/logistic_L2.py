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

learning_rate = 0.1
lambdaVal = 0.1
for i in range(100):
    if i % 10 == 0:
        iteration = i // 10 + 1
        print("Iteration " + str(iteration) + " : " + str(cross_entropy(T, Y)))
    w += learning_rate * (np.dot((T - Y).T, Xb)- lambdaVal * w)
    Y = sigmoid(Xb.dot(w))

print("Final Weights: " + str(w))
#Final Weights: [-0.21467991  3.13865873  3.49757829]