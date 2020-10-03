import numpy as np
import matplotlib.pyplot as plt

N = 1000
D = 2

R_inner = 5
R_outer = 10

# distance from origin is radius + random normal
# angle theta is uniformly distributed between (0, 2pi)
R1 = np.random.randn(N//2) + R_inner
theta = 2*np.pi*np.random.random(N//2)
X_inner = np.concatenate([[R1 * np.cos(theta)], [R1 * np.sin(theta)]]).T

R2 = np.random.randn(N//2) + R_outer
theta = 2*np.pi*np.random.random(N//2)
X_outer = np.concatenate([[R2 * np.cos(theta)], [R2 * np.sin(theta)]]).T

X = np.concatenate([ X_inner, X_outer ])
T = np.array([0]*(N//2) + [1]*(N//2)) # labels: first 50 are 0, last 50 are 1

plt.scatter(X[:,0], X[:,1], c=T)
plt.show()

# add a column of ones
# ones = np.array([[1]*N]).T # old
ones = np.ones((N, 1))

# add a column of r = sqrt(x^2 + y^2)
r = np.sqrt( (X * X).sum(axis=1) ).reshape(-1, 1)
Xb = np.concatenate((ones, r, X), axis=1)

# randomly initialize the weights
w = np.random.randn(D + 2)

# calculate the model output
z = Xb.dot(w)

def sigmoid(z):
    return 1/(1 + np.exp(-z))


Y = sigmoid(z)

# calculate the cross-entropy error
def cross_entropy(T, Y):
    return -(T*np.log(Y) + (1-T)*np.log(1-Y)).sum()

learning_rate = 0.0001

costs = []

for iteration in range(5000):
    cost = cross_entropy(T, Y)
    costs.append(cost)
    if iteration % 500 == 0:
        print("Iteration " + str(iteration) + " : " + str(cost))

    delta = T - Y
    w += learning_rate * (np.dot(delta.T,Xb) - 0.1 * w)

    Y = sigmoid(Xb.dot(w))

plt.plot(costs)
plt.title("Cross-entropy per iteration")
plt.show()

print("Final w:", w)
print("Final classification rate:", 1 - np.abs(T - np.round(Y)).sum() / N)

# Iteration 4500 : 62.42389356487021
# Final w: [-1.16542533e+01  1.58090248e+00  3.20358062e-03 -1.47779165e-02]
# Final classification rate: 0.993