import numpy as np
from matplotlib import pyplot as plt

def sigmoid(z):
    return 1 / (1 + np.exp(-z))

N = 50
D = 50

X = (np.random.random((N,D)) - 0.5 ) * 10

true_w = np.array([1,0.5, -0.5] + [0] * (D-3))

Y = np.round(sigmoid(X.dot(true_w) + np.random.randn(N) * 0.5))

costs = []

w = np.random.randn(D) / np.sqrt(D)
learning_rate = 0.001
l1 = 10.0

for iteration in range(5000):
    Yhat = sigmoid(X.dot(w))
    delta = Yhat - Y
    w = w - learning_rate * (np.dot(delta.T, X) + l1 * np.sign(w))

    cost = -(Y*np.log(Yhat) + (1-Y)*np.log(1 - Yhat)).mean() + l1*np.abs(w).mean()
    costs.append(cost)

plt.plot(costs)
plt.show()

plt.plot(true_w, label="true w")
plt.plot(w, label="w map")
plt.legend()
plt.show()

print("Final Weights: " + str(w))

# Final Weights: [ 4.80179209e-01  7.61178982e-02 -2.63552004e-01 -8.87214114e-03
#  -8.19565832e-03 -5.28652563e-02 -9.59350495e-03  1.15648366e-02
#  -8.57003178e-03  4.76377185e-03 -6.60619269e-03 -5.74034449e-04
#   1.03049043e-02  1.08705040e-02  8.80693879e-04 -8.15611939e-03
#  -1.01890429e-02  1.33568374e-03 -6.66461188e-03  1.92494240e-03
#  -4.12632639e-03  1.01828873e-02 -1.41180074e-02  2.62793294e-03
#  -3.73747449e-03 -5.68245884e-04 -1.05308633e-02  1.40688636e-03
#  -1.07557801e-02  1.49470093e-02 -3.76901304e-03 -5.45428698e-03
#  -3.24113363e-02 -7.43334524e-03  5.37606679e-03  1.08853008e-03
#  -9.11880765e-03  2.39752416e-03 -1.13234912e-02 -5.34222459e-03
#  -9.14953900e-03 -1.14286241e-01  2.31162953e-04  8.89442652e-03
#  -8.03710977e-03 -2.53470366e-03 -9.27641023e-03 -2.80549556e-03
#  -5.79196912e-03 -8.61327474e-03]
