import numpy as np
import matplotlib.pyplot as plt

N = 50
D = 50

X = (np.random.random((N,D)) - 0.5) * 10

true_w = np.array([1, 0.5, -0.5] + [0]*(D-3))

Y = X.dot(true_w) + np.random.randn(N)*0.5

costs = []
w = np.random.randn(D) / np.sqrt(D)
learning_rate = 0.001

l1 = 10

for iteration in range(500):
    Yhat = X.dot(w)
    delta = Yhat-Y
    w = w - learning_rate * (np.dot(X.T, delta) + l1 * np.sign(w))

    mse = np.dot(delta, delta.T)
    costs.append(mse)

#print(costs)
plt.plot(costs)
plt.show()

print("Final weight is w: " + str(w))

# Final weight is w: [ 9.92732245e-01  4.89053054e-01 -4.95569680e-01 -2.08935176e-04
#   3.03607699e-03 -1.44935040e-02  1.00448881e-02  1.51210723e-02
#  -7.20201047e-03  1.03366936e-03  1.58743160e-02 -1.35131519e-02
#   1.31935275e-02  6.34391541e-03 -1.65152679e-02  6.56863168e-04
#   1.48039752e-02 -8.63638821e-03  3.40015277e-02  1.93297579e-03
#  -7.38917623e-03 -4.16752914e-03  9.85696368e-03  2.25250803e-02
#   1.53714512e-02 -6.11612963e-03 -1.75179445e-02  1.34065264e-02
#   6.07438890e-03 -9.93918196e-03 -2.42128435e-02  3.18327088e-02
#  -2.63592511e-02  4.22464638e-03  1.54605832e-02  1.16576425e-02
#  -1.60221118e-02 -1.53084705e-02  1.91590558e-02  6.34946251e-03
#   2.62716783e-03 -2.89609922e-02  1.26033544e-02 -8.63884494e-03
#  -1.26158940e-02 -3.33929815e-02 -1.75228411e-03  1.20385907e-02
#   1.72197545e-02 -4.21547743e-04]

plt.plot(true_w, label = "true weight")
plt.plot(w, label = "L1 weight")
plt.legend()
plt.show()