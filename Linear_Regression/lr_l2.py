import numpy as np
import matplotlib.pyplot as plt

N = 50
X = np.linspace(0, 10, N)
Y = X * 0.5 + np.random.randn(N)

#make outlier points
Y[-2]+=30
Y[-1]+=30

plt.scatter(X, Y)
plt.show()

X = np.vstack([np.ones(N), X]).T

#original
w_ml = np.linalg.solve(np.dot(X.T, X), np.dot(X.T, Y))
Yhat_ml = np.dot(X, w_ml)
plt.scatter(X[:,1], Y)
plt.plot(sorted(X[:, 1]), sorted(Yhat_ml), c="red")
plt.show()

#L2 regularization
l2 = 1000
w = np.linalg.solve(l2 * np.eye(2) + np.dot(X.T, X), np.dot(X.T, Y))
Yhat = np.dot(X, w)
plt.scatter(X[:,1], Y)
plt.plot(sorted(X[:, 1]), sorted(Yhat_ml), color="red", label="max likelihood")
plt.plot(sorted(X[:, 1]), sorted(Yhat), color="blue", label="L2")
plt.legend()
plt.show()
