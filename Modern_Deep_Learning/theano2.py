import numpy as np
import theano
import theano.tensor as T
import matplotlib.pyplot as plt

from Modern_Deep_Learning.util import get_normalized_data, y2indicator

def error_rate(p, t):
    return np.mean(p != t)

def relu(a):
    return a * (a > 0)

def main():
    Xtrain, Xtest, Ytrain, Ytest = get_normalized_data()

    lr = 0.0004
    reg = 0.01

    max_iter = 50
    print_period = 10

    Xtrain = Xtrain.astype(np.float32)
    Ytrain = Ytrain.astype(np.float32)
    Xtest = Xtest.astype(np.float32)
    Ytest = Ytest.astype(np.float32)
    Ytrain_ind = y2indicator(Ytrain).astype(np.float32)
    Ytest_ind = y2indicator(Ytest).astype(np.float32)

    N, D = Xtrain.shape
    batch_sz = 500
    n_batches = N // batch_sz

    M = 300
    K = 10
    W1_init = np.random.randn(D, M) / np.sqrt(D)
    b1_init = np.zeros(M)
    W2_init = np.random.randn(M, K) / np.sqrt(M)
    b2_init = np.zeros(K)

    # step 2: define theano variables and expressions
    thX = T.matrix("X")
    thT = T.matrix("T")
    W1 = theano.shared(W1_init, "W1")
    b1 = theano.shared(b1_init, "b1")
    W2 = theano.shared(W2_init, "W2")
    b2 = theano.shared(b2_init, "b2")

    thZ = relu( thX.dot(W1) + b1)
    thY = T.nnet.softmax( thZ.dot(W2) + b2)

    # define the cost function and prediction
    cost = -(thT * T.log(thY)).sum() + reg * ((W1 * W1).sum() + (b1 * b1).sum() + (W2 * W2).sum() + (b2 * b2).sum())
    prediction = T.argmax(thY, axis= 1)

    # step 3: training expressions and functions
    # we can just include regularization as part of the cost because it is also automatically differentiated!
    update_W1 = W1 - lr * T.grad(cost, W1)
    update_b1 = b1 - lr * T.grad(cost, b1)
    update_W2 = W2 - lr * T.grad(cost, W2)
    update_b2 = b2 - lr * T.grad(cost, b2)

    train = theano.function(
        inputs = [thX, thT],
        updates = [(W1, update_W1), (b1, update_b1), (W2, update_W2), (b2, update_b2)]
    )

    get_prediction = theano.function(
        inputs = [thX, thT],
        outputs= [cost, prediction]
    )

    costs = []
    for i in range(max_iter):
        for j in range(n_batches):
            Xbatch = Xtrain[j * batch_sz:(j * batch_sz + batch_sz), ]
            Ybatch = Ytrain_ind[j * batch_sz:(j * batch_sz + batch_sz), ]

            train(Xbatch, Ybatch)

            if j % print_period == 0:
                cost_val , pred_val = get_prediction(Xtest, Ytest_ind)
                err = error_rate(pred_val, Ytest)
                print("Cost / err at iteration i=%d, batch j=%d: %.3f / %.3f" % (i, j, cost_val, err))
                costs.append(cost_val)

    plt.plot(costs)
    plt.show()

#Iteration 20
# Cost / err at iteration i=19, batch j=80: 212.307 / 0.033

#Iteration 50
# Cost / err at iteration i=49, batch j=80: 116.058 / 0.029
if __name__ == "__main__":
    main()