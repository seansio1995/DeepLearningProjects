import numpy as np
import tensorflow.compat.v1 as tf
tf.disable_v2_behavior()

# you have to specify the type
A = tf.placeholder(tf.float32, shape=(5, 5), name='A')
print(A)

# but shape and name are optional
v = tf.placeholder(tf.float32)

w = tf.matmul(A, v)

with tf.Session() as session:
    output = session.run(w, feed_dict = {
        A: np.random.randn(5,5),
        v: np.random.randn(5,1)
    })

    print(output, type(output))

# TensorFlow variables are like Theano shared variables.
# But Theano variables are like TensorFlow placeholders.

shape = (2,2)
x = tf.Variable(tf.random_normal(shape))
t = tf.Variable(0)

init = tf.global_variables_initializer()

with tf.Session() as session:
    out = session.run(init)
    print(out)

    print(x.eval())
    print(t.eval())

# find mininum of cost function
u = tf.Variable(20.0)
cost = u*u + u + 1.0

# One difference between Theano and TensorFlow is that you don't write the updates
# yourself in TensorFlow. You choose an optimizer that implements the algorithm you want.
# 0.3 is the learning rate.
train = tf.train.GradientDescentOptimizer(0.3).minimize(cost)

init = tf.global_variables_initializer()
with tf.Session() as session:
    session.run(init)

    for i in range(20):
        session.run(train)
        print("i = %d, cost = %.3f, u = %.3f" % (i, cost.eval(), u.eval()))

