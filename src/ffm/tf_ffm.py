# -*- coding: utf-8 -*-
"""
Created on Thu Aug 29 09:08:52 2019

@author: lixin
"""

import numpy as np
import tensorflow as tf
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split

data = load_breast_cancer()
field_dict = {i: i // 5 for i in range(30)}
X_train, X_test, y_train, y_test = train_test_split(data.data, data.target, test_size=0.2,
                                                    random_state=27, stratify=data.target)
y_train.shape += (1,)
y_test.shape += (1,)

n, p = X_train.shape

# 隐变量的个数
k = 5
field_dim = len(field_dict)

X = tf.placeholder('float', shape=[None, p])
y = tf.placeholder('float', shape=[None, 1])

w0 = tf.Variable(tf.zeros([1]))
W = tf.Variable(tf.zeros([p]))

V = tf.get_variable('v', shape=[p, field_dim, k], initializer=tf.truncated_normal_initializer(mean=0, stddev=0.01))

y_hat = tf.Variable(tf.zeros([n, 1]))

linear_terms = tf.add(w0, tf.reduce_sum(tf.multiply(X, W), 1, keep_dims=True))

interactions = tf.constant(0, dtype='float32')
for i in range(p):
    for j in range(i + 1, p):
        interactions += tf.multiply(
            tf.reduce_sum(tf.multiply(V[i, field_dict[j]], V[j, field_dict[i]])),
            tf.multiply(X[:, i], X[:, j]))

interactions = tf.reshape(interactions, [-1, 1])
y_hat = tf.add(linear_terms, interactions)
cross_entropy = tf.nn.sigmoid_cross_entropy_with_logits(labels=y, logits=y_hat)
loss = tf.reduce_mean(cross_entropy)

eta = tf.constant(0.05)
optimizer = tf.train.AdagradOptimizer(eta).minimize(loss)

y_out_prob = tf.nn.sigmoid(y_hat)
predicted = tf.cast(y_out_prob > 0.5, dtype=tf.float32)
accuracy = tf.reduce_mean(tf.cast(tf.equal(predicted, y), dtype=tf.float32))

n_epochs = 1000
init = tf.global_variables_initializer()
with tf.Session() as sess:
    sess.run(init)
    for epoch in range(n_epochs):
        indices = np.arange(n)
        np.random.shuffle(indices)
        x_data, y_data = X_train[indices], y_train[indices]
        sess.run(optimizer, feed_dict={X: x_data, y: y_data})

    h, c, a = sess.run([y_out_prob, predicted, accuracy],
                       feed_dict={X: X_test, y: y_test})
    print("\nHypothesis: ", h, "\nCorrect (Y): ", c, "\nAccuracy: ", a)
