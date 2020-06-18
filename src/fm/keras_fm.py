# -*- coding: utf-8 -*-
"""
Created on Mon Aug 26 11:20:44 2019
@author: lixin

"""

import tensorflow as tf
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn import metrics
from collections import Counter
import keras_lr

K = tf.keras.backend


# 自定义交叉层
class CrossLayer(tf.keras.layers.Layer):
    def __init__(self, input_dim, output_dim=2, **kwargs):
        self.input_dim = input_dim
        self.output_dim = output_dim
        super(CrossLayer, self).__init__(**kwargs)

    def build(self, input_shape):
        self.kernel = self.add_weight(name='kernel',
                                      shape=(self.input_dim, self.output_dim),
                                      initializer='glorot_uniform',
                                      trainable=True)
        super(CrossLayer, self).build(input_shape)

    def call(self, x):
        """
        交叉项公式
        cross = 0.5 * \sum_{f=1}^{k} ((\sum_{i=1}^n v_{i,f}x_i)^2 - \sum_{i=1}^n v_{i,f}^2 x_i^2
        """
        a = K.pow(K.dot(x, self.kernel), 2)  # a(？，output_dim)
        b = K.dot(K.pow(x, 2), K.pow(self.kernel, 2))  # b(？，output_dim)
        return K.sum(a - b, 1, keepdims=True) * 0.5  # 输出维数为(?,1)为向量

    def compute_output_shape(self, input_shape):
        return input_shape[0], self.output_dim


def FM(feature_dim):
    inputs = tf.keras.Input((feature_dim,))
    liner = tf.keras.layers.Dense(units=1,
                                  bias_regularizer=tf.keras.regularizers.l2(0.01),
                                  kernel_regularizer=tf.keras.regularizers.l1(0.02),
                                  )(inputs)
    cross = CrossLayer(feature_dim)(inputs)
    add = tf.keras.layers.Add()([liner, cross])
    predictions = tf.keras.layers.Activation('sigmoid')(add)
    model = tf.keras.Model(inputs=inputs, outputs=predictions)
    model.compile(loss='binary_crossentropy',
                  optimizer=tf.train.AdamOptimizer(0.001),
                  metrics=['binary_accuracy'])
    print(model.summary())
    return model


if __name__ == '__main__':
    fm = FM(30)
    data = load_breast_cancer()
    X_train, X_test, y_train, y_test = train_test_split(data.data, data.target, test_size=0.2,
                                                        random_state=27, stratify=data.target)
    fm.fit(X_train, y_train, epochs=100, batch_size=16, validation_data=(X_test, y_test))
    pred = fm.predict(X_test)
    pred = keras_lr.convert_prob_into_class(pred)
    pred = pred.reshape(-1, )
    metrics.confusion_matrix(y_test, pred)
    acc = metrics.accuracy_score(pred, y_test)
    print('acc is: ', Counter(y_test - pred)[0] / float(len(y_test)))
