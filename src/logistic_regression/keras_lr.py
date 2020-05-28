# -*- coding: utf-8 -*-
"""
Created on Mon Aug 26 14:57:30 2019

@author: lixin
"""
from __future__ import division, print_function
import tensorflow as tf
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
import numpy as np
from sklearn import metrics
from collections import Counter


def lr_model():
    inputs = tf.keras.Input((30,))
    pred = tf.keras.layers.Dense(units=1, 
                                 bias_regularizer=tf.keras.regularizers.l2(0.01),
                                 kernel_regularizer=tf.keras.regularizers.l1(0.02),
                                 activation=tf.nn.sigmoid)(inputs)
    lr = tf.keras.Model(inputs, pred)
    lr.compile(loss='binary_crossentropy',
               optimizer=tf.train.AdamOptimizer(0.001),
               metrics=['binary_accuracy'])
    
    return lr


def lr_model_seq():
    
    model = tf.keras.models.Sequential()
    model.add(tf.keras.layers.Dense(units=1, input_dim=30, activation='sigmoid',
                                    bias_regularizer=tf.keras.regularizers.l2(0.01)))
    sgd = tf.keras.optimizers.SGD(lr=0.1)
    
    model.compile(loss='binary_crossentropy', optimizer=sgd)
    return model


def convert_prob_into_class(probs):
    probs_ = np.copy(probs)
    probs_[probs_ > 0.5] = 1
    probs_[probs_ <= 0.5] = 0
    return probs_


if __name__ == '__main__':
    lr = lr_model()
    data = load_breast_cancer()
    X_train, X_test, y_train, y_test = train_test_split(data.data, data.target, test_size=0.2,
                                                        random_state=27, stratify=data.target)
    lr.fit(X_train, y_train, epochs=100, batch_size=16, validation_data=(X_test, y_test))
    
    # lr.fit(X_train, y_train, epochs=200, batch_size=16)
    y_predict = lr.predict(X_test)

    pred = convert_prob_into_class(y_predict)
    print(pred)

    y_test = y_test.reshape(-1, 1)
    
    metrics.confusion_matrix(y_test, pred)
    acc = metrics.accuracy_score(pred, y_test)
    
    sam = (y_test-pred).reshape(-1,)
    print('acc is: ', Counter(sam)[0]/float(len(y_test)))

    auc = metrics.roc_auc_score(pred, y_test)
    print('auc is: ', auc)
