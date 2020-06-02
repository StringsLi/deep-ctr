# -*- coding: utf-8 -*-
"""
Created on Mon Sep 30 15:35:23 2019

@author: lixin
"""

import pandas as pd
import numpy as np
from keras.layers import Input, Dense, Flatten, Embedding
from keras.layers import concatenate, dot
from keras.models import Model
from sklearn import metrics
from collections import Counter
import warnings
warnings.filterwarnings("ignore")

class DeepFM():
    def __init__(self, embedding_list, emb_size, train_x, train_y, test_x, test_y):
        self.embedding_list = embedding_list
        self.emb_size = emb_size
        self.train_x = train_x
        self.train_y = train_y
        self.test_x  = test_x
        self.test_y = test_y
        self.mxlen_set = self.get_mxlen_set()
        
    def builtModel(self):
        emb_list = []
        inp_list = []
        fm_list = []
        product_list = []
                
        # embedding part and fm part
        for feat in self.embedding_list:
            inp_temp = Input(shape=[1], name=feat)
            emb_temp = Flatten()(Embedding(self.mxlen_set[feat], self.emb_size)(inp_temp))
            fm_temp = Flatten()(Embedding(self.mxlen_set[feat], 1)(inp_temp))
            inp_list.append(inp_temp)
            emb_list.append(emb_temp)
            fm_list.append(fm_temp)
        
        # fm product part
        for i in range(0, len(emb_list)):
            for j in range(i+1, len(emb_list)):
                temp = dot([emb_list[i], emb_list[j]], axes=1)
                product_list.append(temp)
                        
        # dnn part
        dnn_part = Dense(128, activation='relu')(concatenate(emb_list))
        # fm_part
        fm_part = Dense(128, activation='relu')(concatenate(product_list+fm_list))
        inp = Dense(64, activation='relu')(concatenate([dnn_part, fm_part], axis=1))
        outp = Dense(1, activation='sigmoid')(inp)
        model = Model(inputs=inp_list,outputs=outp)
        model.compile(loss="binary_crossentropy", optimizer="adam",metrics=['accuracy'])
        return model
        
                    
    def get_mxlen_set(self):
        X = {}
        for ebd in self.embedding_list:
            X[ebd] = np.max([self.train_x[ebd].max(),self.test_x[ebd].max()])+1
        return X
            
    def get_kears_data(self,data):
        X = {}
        for ebd in self.embedding_list:
            X[ebd] = np.array(data[ebd])
        return X
    
    def train(self,batch_size,epochs):
        self.model = self.builtModel()
        X_train = self.get_kears_data(self.train_x)
        self.model.fit(X_train,self.train_y,batch_size=batch_size,epochs=epochs)
            
    def predict(self,batch_size):
        X_val = self.get_kears_data(self.test_x)
        pred = self.model.predict(X_val,batch_size=batch_size)[:,0]
        return pred

def val2idx(df, cols):
    """helper to index categorical columns before embeddings.
    """
    val_types = dict()
    for c in cols:
        val_types[c] = df[c].unique()

    ## 将每一列的值转化为index
    val_to_idx = dict()
    for k, v in val_types.items():
        val_to_idx[k] = {o: i for i, o in enumerate(val_types[k])}

    for k, v in val_to_idx.items():
        df[k] = df[k].apply(lambda x: v[x])

    unique_vals = dict()
    for c in cols:
        unique_vals[c] = df[c].nunique()

    return df, unique_vals

# an auxiliary function that converts probability into class
def convert_prob_into_class(probs):
    probs_ = np.copy(probs)
    probs_[probs_ > 0.5] = 1
    probs_[probs_ <= 0.5] = 0
    return probs_


if __name__ == '__main__':
    COLUMNS = ["age", "workclass", "fnlwgt", "education", "education_num",
               "marital_status", "occupation", "relationship", "race", "gender",
               "capital_gain", "capital_loss", "hours_per_week", "native_country",
               "income_bracket"]
    
    df_train = pd.read_csv("../data/adult.data",names=COLUMNS, skipinitialspace=True)
    df_test = pd.read_csv("../data/adult.test",names=COLUMNS, skipinitialspace=True)

    # Add a feature to illustrate the logistic regression example
    df_train['income_label'] = (
        df_train["income_bracket"].apply(lambda x: ">50K" in x)).astype(int)
    df_test['income_label'] = (
        df_test["income_bracket"].apply(lambda x: ">50K" in x)).astype(int)

    # Add a feature to illustrate multiclass classification[将age分组]
    age_groups = [0, 25, 65, 90]
    age_labels = range(len(age_groups) - 1)
    df_train['age_group'] = pd.cut(
        df_train['age'], age_groups, labels=age_labels)
    df_test['age_group'] = pd.cut(
        df_test['age'], age_groups, labels=age_labels)

    embedding_cols = ['workclass', 'education', 'marital_status', 'occupation',
                      'relationship', 'race', 'gender', 'native_country']
    cont_cols = ['age', 'capital_gain', 'capital_loss', 'hours_per_week']

    target = 'income_label'
    
    df_train['IS_TRAIN'] = 1
    df_test['IS_TRAIN'] = 0
    df_deep = pd.concat([df_train, df_test])
    
    deep_cols = embedding_cols + cont_cols
    df_deep = df_deep[deep_cols + [target, 'IS_TRAIN']]
    
    from sklearn.preprocessing import StandardScaler
    scaler = StandardScaler()
    df_deep[cont_cols] = pd.DataFrame(scaler.fit_transform(df_train[cont_cols]),
        columns=cont_cols)
    df_deep, unique_vals = val2idx(df_deep, embedding_cols)
    
    train = df_deep[df_deep.IS_TRAIN == 1].drop('IS_TRAIN', axis=1)
    test = df_deep[df_deep.IS_TRAIN == 0].drop('IS_TRAIN', axis=1)
    
    X_train = train[deep_cols]
    y_train = np.array(train[target].values).reshape(-1, 1)
    X_test = test[deep_cols]
    y_test = np.array(test[target].values).reshape(-1, 1)
    
    embedding_list = embedding_cols
    
    deepfm = DeepFM(embedding_list,5,X_train,y_train,X_test,y_test)
    
    batch_size,epochs = 128, 6
    
    deepfm.train(batch_size,epochs)
    
    pred = deepfm.predict(batch_size)
    
    y_test = y_test.reshape(-1,)
    pred = convert_prob_into_class(pred)
    
    metrics.confusion_matrix(y_test, pred)
    acc = metrics.accuracy_score(pred, y_test)
    
    print('acc is: ', Counter(y_test-pred)[0]/float(len(y_test)))