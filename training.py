## Ali Akbar Naqvi
## Internship Project 3
## KNN Image Classifier

import tensorflow.compat.v1 as tf
tf.disable_v2_behavior()
import numpy as np
import pandas as pd
import pickle
from sklearn.model_selection import train_test_split, KFold
from sklearn.preprocessing import StandardScaler
from sklearn.utils import resample


train_data=pd.read_csv("train.csv").head(3500)
test_data=pd.read_csv("test.csv").head(3500)

y_train = train_data['label']
x_train = train_data.drop('label',axis=1)

x_train,x_test,y_train,y_test= train_test_split(x_train,y_train, test_size=0.3, random_state=42)

Scaler= StandardScaler()
Scaler.fit(x_train,y_train)
x_train= Scaler.transform(x_train)

data=np.column_stack((x_train,y_train))
min_class_size= min(np.bincount(y_train))


data_downsampled = []
for label in np.unique(y_train):
    class_data = data[data[:, -1] == label]
    if len(class_data)> min_class_size:
        class_data_downsampled = resample(class_data, replace=False, n_samples=min_class_size, random_state=42)
    else:
        class_data_downsampled=class_data

    data_downsampled.append(class_data_downsampled)

data_downsampled = np.vstack(data_downsampled)
x_sampled=data_downsampled[:,:-1]
y_sampled=data_downsampled[:,-1]

x_train=x_sampled
y_train=y_sampled


def knn_with_tf(x_train, y_train, x_test, k=1, batching_size=100):
    x_train=tf.cast(x_train, dtype=tf.float32)
    x_test= tf.cast(x_test, dtype=tf.float32)
    y_train = tf.cast(y_train, dtype=tf.int32)
    x_test_expanded = tf.expand_dims(x_test, axis=1)
    distance = tf.sqrt(tf.reduce_sum(tf.square(x_test_expanded - x_train), axis=2))
    KNN_Indices = tf.argsort(distance, axis=1)[:, :k]
    KNN_Indices = tf.cast(KNN_Indices, dtype=tf.int32)
    KNN_Label = tf.gather(y_train, KNN_Indices)
    mode, _ = tf.math.top_k(tf.reduce_sum(tf.one_hot(KNN_Label, depth=3), axis=1), k=1)
    return tf.squeeze(mode)


kf = KFold(n_splits=5)

best_k = 1
best_k_accuracy = tf.Variable(0.0, dtype=tf.float32)
with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    #check for optimal k in ranges of 5 digits if CPU performance is limited
    for k in range(1,5):
        accuracy=0
        for train_index, test_index in kf.split(x_train):
            x_train_fold, x_test_fold= x_train[train_index], x_train[test_index]
            y_train_fold, y_test_fold = y_train[train_index], y_train[test_index]
            prediction=knn_with_tf(x_train,y_train,x_test, k=k, batching_size=1000)
            accuracy= tf.reduce_mean(tf.cast(tf.equal(prediction, y_test), tf.float32))
            acc_val = sess.run(accuracy)
            accuracy += acc_val
            avrg_accuracy= accuracy / kf.get_n_splits()

        sess.run(tf.cond(avrg_accuracy > best_k_accuracy,
                         lambda: best_k_accuracy.assign(avrg_accuracy),
                         lambda: best_k_accuracy))
        if sess.run(avrg_accuracy > best_k_accuracy):
            best_k_accuracy=avrg_accuracy
            best_k=k

        prediction = knn_with_tf(x_train, y_train, x_test, k=best_k, batching_size=1000)

        with open("best_k.pkl", "wb") as file:
            pickle.dump({'k': best_k, 'best_k_accuracy': sess.run(best_k_accuracy)}, file)

    print(f"best K is {best_k} with Accuracy {sess.run(best_k_accuracy):.4f}")
