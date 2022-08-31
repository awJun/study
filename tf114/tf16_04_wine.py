import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.datasets import load_wine
from sklearn.preprocessing import OneHotEncoder
import tensorflow as tf
tf.set_random_seed(1004)
import pandas as pd

#1. 데이터
datasets = load_wine()
x_data = datasets.data   # (178, 13)
y_data = datasets.target  # (178,)
y_data = pd.get_dummies(y_data)

# print(x_data.shape)   (178, 13)
# print(y_data.shape)   (178, 3)

#2. 모델구성
from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(x_data, y_data,
                                                    train_size = 0.7,
                                                    random_state=104
                                                    )


x = tf.compat.v1.placeholder(tf.float32, shape=[None, 13])

w = tf.compat.v1.Variable(tf.compat.v1.zeros([13,3]), name='weight')

y = tf.compat.v1.placeholder(tf.float32, shape=[None, 3])

b = tf.compat.v1.Variable(tf.compat.v1.zeros([3]), name='bias')

hypothesis =tf.nn.softmax(tf.matmul(x, w) + b)
# model.add(Dense(3, activation='softmax', input_dim=4))  : 텐서2

#3-1. 컴파일
loss = tf.reduce_mean(-tf.reduce_sum(y * tf.log(hypothesis), axis=1))
# loss = "categorical_crossentropy"  : 텐서2

# optimizer = tf.train.GradientDescentOptimizer(learning_rate=0.01)  
# train = optimizer.minimize(loss)

train = tf.train.GradientDescentOptimizer(learning_rate=0.000001).minimize(loss)


# 3-2 훈련
with tf.compat.v1.Session() as sess:
    sess.run(tf.global_variables_initializer()) 
    
    epoch = 1001
    for epochs in range(epoch):
        cost_val, hy_val, _ = sess.run([loss, hypothesis, train],
                                       feed_dict={x : x_train, y : y_train})
        
        if epochs % 20 ==0:
            print(epochs, "loss : ", cost_val, "\n", hy_val)
            
###[ acc score ]#####################################################################
    from sklearn.metrics import r2_score, mean_absolute_error, accuracy_score
    
    y_acc_test = sess.run(tf.math.argmax(y_test, axis=1))
    predict = sess.run(tf.argmax(sess.run(hypothesis, feed_dict={x : x_test}), axis=1))
    acc = accuracy_score(y_acc_test, predict)
    print("\nacc : ", acc)

# acc :  0.7222222222222222








