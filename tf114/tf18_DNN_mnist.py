# [실습]
# DNN으로 구성!!!

import tensorflow as tf
import keras 
import numpy as np
import pandas as pd

tf.compat.v1.set_random_seed(123)

#1. 데이터 
from keras.datasets import mnist
(x_train, y_train), (x_test, y_test) = mnist.load_data()

# print(x_train.shape)  # (60000, 28, 28)
# print(y_train.shape)  # (60000,)
# print(x_test.shape)   # (10000, 28, 28)
# print(y_test.shape)   # (10000,)


x_train = x_train.reshape(60000, 28*28).astype("float32")/255.
x_test = x_test.reshape(10000, 28*28).astype("float32")/255.

# print(x_train.shape)  # (60000, 784)
# print(x_test.shape)   # (10000, 784)

y_train = pd.get_dummies(y_train)
y_test = pd.get_dummies(y_test)

print(y_train.shape)  # (60000, 10)
print(y_test.shape)   # (10000, 10)

###[ 여기서부터 텐서1 ]#############################################

#2. 모델구성
x = tf.compat.v1.placeholder(tf.float32, [None, 784])    # input_shape    / [참고] # 1부분은 커널사이즈임
y = tf.compat.v1.placeholder(tf.float32, [None, 10])           # output_shape

w1 = tf.compat.v1.get_variable("w1", [784, 10])  # 4차원으로 맞춰줘야한다.
# 2, 2는 커널사이즈  /  1은 컬러  /  (64은 필터  즉, output) 앞에서 연산된 것이 64로 output짐

b1 = tf.Variable(tf.random_normal([1, 10]), dtype=tf.float32) 

# L1 = tf.nn.conv2d(x, w1, strides=[1, 1, 1, 1], padding="VALID")  #  가운데 1, 1은 실질적인 stride이고 나머지 양옆에 1은 그냥 형태 맞추려고 reshape한거랑 같은거임
#                                                                  # 즉 만약에 1칸씩이 아니라 2칸씩 움직여서 연산하고 싶으면 [1, 2, 2, 1]로 설정하면 된다.
#                                                                  # stride는 연산할 때 움직이는 칸수를 뜻한다.
# model.add(Conv2d(64, kernel_size=(2, 2), input_shape=(28, 28, 1)))   # stridesms 디폴트로 1이므로 생략함

# print(w1)  # <tf.Variable 'w1:0' shape=(2, 2, 1, 64) dtype=float32_ref>
# print(L1)  # Tensor("Conv2D:0", shape=(?, 27, 27, 64), dtype=float32)


hypothesis =tf.nn.softmax(tf.matmul(x, w1) + b1)

loss = tf.reduce_mean(-tf.reduce_sum(y * tf.log(hypothesis), axis=1))

# train = tf.train.GradientDescentOptimizer(learning_rate=0.01).minimize(loss)
# 텐서1에서는 train과 optimizer가 같은 이름임
train = tf.train.AdamOptimizer(learning_rate=0.01).minimize(loss)

# 3-2 훈련

with  tf.compat.v1.Session() as sess:   # 나는 tf.compat.v1.Session()를 sess라고 할거야 그리고 같이(with) 실행할거야 / with문을 사용하면 안에서 작업이 끝난 후 자동으로 close를 해준다.
    sess.run(tf.global_variables_initializer())

    epoch = 1001
    for epochs in range(epoch):
        cost_val, hy_val, _ = sess.run([loss, hypothesis, train],    # _의 뜻은 반환하지 않지만 실행은 시키겠다. 라는 뜻
                                   feed_dict={x : x_train, y : y_train})
        # cost_val : loss와 같음   /  hy_val : hypothesis의 값
            
        if epochs % 20 == 0:
            print(epochs, "loss : ", cost_val, "\n", hy_val)    

# # [ acc score ]#########################################################
    from sklearn.metrics import r2_score, mean_absolute_error, accuracy_score
    
    y_acc_test = sess.run(tf.math.argmax(y_test, axis=1))   # axis=1하면 행의 최대값을 선별후 선별한 행의 열의 인덱스를 반환
    predict = sess.run(tf.argmax(sess.run(hypothesis, feed_dict={x:x_test}), axis=1))
    acc = accuracy_score(y_acc_test, predict)
    print("\nacc : ", acc)


    mae = mean_absolute_error(y_acc_test, predict)
    print('mae: ', mae)


# [GradientDescentOptimizer] 결과    
# acc :  0.8279
# mae:  0.6309

# [AdamOptimizer] 결과    
# acc :  0.9269
# mae:  0.2669