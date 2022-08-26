from sklearn.datasets import load_breast_cancer
import numpy as np
import pandas as pd
import tensorflow as tf
tf.compat.v1.set_random_seed(123)
from sklearn.model_selection import train_test_split

datasets = load_breast_cancer()
x_data, y_data = datasets.data, datasets.target
# print(x_data.shape, y_data.shape)    # (569, 30) (569,)

y_data = y_data.reshape(569, 1)
# print(y_data.shape)   # (569, 1)

x_train, x_test, y_train, y_test = train_test_split(x_data, y_data,
                                                    train_size=0.8,
                                                    random_state=123,
                                                    stratify=y_data
                                                    )


# placeholder
x = tf.compat.v1.placeholder(tf.float32, shape=[None, 30])   # None : 행 무시하겟다 라는 느낌 / 컬럼은 3이다!
y = tf.compat.v1.placeholder(tf.float32, shape=[None, 1])   # None : 행 무시하겟다 라는 느낌 / 컬럼은 3이다!

# variable (나는 랜덤으로 하고싶어서 랜덤으로 해줬음)
# w = tf.compat.v1.Variable(tf.compat.v1.random_normal([30, 1]), name="weight")   # x가 (569, 30)이므로 [30, 1]
# b = tf.compat.v1.Variable(tf.compat.v1.random_normal([1]), name="bias")
w = tf.compat.v1.Variable(tf.zeros([x_data.shape[1], 1]), name='weight')
b = tf.compat.v1.Variable(tf.zeros([1]), name='bias')  

#2. 모델
hypothesis = tf.compat.v1.sigmoid(tf.matmul(x, w) + b) 
# 텐서1은 여기 부분에 활성화함수를 사용한다 여기서는 분류이므로 sigmoid
# matmul : 행렬곱을 해라! 라는 뜻

# model.add(Dense(1, activation="sigmoid", input_dim=2)) 텐서2의 이부분과 같은 뜻 



#3-1. 컴파일
# loss = tf.reduce_mean(tf.square(hypothesis - y_data)) 
loss = -tf.reduce_mean(y*tf.log(hypothesis)+(1-y)*tf.log(1-hypothesis))  # binary_crossentropy 풀어쓴 것
# model.compile(loss='binary_crossentropy')


optimizer = tf.train.GradientDescentOptimizer(learning_rate=1e-7)  # 멀티에서는 이렇게 안하니까 nan으로 출력되었음..
# optimizer = tf.compat.v1.train.AdamOptimizer(learning_rate=1e-6)

train = optimizer.minimize(loss)

#3-2. 훈련
sess = tf.compat.v1.Session()
sess.run(tf.compat.v1.global_variables_initializer())

for epoch in range(3001):
    loss_val, hy_val, _= sess.run([loss, hypothesis, train], feed_dict={x:x_train, y:y_train})
    
    if epoch % 500 == 0:
        print(epoch, '\t', loss_val)


#4. 평가, 예측
y_predict = tf.cast(hypothesis >= 0.5, dtype=tf.float32)  

# accuracy score
accuracy = tf.reduce_mean(tf.cast(tf.equal(y, y_predict), dtype=tf.float32))
pred, acc = sess.run([y_predict, accuracy], feed_dict={x:x_test, y:y_test})

print("Accuracy - \n" , acc)

sess.close()



# 500      0.6496939
# 1000     0.62387365
# 1500     0.60237914
# 2000     0.5840204
# 2500     0.5680275
# 3000     0.5538816
# Accuracy - 
#  0.88596493










