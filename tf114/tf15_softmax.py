import numpy as np
import tensorflow as tf
tf.set_random_seed(123)

x_data = [[1, 2, 1, 1],
          [2, 1, 3, 2],
          [3, 1, 3, 4],
          [4, 1, 5, 5],
          [1, 7, 5, 5],
          [1, 2, 5, 6],
          [1, 6, 6, 6],
          [1, 7, 6, 7]]
y_data = [[0, 0, 1],   # 2
          [0, 0, 1],
          [0, 0, 1],
          [0, 1, 0],   # 1
          [0, 1, 0],
          [0, 1, 0],
          [1, 0, 0],   # 0
          [1, 0, 0]]
# y 데이터는 원핫까지 된 상태임 굿...  
# 그래서 마지막에 argmax 해야한다

#2. 모델구성 // 시작

from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split (x_data, y_data, train_size = 0.7, random_state=104)


x = tf.compat.v1.placeholder(tf.float32, shape=[None, 4])

w = tf.Variable(tf.random_normal([4, 3]), name='weight')

b = tf.Variable(tf.random_normal([1, 3]), name='bias')

y = tf.compat.v1.placeholder(tf.float32, shape=[None, 3])

hypothesis =tf.nn.softmax(tf.matmul(x, w) + b)
# model.add(Dense(3, activation='softmax', input_dim=4))  : 텐서2
# matmul는 행렬연산

#3-1. 컴파일
loss = tf.reduce_mean(-tf.reduce_sum(y * tf.log(hypothesis), axis=1))
# loss = "categorical_crossentropy"  : 텐서2

# optimizer = tf.train.GradientDescentOptimizer(learning_rate=0.01)  
# train = optimizer.minimize(loss)

train = tf.train.GradientDescentOptimizer(learning_rate=0.01).minimize(loss)

# [실습]
# 맹그러!!!
# loss만 출력해 / 맘 바꼈다. accuracy_score도 출력해


# 3-2 훈련

with  tf.compat.v1.Session() as sess:   # 나는 tf.compat.v1.Session()를 sess라고 할거야 그리고 같이(with) 실행할거야 / with문을 사용하면 안에서 작업이 끝난 후 자동으로 close를 해준다.
    sess.run(tf.global_variables_initializer())

    epoch = 1001
    for epochs in range(epoch):
        # sess.run(train)
        cost_val, hy_val, _ = sess.run([loss, hypothesis, train],    # _의 뜻은 반환하지 않지만 실행은 시키겠다. 라는 뜻
                                   feed_dict={x : x_train, y : y_train})
        # cost_val : loss와 같음   /  hy_val : hypothesis의 값
            
        if epochs % 20 == 0:
            print(epochs, "loss : ", cost_val, "\n", hy_val)    

# # [ acc score ]#########################################################
    from sklearn.metrics import r2_score, mean_absolute_error, accuracy_score
    
    y_acc_test = sess.run(tf.math.argmax(y_test, axis=1))   # axis=1하면 행의 최대값을 선별후 선별한 행의 열의 인덱스를 반환
    predict = sess.run(tf.argmax(sess.run(hypothesis, feed_dict={x:x_test}), axis=1))
    # predict = sess.run(tf.math.argmax(hy_val, axis=1))
    acc = accuracy_score(y_acc_test, predict)
    print("\nacc : ", acc)
    
    # acc :  0.3333333333333333