
# [실습] 히든 레이어를 여러개로 구성해서 1.0만들어 인공지능의 겨울 극복!


import tensorflow as tf

tf.compat.v1.set_random_seed(502)

#1. 데이터
x_data = [[0, 0], [0, 1], [1, 0], [1, 1]]  # (4, 2)
y_data = [[0], [1], [1], [0]]              # (4, 1)


#2. 모델구성
# input layer
x = tf.compat.v1.placeholder(tf.float32, shape=[None, 2])
y = tf.compat.v1.placeholder(tf.float32, shape=[None, 1])

# hidden layer
w1 = tf.compat.v1.Variable(tf.random_normal([2, 64]), name="weight")
b1 = tf.compat.v1.Variable(tf.compat.v1.random_normal([64]), name="bias")  # 디폴트로 1, 가 64 앞에 있음

# hidden layer
w2 = tf.compat.v1.Variable(tf.random_normal([64, 128]), name="weight")
b2 = tf.compat.v1.Variable(tf.compat.v1.random_normal([128]), name="bias")

# hidden layer
w3 = tf.compat.v1.Variable(tf.random_normal([128, 128]), name="weight")
b3 = tf.compat.v1.Variable(tf.compat.v1.random_normal([128]), name="bias")

# output layer
w4 = tf.compat.v1.Variable(tf.random_normal([128, 1]), name="weight")
b4 = tf.compat.v1.Variable(tf.compat.v1.random_normal([1]), name="bias")


hidden_layer1 = tf.compat.v1.matmul(x, w1) + b1
hidden_layer2 = tf.compat.v1.matmul(hidden_layer1, w2) + b2
hidden_layer3 = tf.compat.v1.sigmoid(tf.matmul(hidden_layer2, w3) + b3)

# [실습 시작!!!] 완성해 보아요!!!
#2. 모델
hypothesis = tf.compat.v1.sigmoid(tf.matmul(hidden_layer3, w4) + b4)


#3-1. 컴파일
loss = -tf.reduce_mean(y*tf.log(hypothesis)+(1-y)*tf.log(1-hypothesis)) 

# optimizer = tf.train.GradientDescentOptimizer(learning_rate=0.01)  # 멀티에서는 이렇게 안하니까 nan으로 출력되었음..
optimizer = tf.compat.v1.train.AdamOptimizer(learning_rate=0.01)
train = optimizer.minimize(loss)
 
# 3-2 훈련
 
with  tf.compat.v1.Session() as sess:   # 나는 tf.compat.v1.Session()를 sess라고 할거야 그리고 같이(with) 실행할거야 / with문을 사용하면 안에서 작업이 끝난 후 자동으로 close를 해준다.
    sess.run(tf.global_variables_initializer())
 
    epoch = 100
    for epochs in range(epoch):
        # sess.run(train)
        cost_val, hy_val, _ = sess.run([loss, hypothesis, train],    # _의 뜻은 반환하지 않지만 실행은 시키겠다. 라는 뜻
                                   feed_dict={x : x_data, y : y_data})
        # cost_val : loss와 같음   /  hy_val : hypothesis의 값
           
        if epochs % 20 == 0:
            print(epochs, "loss : ", cost_val, "\n", hy_val)    
 
# # [ r2 score ]#########################################################
 
    y_predict = sess.run(tf.cast(hy_val > 0.5, dtype=tf.float32))    # 2진분류식
                                                                # cast로 반올림 대신 조건걸음 이거말고 반올림도 상관없음
                                                                # cast의 역할은 해당 조건이 만족하면 1 아니면 0으로 반환한다.
    from sklearn.metrics import r2_score, mean_absolute_error, accuracy_score
    acc = accuracy_score(y_data, y_predict)
    print("acc: ", acc)
 
    mae = mean_absolute_error(y_data, hy_val)
    print("mae : ", mae)



# acc:  1.0
# mae :  1.7076730728149414e-05








