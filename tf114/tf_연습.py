import tensorflow as tf

tf.compat.v1.set_random_seed(502)

#1. 데이터 
x_data = [[0, 0], [0, 1], [1, 0], [1, 1]]  # [4, 2]
y_data = [[0], [1], [1], [0]]              # [4, 1]


#2. 모델구성
# input layer
x = tf.compat.v1.placeholder(tf.float32, shape=[None, 2])   # 각각의 공간을 만듬
y = tf.compat.v1.placeholder(tf.float32, shape=[None, 1])   # 각각의 공간을 만듬

# hidden layer
first_w = tf.compat.v1.Variable(tf.random_normal([2, 64]))
first_b = tf.compat.v1.Variable(tf.random_normal([64]))
first_hidden_layer = tf.compat.v1.matmul(x, first_w) + first_b

w = tf.compat.v1.Variable(tf.random_normal([64, 128]))      # random_normal로 변수값을 랜덤으로 지정시킴 
b = tf.compat.v1.Variable(tf.random_normal([128]))
hidden_layer = tf.compat.v1.matmul(first_hidden_layer, w) + b

w = tf.compat.v1.Variable(tf.random_normal([128, 256]))
b = tf.compat.v1.Variable(tf.random_normal([256]))
hidden_layer = tf.compat.v1.matmul(hidden_layer, w) + b

last_w = tf.compat.v1.Variable(tf.random_normal([256, 1]))
last_b = tf.compat.v1.Variable(tf.random_normal([1]))


#2. 모델
hypothesis = tf.compat.v1.sigmoid(tf.matmul(hidden_layer, last_w) + last_b)


#3-1. 컴파일
loss = -tf.reduce_mean(y*tf.log(hypothesis)+(1-y)*tf.log(1-hypothesis)) 

# optimizer = tf.train.GradientDescentOptimizer(learning_rate=0.01)  # 멀티에서는 이렇게 안하니까 nan으로 출력되었음..
optimizer = tf.compat.v1.train.AdamOptimizer(learning_rate=0.01)
train = optimizer.minimize(loss)


# 3-2 훈련
with tf.compat.v1.Session() as sess:
    sess.run(tf.global_variables_initializer())
    
    epoch = 100
    for epoch in range(epoch):
        cost_val, hy_val, _ = sess.run([loss, hypothesis, train],
                                       feed_dict={x : x_data, y : y_data})
        
        if epoch % 20 == 0:
            print(epoch, "loss : ", cost_val, "\n", hy_val)
            

#4. 예측
    y_predict = sess.run(tf.cast(hy_val > 0.5, dtype=tf.float32))
# 2진분류식
# cast로 반올림 대신 조건걸음 이거말고 반올림도 상관없음
# cast의 역할은 해당 조건이 만족하면 1 아니면 0으로 반환한다.

    from sklearn.metrics import r2_score, mean_absolute_error, accuracy_score
    acc = accuracy_score(y_data, y_predict)
    print("acc : ", acc)
    
    mae = mean_absolute_error(y_data, hy_val)
    print("men : ", mae)











