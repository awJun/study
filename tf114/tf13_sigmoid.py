import tensorflow as tf
tf.compat.v1.set_random_seed(123)

#1. 데이터
x_data = [[1, 2], [2, 3], [3, 1], [4, 3], [5, 3], [6, 2]]     # (6, 2)
y_data = [[0], [0], [0], [1], [1], [1]]                       # (6, 1)

# [실습] 시그모이드 빼고 걍 만들어봐!!!

# placeholder
x = tf.compat.v1.placeholder(tf.float32, shape=[None, 2])   # None : 행 무시하겟다 라는 느낌 / 컬럼은 3이다!
y = tf.compat.v1.placeholder(tf.float32, shape=[None, 1])   # None : 행 무시하겟다 라는 느낌 / 컬럼은 3이다!


# variable (나는 랜덤으로 하고싶어서 랜덤으로 해줬음)
w = tf.compat.v1.Variable(tf.compat.v1.random_normal([2, 1]), name="weight")   # x가 (6, 2)이므로 [6, 1]
b = tf.compat.v1.Variable(tf.compat.v1.random_normal([1]), name="bias")


#2. 모델
hypothesis = tf.compat.v1.sigmoid(tf.matmul(x, w) + b) 
# 텐서1은 여기 부분에 활성화함수를 사용한다 여기서는 분류이므로 sigmoid
# matmul : 행렬곱을 해라! 라는 뜻

# model.add(Dense(1, activation="sigmoid", input_dim=2)) 텐서2의 이부분과 같은 뜻 


#3-1. 컴파일
# loss = tf.reduce_mean(tf.square(hypothesis - y_data)) 
loss = -tf.reduce_mean(y*tf.log(hypothesis)+(1-y)*tf.log(1-hypothesis))  # binary_crossentropy 풀어쓴 것
# model.compile(loss='binary_crossentropy')


optimizer = tf.train.GradientDescentOptimizer(learning_rate=0.01)  # 멀티에서는 이렇게 안하니까 nan으로 출력되었음..

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

# sess.close()                      # with를 사용했으므로 주석처리



# acc:  1.0
# mae :  0.2841140826543172












