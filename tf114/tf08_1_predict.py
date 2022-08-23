# x와 y를 이제 placeholder에 담아서 사용한다.
# 탠서1은 레이어마다 shape 잡아준다고함 걍 떡밥임

import tensorflow as tf
# tf.set_random_seed(123)   이걸하면 랜덤하게 안들어감

#1. 데이터
# x = [1, 2, 3, 4, 5]
# y = [1, 2, 3, 4, 5]
x_train = tf.placeholder(tf.float32, shape=[None])   # shape는 input_shape를 뜻한다.  / None 자동으로 잡아줌   /  의미는 5, 1 뜻이라고함  ?? <-- feed_dict의 형태
y_train = tf.placeholder(tf.float32, shape=[None])


# W = tf.Variable(31, dtype=tf.float32)  # 연산되는 값  / 연산되는 것은 변수로
# b = tf.Variable(10, dtype=tf.float32)    # 1은 임의의 값이고 아래에서 연산이 되면서 갱신이 된다.

# 변수를 랜덤으로 방법
W = tf.Variable(tf.random_normal([1]), dtype=tf.float32)    # W의 출력량의 갯수 1하면 1개가 출력 2하면 2개가 출력
b = tf.Variable(tf.random_normal([1]), dtype=tf.float32)    # random_normal의 괄호안에 수는 1또는 feed_dict과 갯수를 맞쳐줘야 한다.

#2. 모델구성
hypothesis = x_train * W + b

#3-1. 컴파일
loss = tf.reduce_mean(tf.square(hypothesis - y_train))   # square: 제곱

optimizer = tf.train.GradientDescentOptimizer(learning_rate=0.01)

train = optimizer.minimize(loss)

#3-2 훈련
with  tf.compat.v1.Session() as sess:   # 나는 tf.compat.v1.Session()를 sess라고 할거야 그리고 같이(with) 실행할거야 / with문을 사용하면 안에서 작업이 끝난 후 자동으로 close를 해준다.
# sess = tf.compat.v1.Session()         # 위에 with 작업을 해줬으므로 주석처리 
    sess.run(tf.global_variables_initializer())

    epochs = 2001
    for step in range(epochs):
        # sess.run(train)
        _,loss_val, W_val, b_val = sess.run([train, loss, W, b],    # _의 뜻은 반환하지 않지만 실행은 시키겠다. 라는 뜻
                                   feed_dict={x_train:[1, 2, 3, 4, 5], y_train:[1, 2, 3, 4, 5]})
        if step %20 == 0:
            print(step, loss_val, W_val, b_val)
 # sess.close()                      # with를 사용했으므로 주석처리

    x_test_data = [6, 7, 8]
    x_test = tf.compat.v1.placeholder(tf.float32, shape=[None]) 

    y_predict = x_test * W_val + b_val
    print("[6, 7, 8] 예측 : ", sess.run(y_predict, feed_dict={x_test : x_test_data}))







# # loss, w, b가 수럼되는 것 확인  갱신은 optimizer부분 아직 열 수 없다 일단 그냥 넘겨



