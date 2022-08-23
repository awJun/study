# 전에는 처음 시작하는 노드의 값을 임의로 w는 31 b는 10로 줫다 즉, 변수가 랜덤하게 선택해주 주게 하는 것이 핵심

import tensorflow as tf
# tf.set_random_seed(123)   이걸하면 랜덤하게 안들어감

#1. 데이터
x = [1, 2, 3, 4, 5]
y = [1, 2, 3, 4, 5]

W = tf.Variable(31, dtype=tf.float32)  # 연산되는 값  / 연산되는 것은 변수로
b = tf.Variable(10, dtype=tf.float32)    # 1은 임의의 값이고 아래에서 연산이 되면서 갱신이 된다.

# 변수를 랜덤으로 방법
W = tf.Variable(tf.random_normal([1]), dtype=tf.float32)    # W의 출력량의 갯수 1하면 1개가 출력 2하면 2개가 출력
b = tf.Variable(tf.random_normal([1]), dtype=tf.float32)

# #2. 모델구성
# hypothesis = x * W + b

# #3-1. 컴파일
# loss = tf.reduce_mean(tf.square(hypothesis - y))   # square: 제곱

# optimizer = tf.train.GradientDescentOptimizer(learning_rate=0.01)

# train = optimizer.minimize(loss)

# #3-2 훈련
# with  tf.compat.v1.Session() as sess:   # 나는 tf.compat.v1.Session()를 sess라고 할거야 그리고 같이(with) 실행할거야 / with문을 사용하면 안에서 작업이 끝난 후 자동으로 close를 해준다.
# # sess = tf.compat.v1.Session()         # 위에 with 작업을 해줬으므로 주석처리 
#     sess.run(tf.global_variables_initializer())

#     epochs = 2001
#     for step in range(epochs):
#         sess.run(train)
#         if step %20 == 0:
#             print(step, sess.run(loss), sess.run(W), sess.run(b))
        
# # sess.close()                      # with를 사용했으므로 주석처리





# # loss, w, b가 수럼되는 것 확인  갱신은 optimizer부분 아직 열 수 없다 일단 그냥 넘겨



