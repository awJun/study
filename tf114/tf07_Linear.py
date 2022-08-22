# y = wx + b   이게 모델이여 ~ 

from pickletools import optimize
import tensorflow as tf
tf.set_random_seed(123)

#1. 데이터
x = [1, 2, 3]
y = [1, 2, 3]

W = tf.Variable(1, dtype=tf.float32)  # 연산되는 값  / 연산되는 것은 변수로
b = tf.Variable(1, dtype=tf.float32)    # 1은 임의의 값이고 아래에서 연산이 되면서 갱신이 된다.

#2. 모델구성
hypothesis = x * W + b    # y를 통상적으로 이렇게 많이 부른다고 한다.   hypothesis : 가설 즉,! 가설을 세우다 이 모델을 우리가 만들어볼거야 ~~ 
                         # y = xw + b 사실 이거야 ~ 미안! 차이점은 뭘까 ? 두 개의 연산이 아예 달라서 행렬이 연산이기 때문에
                        # 그래서 우리는 input값에 w를 곱해야한다.

#3-1. 컴파일  (loss와 mse)가 들어가는데 여기서 우리는  loss를 풀어서 넣어보자 ~
loss = tf.reduce_mean(tf.square(hypothesis - y))   # 이게 mse이다  즉! loss는 mse를 쓸거에요 ~ 라는 뜻임

# 양수값이 . 상쇄도니ㅡㄴ 것을 방지하기 위해서 절ㄷ값 사용했었었음

optimizer = tf.train.GradietnDescentOptimizer(learning_rate=0.01)       # GradientDescentOptimizer는 경사하강법의 최하단 부분을 찾는것 
                                                                        # y는 loss는 

train = optimizer.minimize(loss)  # loss의 최하단 부분의 값을 찾는다.

# model.compile(loss='mse', optimizer='sgd') 텐서2의 이거랑 같은 거라고함 

#3-2 훈련
sess = tf.compat.v1.Session()
sess.run(tf.global_variables_initializer())

epochs = 2001
for step in range(epochs):   # 2000번 훈련시킬거야 
    sess.run(train)
    if step %20 == 0:  # 20번에 한 번만 출력할거야
        print(step, sess.run(loss), sess.run(W), sess.run(b))
        
sess.close()   # 닫아줘야함

# 0 0.8110667 0.96 0.98
# 20 0.09747055 0.68079156 0.8102924

# 1980 7.2730195e-06 0.99686784 0.007120189
# 2000 6.6050598e-06 0.99701506 0.006785556

# GradietnDescentOptimizer에서 경사하강법으로 최적의 값을 찾아주는 원리임
