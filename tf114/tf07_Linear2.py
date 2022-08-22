# y = wx + b   이게 모델이여 ~ 

from pickletools import optimize
import tensorflow as tf
tf.set_random_seed(123)

#1. 데이터
x = [1, 2, 3, 4, 5]
y = [1, 2, 3, 4, 5]

W = tf.Variable(31, dtype=tf.float32)  # 연산되는 값  / 연산되는 것은 변수로
b = tf.Variable(10, dtype=tf.float32)    # 1은 임의의 값이고 아래에서 연산이 되면서 갱신이 된다.

# loss는 0에 가까울수록 좋음 / w는 1에 가까울수록 좋음 / b는 0에 가까울수록 좋음

#2. 모델구성
hypothesis = x * W + b

#3-1. 컴파일
loss = tf.reduce_mean(tf.square(hypothesis - y))

optimizer = tf.train.GradientDescentOptimizer(learning_rate=0.01)

train = optimizer.minimize(loss)

#3-2 훈련
sess = tf.compat.v1.Session()
sess.run(tf.global_variables_initializer())

epochs = 2001
for step in range(epochs):
    sess.run(train)
    if step %20 == 0:
        print(step, sess.run(loss), sess.run(W), sess.run(b))
        
sess.close()





# loss, w, b가 수럼되는 것 확인  갱신은 opimizer부분 아직 열 수 없다 일단 그냥 넘겨



