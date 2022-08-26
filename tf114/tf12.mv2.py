import tensorflow as tf
tf.compat.v1.set_random_seed(123)

x_data = [[73, 51, 65],                        # (5, 3)
          [92, 98, 11],
          [89, 31, 33],
          [99, 33, 100],
          [17, 66, 79]]
y_data = [[152], [185], [180], [205], [142]]   # (5, 1)
          
x = tf.compat.v1.placeholder(tf.float32, shape=[None, 3])   # None : 행 무시하겟다 라는 느낌 / 컬럼은 3이다!
y = tf.compat.v1.placeholder(tf.float32, shape=[None, 1])   # None : 행 무시하겟다 라는 느낌 / 컬럼은 3이다!

w = tf.compat.v1.Variable(tf.compat.v1.random_normal([3, 1]), name="weight")
b = tf.compat.v1.Variable(tf.compat.v1.random_normal([1]), name="bias")


#2. 모델
hypothesis = tf.compat.v1.matmul(x, w) + b   # matmul : 행렬곱을 해라! 라는 뜻
# hypothesis와 y의 열의값과 똑같다


#3-1. 컴파일
loss = tf.reduce_mean(tf.square(hypothesis - y_data)) 

optimizer = tf.train.GradientDescentOptimizer(learning_rate=1e-5)  # 멀티에서는 이렇게 안하니까 nan으로 출력되었음..

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

# [ r2 score ]#########################################################

from sklearn.metrics import r2_score, mean_absolute_error
r2 = r2_score(y_data, hy_val)
print("r2 : ", r2)
mae = mean_absolute_error(y_data, hy_val)
print("mae : ", mae)

# sess.close()                      # with를 사용했으므로 주석처리

# 0 loss :  80723.35 
#  [[-100.08837]
#  [-166.6625 ]
#  [-107.20363]
#  [-104.32795]
#  [ -53.53002]]

# 20 loss :  729.5248
#  [[175.02702]
#  [157.17424]
#  [139.15302]
#  [230.02277]
#  [148.96365]]

# 40 loss :  449.87534 
#  [[174.01619]
#  [172.46347]
#  [144.40575]
#  [223.40608]
#  [140.67157]]

# 60 loss :  355.88705
#  [[172.84326]
#  [180.63036]
#  [147.03847]
#  [218.9249 ]
#  [135.25208]]

# 80 loss :  324.09003 
#  [[172.16144]
#  [185.31671]
#  [148.62831]
#  [216.38707]
#  [131.9991 ]]

# r2 :  0.40125863627459935
# mae :  15.004840087890624


