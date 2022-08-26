"""
mv : 멀티 변수  

[핵심]
여기서는 앞에서는 하나의 변수로 했지만 여기서는 여러개의 변수로 실습하는 것임

optimizer = tf.train.GradientDescentOptimizer(learning_rate=1e-5)
            # 멀티에서는 (learning_rate=1e-5)를 1e-5로 안하니까 nan으로 출력되었음.. 그래서 1e-5를 넣어서 사용함

"""

import tensorflow as tf
tf.compat.v1.set_random_seed(123)

#1. 데이터

x1_data = [73., 93., 89., 96., 73.]           # 국어
x2_data = [80., 88., 91., 98., 66.]           # 영어
x3_data = [75., 93., 90., 100., 70.]          # 수학
y_data = [152., 185., 180., 196., 142.]       # 환산점수

# placeholder
x1 = tf.compat.v1.placeholder(tf.float32)   # 각각의 공간을 만듬
x2 = tf.compat.v1.placeholder(tf.float32)   # 각각의 공간을 만듬
x3 = tf.compat.v1.placeholder(tf.float32)   # 각각의 공간을 만듬
y  = tf.compat.v1.placeholder(tf.float32) 

# variable (나는 랜덤으로 하고싶어서 랜덤으로 해줬음)
w1 = tf.compat.v1.Variable(tf.random_normal([1]), dtype=tf.float32)
w2 = tf.compat.v1.Variable(tf.random_normal([1]), dtype=tf.float32)
w3 = tf.compat.v1.Variable(tf.random_normal([1]), dtype=tf.float32)
b =  tf.compat.v1.Variable(tf.random_normal([1]), dtype=tf.float32)


#2. 모델
hypothesis = x1*w1 + x2*w2 + x3*w3 + b

#3-1. 컴파일
loss = tf.reduce_mean(tf.square(hypothesis - y_data)) 

optimizer = tf.train.GradientDescentOptimizer(learning_rate=1e-5)  # 멀티에서는 이렇게 안하니까 nan으로 출력되었음..

train = optimizer.minimize(loss)

# 3-2 훈련

with  tf.compat.v1.Session() as sess:   # 나는 tf.compat.v1.Session()를 sess라고 할거야 그리고 같이(with) 실행할거야 / with문을 사용하면 안에서 작업이 끝난 후 자동으로 close를 해준다.
    sess.run(tf.global_variables_initializer())

    epoch = 1000
    # for step in range(epochs):
    #     # sess.run(train)
    #     _, loss_val, W_val_1, W_val_2, W_val_3, b_val = sess.run([train, loss, w1, w2, w3, b],    # _의 뜻은 반환하지 않지만 실행은 시키겠다. 라는 뜻
    #                                feed_dict={x1:x1_data, x2:x2_data, x3:x3_data,\
    #                                   y : y_data})
    
    for epochs in range(epoch):
        # sess.run(train)
        cost_val, hy_val, _ = sess.run([loss, hypothesis, train],    # _의 뜻은 반환하지 않지만 실행은 시키겠다. 라는 뜻
                                   feed_dict={x1:x1_data, x2:x2_data, x3:x3_data,\
                                      y : y_data})
        # cost_val : loss와 같음   /  hy_val : hypothesis의 값
        

        # if step %200 == 0:
        #     print(step, loss_val, W_val_1, W_val_2, W_val_3, b_val)
            
        if epochs % 20 == 0:
            print(epochs, "loss : ", cost_val, "\n", hy_val)    

# [ r2 score ]#########################################################
# y_predict = x1_data * W_val_1 + x2_data * W_val_2 + x3_data * W_val_3
# print(y_predict)

# 선생님 버전은 y_predict할 필요 없음



# from sklearn.metrics import r2_score, mean_absolute_error
# r2 = r2_score(y_data, y_predict)
# print("r2 : ", r2)
# mae = mean_absolute_error(y_data, y_predict)
# print("mae : ", mae)


from sklearn.metrics import r2_score, mean_absolute_error
r2 = r2_score(y_data, hy_val)
print("r2 : ", r2)
mae = mean_absolute_error(y_data, hy_val)
print("mae : ", mae)

# sess.close()                      # with를 사용했으므로 주석처리


# [결과]
#   0 42473.758 [-0.5758695] [1.2433386] [-0.00694737] [0.6339202]
# 200 32.316246 [-0.07519097] [1.6242886] [0.45776194] [0.6396808]
# 400 29.020153 [-0.02456208] [1.5619111] [0.46930793] [0.6402202]
# 600 26.062634 [0.02342985] [1.5028408] [0.48019484] [0.64073914]
# 800 23.408802 [0.06892339] [1.4469036] [0.49045753] [0.6412395]
# [157.20586056 179.59768206 181.83264971 197.37569761 135.18676293]

# r2 :  0.948689462206758
# mae :  4.125952577590942




