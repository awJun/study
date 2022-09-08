"""
[해당 프로젝트 설명]
https://colab.research.google.com/drive/1Kxmxi8RmzpydE6soHyZ0yed4mWTM07uD



tf.compat.v1.disable_eager_execution()를 사용하면

initializer=tf.compat.v1.keras.initializers.glorot_normal())



"""

import tensorflow as tf
import keras 
import numpy as np

tf.compat.v1.set_random_seed(123)

"""
GPU 환경에서 텐서1 사용하기 위해서 작업
"""
tf.compat.v1.disable_eager_execution()
print(tf.executing_eagerly())

#1. 데이터
# pip install keras==2.3
from keras.datasets import mnist
(x_train, y_train), (x_test, y_test) = mnist.load_data()

from keras.utils import to_categorical
y_train = to_categorical(y_train)
y_test = to_categorical(y_test)


x_train = x_train.reshape(60000, 28, 28, 1).astype('float32')/255.   # 이미지의 최대값은 255  / 최소값은 1
x_test = x_test.reshape(10000, 28, 28, 1).astype('float32')/255.   # 이미지의 최대값은 255  / 최소값은 1

###[ 여기서부터 텐서1 ]#############################################

#2. 모델구성                             # 4차원이므로 [None, 28, 28, 1]
x = tf.compat.v1.placeholder(tf.float32, [None, 28, 28, 1])    # input_shape    / [참고] # 1부분은 커널사이즈임
y = tf.compat.v1.placeholder(tf.float32, [None, 10])           # output_shape
#    x = inpuy_shaped이다. 
                            
rate = tf.compat.v1.placeholder(tf.float32)    #드랍 아웃 #0.3 혹은 0
# dropout을 해줘야 하는 부분에만 해주기 위해서 공간을 생성함 
#  - 텐서 2에서보면 훈련할 때는 과접합 방지를 위해서 데이터를 빼서 훈련하고 
#    마지막에 예측할때는 다시 워내 데이터로 예측한다. 그러므로 이 작업을 해준것임

# Layer1

w1 = tf.compat.v1.get_variable('w1', shape=[2, 2, 1, 128])  # 4차원으로 맞춰줘야한다.
# 2, 2는 커널사이즈  /  1은 컬러  /  (64은 필터  즉, output) 앞에서 연산된 것이 64로 output짐


L1 = tf.nn.conv2d(x, w1, strides=[1, 1, 1, 1], padding='SAME')  #  가운데 1, 1은 실질적인 stride이고 나머지 양옆에 1은 그냥 형태 맞추려고 reshape한거랑 같은거임
                                                                 # 즉 만약에 1칸씩이 아니라 2칸씩 움직여서 연산하고 싶으면 [1, 2, 2, 1]로 설정하면 된다.
                                                                 # stride는 연산할 때 움직이는 칸수를 뜻한다.
L1 = tf.nn.relu(L1)
L1_maxpool = tf.nn.max_pool2d(L1, ksize=[1, 2, 2, 1], strides= [1, 2, 2, 1], padding='SAME')   # ksize : 커널사이즈임  
                                                                 
                                                                 
# model.add(Conv2d(64, kernel_size=(2, 2), input_shape=(28, 28, 1), activation="relu"))   # stridesms 디폴트로 1이므로 생략함

print(w1)  # <tf.Variable 'w1:0' shape=(2, 2, 1, 64) dtype=float32_ref>
print(L1)  # Tensor("Conv2D:0", shape=(?, 28, 28, 128), dtype=float32)
print(L1_maxpool)   # Tensor("MaxPool2d:0", shape=(?, 14, 14, 128), dtype=float32)



# Layer2    # get_variable는 애를 불러서 사용하겠다. /  variable애는 그냥 변수 선언
w2 = tf.compat.v1.get_variable('w2', shape=[3, 3, 128, 64])   # w1의 128 때문에 3번째에 128을 넣음 / 아웃풋으로 64를 설정했음
L2 = tf.nn.conv2d(L1_maxpool, w2, strides=[1, 1, 1, 1], padding='VALID')
L2 = tf.nn.selu(L2) 
L2_maxpool = tf.nn.max_pool2d(L2, ksize=[1, 2, 2, 1], strides= [1, 2, 2, 1], padding='SAME')   # ksize : 커널사이즈임  

print(L2)  # Tensor("Selu:0", shape=(?, 12, 12, 64), dtype=float32)
print(L2_maxpool)   # Tensor("MaxPool2d_1:0", shape=(?, 6, 6, 64), dtype=float32)



# Layer3
w3 = tf.compat.v1.get_variable('w3', shape=[3, 3, 64, 32])   # w2의 64 때문에 3번째에 128을 넣음 / 아웃풋으로 32를 설정했음
L3 = tf.nn.conv2d(L2_maxpool, w3, strides=[1, 1, 1, 1], padding='VALID') 
L3 = tf.nn.elu(L3)

print(L3)   # Tensor("Elu:0", shape=(?, 4, 4, 32), dtype=float32)

# Flatten
L_flat = tf.reshape(L3, [-1, 4*4*32])
print("Flatten", L_flat)   # Flatten Tensor("Reshape:0", shape=(?, 512), dtype=float32)



# Layer4  DNN
w4 = tf.compat.v1.get_variable('w4', shape=[4*4*32, 100],
                     initializer=tf.compat.v1.keras.initializers.glorot_normal())      # initializer : 가중치를 초기화를 해준다.   (성능향상에 영향을 많이 미칠만큼 큰 비중을 차지한다.)

# initializer=tf.contrib.layers.xavier_initializer())
# initializer말고도 가중치 규제하는 녀석이 많다. 배치노말라이제이션? 이것도 인데 더 찾아봐야할 듯

b4 = tf.Variable(tf.compat.v1.random_normal([100]), name='b4')
L4 = tf.nn.selu(tf.matmul(L_flat, w4) + b4)
L4 = tf.nn.dropout(L4, rate = 0.3)   # rate = 0.3



# Layer5
w5 = tf.compat.v1.get_variable('w5', shape=[100, 10],
                     initializer=tf.compat.v1.keras.initializers.glorot_normal())      # initializer : 가중치를 초기화를 해준다.   (성능향상에 영향을 많이 미칠만큼 큰 비중을 차지한다.)

b5 = tf.Variable(tf.compat.v1.random_normal([10]), name='b5')
L5 = tf.matmul(L4, w5) + b5

hypothesis = tf.compat.v1.nn.softmax(L5)
print(hypothesis)


#3-1. 컴파일
###[두 개는 같은거임]#################################################################################
# optimizer = tf.compat.v1.train.AdamOptimizer(learning_rate=0.0001).minimize(loss)
loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=hypothesis, labels=y)) 
######################################################################################################

optimizer = tf.compat.v1.train.AdamOptimizer(learning_rate=0.0001).minimize(loss)

sess = tf.compat.v1.Session() 
sess.run(tf.compat.v1.global_variables_initializer())  

# 3-2 훈련
training_epochs = 30
batch_size = 100
total_batch = int(len(x_train)/batch_size)  # 60000/100 = 600
# print(total_batch)  # 600

for epoch in range(training_epochs):  # 총 30번 돈다.
    avg_loss = 0
    
    for i in range(total_batch):    # 총 600번 돈다
        start = i * batch_size      # 0
        end = start + batch_size    # 100
        batch_x, batch_y = x_train[start:end], y_train[start:end]   # 0~100

        feed_dict = {x:batch_x, y:batch_y, rate: 0.3}  # #드랍 아웃
        
        batch_loss, _ = sess.run([loss, optimizer], feed_dict=feed_dict)

        avg_loss += batch_loss / total_batch
    
    print('Epoch : ', '%04d' %(epoch + 1), 'loss : {:.9f}'.format(avg_loss))    # 1 epoch 값의 loss가 30번마다 출력  
print('훈련 끗!!!')  
        
        
#4. 평가, 예측
# y_predict = sess.run(tf.math.argmax(sess.run(hypothesis, feed_dict={x:x_test}), axis=1)) 
# y_test = sess.run(tf.math.argmax(y_test, axis=1))

# from sklearn.metrics import accuracy_score
# acc = accuracy_score(y_test, y_predict)
# print('acc : ', acc)      

prediction = tf.equal(tf.argmax(hypothesis, 1), tf.argmax(y, 1))
accuracy = tf.reduce_mean(tf.cast(prediction, tf.float32))
print("ACC : ", sess.run(accuracy, feed_dict={x:x_test, y:y_test,  rate: 0.0}))  # 드랍 아웃 0.0은 안날리겠다 라는 뜻


# ACC :  0.9894