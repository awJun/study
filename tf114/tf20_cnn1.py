import tensorflow as tf
import keras 
import numpy as np

tf.compat.v1.set_random_seed(123)

#1. 데이터 
from keras.datasets import mnist
(x_train, y_train), (x_test, y_test) = mnist.load_data()

from keras.utils import to_categorical
y_train = to_categorical(x_train)
y_test = to_categorical(y_test)


x_train = x_train.reshape(60000, 28, 28, 1).astype("float32")/255.   # 이미지의 최대값은 255  / 최소값은 1
x_test = x_test.reshape(10000, 28, 28, 1).astype("float32")/255.   # 이미지의 최대값은 255  / 최소값은 1

###[ 여기서부터 텐서1 ]#############################################

#2. 모델구성
x = tf.compat.v1.placeholder(tf.float32, [None, 28, 28, 1])    # input_shape    / [참고] # 1부분은 커널사이즈임
y = tf.compat.v1.placeholder(tf.float32, [None, 10])           # output_shape

w1 = tf.compat.v1.get_variable("w1", shape=[2, 2, 1, 64])  # 4차원으로 맞춰줘야한다.
# 2, 2는 커널사이즈  /  1은 컬러  /  (64은 필터  즉, output) 앞에서 연산된 것이 64로 output짐


L1 = tf.nn.conv2d(x, w1, strides=[1, 1, 1, 1], padding="VALID")  #  가운데 1, 1은 실질적인 stride이고 나머지 양옆에 1은 그냥 형태 맞추려고 reshape한거랑 같은거임
                                                                 # 즉 만약에 1칸씩이 아니라 2칸씩 움직여서 연산하고 싶으면 [1, 2, 2, 1]로 설정하면 된다.
                                                                 # stride는 연산할 때 움직이는 칸수를 뜻한다.
# model.add(Conv2d(64, kernel_size=(2, 2), input_shape=(28, 28, 1)))   # stridesms 디폴트로 1이므로 생략함

print(w1)  # <tf.Variable 'w1:0' shape=(2, 2, 1, 64) dtype=float32_ref>
print(L1)  # Tensor("Conv2D:0", shape=(?, 27, 27, 64), dtype=float32)














