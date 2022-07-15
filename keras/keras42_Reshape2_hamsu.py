"""
[내용 정리]

model.add(Flatten())                          <-- 무조건 2차원으로 내려줌        # (N, 700)    
model.add(Reshape(target_shape=(700, 1)))     <-- 3차원으로 변환해줌
model.add(Reshape(target_shape=(700, 1, 1)))  <-- 4차원으로 변환해줌

"""

from tensorflow.python.keras.models import Sequential, Model, Input
from tensorflow.python.keras.layers import Dense, Conv2D, Flatten, MaxPooling2D, Dropout
from tensorflow.python.keras.layers import Conv1D, LSTM, Reshape
from tensorflow.keras.datasets import mnist
from tensorflow.python.keras.callbacks import EarlyStopping, ModelCheckpoint
import numpy as np
import time
import matplotlib.pyplot as plt
from matplotlib import font_manager, rc


#1. 데이터
(x_train, y_train), (x_test, y_test) = mnist.load_data()

print(x_train.shape, y_train.shape)     # (60000, 28, 28) (60000,)
print(x_test.shape, y_test.shape)       # (10000, 28, 28) (10000,)

x_train = x_train.reshape(60000, 28, 28, 1)  # (60000, 28, 28) (60000,)
x_test = x_test.reshape(10000, 28, 28, 1)   # (10000, 28, 28) (10000,)
# print(x_train.shape)    # (60000, 28, 28)
print(np.unique(x_train, return_counts=True))  # [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]



# [실습] 
# acc 0.98 이상
# convolution 3개 이상 사용


# One Hot Encoding
import pandas as pd
# df = pd.DataFrame(y)
y_train = pd.get_dummies(y_train)
y_test = pd.get_dummies(y_test)
print(y_train)
print(y_train.shape)


#2. 모델링 
# Input, Model(함수형) 모델
input_01 = Input(shape=(28, 28, 1))  # Dense 구성을하고  node 값을 넣고 받아오고 싶은 변수 받아온다.

conv2D_01 = Conv2D(64, (3, 3), padding='same')(input_01)   # 받아온 변수를 통해 훈련의 순서를 사용자가 원하는대로 할 수 있다.
MaxPooling2D = MaxPooling2D()(conv2D_01)
conv2D_02 = Conv2D(32, (2, 2))(MaxPooling2D)
conv2D_03 = Conv2D(77, (3, 3))(conv2D_02)
Flatten = Flatten()(conv2D_03)
Dense_01 = Dense(100, activation='relu')(Flatten)
#-----------------------------------------------------------------------------------------(핵심내용)
Reshape = (Reshape(target_shape=(100, 1)))(Dense_01)   # 4차원을 2차원에서 돌아갈 수 있게 해결해주는 역할
Conv1D = (Conv1D(10, kernel_size=3))(Reshape)  
LSTM_01 = (LSTM(16))(Conv1D)
#------------------------------------------------------------------------------------------
Dense_02 = Dense(32, activation='relu')(LSTM_01)
Dense_03 = (Dense(10, activation='softmax'))(Dense_02)
model = Model(inputs=input_01, outputs=Dense_03)  # 해당 모델의 input과 output을 설정한다.
model.summary()


# # Sequential 모델
# model = Sequential()
# model.add(Conv2D(filters=64, kernel_size=(3, 3),    
#                  padding='same', 
#                  input_shape=(28, 28, 1)))      
# model.add(MaxPooling2D())                       # (N, 14, 14, 64)       
# model.add(Conv2D(32, kernel_size=(2, 2)))       # (N, 12, 12, 32)       # model.add(Conv2D(32, (2, 2)))   2x2면 1씩 빼고 3x3이면 2씩 빼야함   
# model.add(Conv2D(77, kernel_size=(3, 3)))       # (N, 10, 10, 77)
# model.add(Flatten())                            # (N, 700)          # <-- 무조건 2차원으로 내려줌
# model.add(Dense(100, activation='relu'))        # (N, 100)
# #--------------------------------------------------------------------------(핵심내용)
# model.add(Reshape(target_shape=(100, 1)))       # (N, 100, 1)                   # <--3차원으로 변환해줌  4차원으로 선언하면 4차원으로 Reshape해줌
# model.add(Conv1D(10, kernel_size=3))            # (N, 98, 10)                   # (N, 100, 10)  <--  패딩 잡았을 경우
# model.add(LSTM(16))                             # (N, 16,)
# #--------------------------------------------------------------------------
# model.add(Dense(32, activation='relu'))         # (N, 32)
# model.add(Dense(10, activation='softmax'))      # (N, 10)
# model.summary()

#=================================================================================================================================[비교]
#[함수형 summary]
# Model: "model"
# _________________________________________________________________
# Layer (type)                 Output Shape              Param #
# =================================================================
# input_1 (InputLayer)         [(None, 28, 28, 1)]       0
# _________________________________________________________________
# conv2d (Conv2D)              (None, 28, 28, 64)        640
# _________________________________________________________________
# max_pooling2d (MaxPooling2D) (None, 14, 14, 64)        0
# _________________________________________________________________
# conv2d_1 (Conv2D)            (None, 13, 13, 32)        8224
# _________________________________________________________________
# conv2d_2 (Conv2D)            (None, 11, 11, 77)        22253
# _________________________________________________________________
# flatten (Flatten)            (None, 9317)              0
# _________________________________________________________________
# dense (Dense)                (None, 100)               931800
# _________________________________________________________________
# reshape (Reshape)            (None, 100, 1)            0
# _________________________________________________________________
# conv1d (Conv1D)              (None, 98, 10)            40
# _________________________________________________________________
# lstm (LSTM)                  (None, 16)                1728
# _________________________________________________________________
# dense_1 (Dense)              (None, 32)                544
# _________________________________________________________________
# dense_2 (Dense)              (None, 10)                330
# =================================================================
# Total params: 965,559
# Trainable params: 965,559
# Non-trainable params: 0
# _________________________________________________________________



#[sequential형 summary]
# Model: "sequential"
# _________________________________________________________________
# Layer (type)                 Output Shape              Param #
# =================================================================
# conv2d (Conv2D)              (None, 28, 28, 64)        640
# _________________________________________________________________
# max_pooling2d (MaxPooling2D) (None, 14, 14, 64)        0                      <-- 반으로 줄음 로우 컬럼
# _________________________________________________________________  
# conv2d_1 (Conv2D)            (None, 13, 13, 32)        8224
# _________________________________________________________________
# conv2d_2 (Conv2D)            (None, 11, 11, 77)        22253
# _________________________________________________________________
# flatten (Flatten)            (None, 9317)              0
# _________________________________________________________________
# dense (Dense)                (None, 100)               931800
# _________________________________________________________________
# reshape (Reshape)            (None, 100, 1)            0                       <-- 연산양이 없으므로 0임 얘는 순서와 내용은 바뀌지 않는다. 펼친다는 개념 flatten와 동일함
# _________________________________________________________________
# conv1d (Conv1D)              (None, 98, 10)            40
# _________________________________________________________________
# lstm (LSTM)                  (None, 16)                1728
# _________________________________________________________________
# dense_1 (Dense)              (None, 32)                544
# _________________________________________________________________
# dense_2 (Dense)              (None, 10)                330
# =================================================================
# Total params: 965,559
# Trainable params: 965,559
# Non-trainable params: 0
# _________________________________________________________________


# [위에 summary 비교!]
# [함수형 summary]와 [sequential형 summary]을 비교해보면 같은 결과인 것을 볼 수 있다. 

#=================================================================================================================================[비교 완료]





#3. 컴파일, 훈련
model.compile(loss = 'categorical_crossentropy', optimizer='adam', 
              metrics=['accuracy'])

import datetime
date = datetime.datetime.now()      
date = date.strftime("%m%d_%H%M")  
print(date)

filepath = './_ModelCheckPoint/k28/'
filename = '{epoch:04d}-{val_loss:.4f}.hdf5'

earlyStopping = EarlyStopping(monitor='val_loss', patience=20, mode='auto',
                              restore_best_weights=True,
                              verbose=1)
mcp = ModelCheckpoint(monitor='val_loss', mode='auto', verbose=1, 
                      save_best_only=True, 
                      filepath="".join([filepath, '01_', date, '_', filename])
                      )
start_time = time.time()
hist = model.fit(x_train, y_train, epochs=50, batch_size=32,
                 validation_split=0.2,
                 callbacks=[earlyStopping, mcp],
                 verbose=1)
end_time = time.time() - start_time


#4. 평가, 예측
loss, acc = model.evaluate(x_test, y_test)
print('loss : ', loss)
print('accuracy : ', acc)

import tensorflow as tf
y_predict = model.predict(x_test)
y_predict = tf.argmax(y_predict, axis=1)    
y_test = tf.argmax(y_test, axis=1)          
