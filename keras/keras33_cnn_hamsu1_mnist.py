from tensorflow.python.keras.callbacks import EarlyStopping, ModelCheckpoint
import matplotlib.pyplot as plt
from matplotlib import font_manager, rc
from sklearn.metrics import r2_score, accuracy_score
from tensorflow.python.keras.layers import Dense, Conv2D, Flatten, MaxPooling2D ,Dropout, Input
from tensorflow.python.keras.models import Sequential, Model
from keras.datasets import mnist,cifar10,cifar100,fashion_mnist
import pandas as pd
import numpy as np
import tensorflow as tf       
import time
from sklearn.model_selection import train_test_split # 함수 가능성이 높음
from sklearn.preprocessing import MinMaxScaler, StandardScaler # 클래스 가능성이 높음
from sklearn.preprocessing import MaxAbsScaler, RobustScaler



#1. 데이터
(x_train, y_train), (x_test, y_test) = mnist.load_data()

#--[해당 데이터의 unique형태 확인]- - - - - - - - - - - - - - - - - 
# print(np.unique(x_train, return_counts=True))
# print(np.unique(y_train, return_counts=True))    # <--- 이 항목은 확인 필수 [0, 1, 2, 3, 4, 5, 6, 7, 8, 9] 다중분류 모델이다.
# print(np.unique(x_test, return_counts=True))   
# print(np.unique(y_test, return_counts=True))
#- - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - 
#--[해당 데이터 shape 확인]- - - - - - - - - - - - - - - - - - - - -
# print(x_train.shape)   # (60000, 28, 28)
# print(y_train.shape)   # (60000,)
# print(x_test.shape)    # (10000, 28, 28)
# print(y_test.shape)    # (10000,)
# - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -   
#--[스케일러를 사용하기 위해 차원 변형 작업]- - ( 데이터의 형태가 2x2가 아닐때만 사용할 것 )- - - -
#  ( 아래 스케일러들는 2x2형태에서만 돌아가기 때문에  (60000, 28, 28)형태를 2x2로 변환하는 작업임 )

x_train = x_train.reshape(60000, 784)              
x_test = x_test.reshape(10000, 784)

print(x_train.shape)  # (60000, 784)  
print(x_test.shape)   # (10000, 784)
#- - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
#--[ 스케일러 작업]- - - - - - - - - - - - - - - - - -(데이터 안에 값들의 차이을 줄여줌(평균으로 만들어주는 작업))
# scaler =  MinMaxScaler()
scaler = StandardScaler()
# scaler = MaxAbsScaler()
# scaler = RobustScaler()
                                
scaler.fit(x_train)
x_train = scaler.transform(x_train) # x_train을 수치로 변환해준다.
x_test = scaler.transform(x_test) 
#- - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
#--[차원 변형 작업]- - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - 
x_train = x_train.reshape(60000, 196, 4, 1)              
x_test = x_test.reshape(10000, 196, 4, 1)

print(x_train.shape)  # (120, 2, 2, 1)  <-- "2, 2 ,1"는 input_shape값
print(x_test.shape)   # (30, 2, 2, 1)  
#- - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
#- -[y데이터를 x와 동일한 차원 형태로 변환] - - - - - - - - - - ( [중요]!! 회귀형에서는 할 필요없음  )
from tensorflow.python.keras.utils.np_utils import to_categorical
# y_train = to_categorical(y_train)
# y_test = to_categorical(y_test)
# print(y_train.shape, y_test.shape) #(120, 3) (30, 3)   

"""[시퀀서형]
# #2. 모델구성
model = Sequential()
model.add(Conv2D(filters=50, kernel_size=(1,1),  
                 input_shape=(2, 2, 1)))     #(batsh_size, row, columns, channels)
                                                                        # channels는 장수  / 1장 2장
model.add(Dropout(0.2))
model.add(Conv2D(64, (1, 1), padding='valid', activation='relu'))                         
model.add(Dropout(0.2))
model.add(Conv2D(32, (1, 1), padding='same', activation='relu'))
model.add(Dropout(0.2))
model.add(Conv2D(128, (1, 1), padding='valid', activation='relu'))                         
model.add(Dropout(0.2))
model.add(Conv2D(128, (1, 1), padding='same', activation='relu'))
model.add(Dropout(0.2))

model.add(Flatten())    
model.add(Dense(128, activation='relu'))
model.add(Dropout(0.2))
model.add(Dense(128, activation='relu'))
model.add(Dense(3, activation='softmax')) # sigmoid에선 output은 1이다. (softmax에서는 유니크 갯수만큼)
"""

#2. 모델구성   # 함수형
input_01 = Input(shape=(2, 2, 1))  # Dense 구성을하고  node 값을 넣고 받아오고 싶은 변수 받아온다.
conv2D_01 = Conv2D(32, (1,1), padding='valid', activation='relu')(input_01)   # 받아온 변수를 통해 훈련의 순서를 사용자가 원하는대로 할 수 있다.
conv2D_02 = Conv2D(32, (1,1), activation='relu')(conv2D_01)
conv2D_03 = Conv2D(32, (1,1), activation='relu')(conv2D_02)
conv2D_04 = Conv2D(32, (1,1), activation='relu')(conv2D_03)

flatten_01 = Flatten()(conv2D_04)   # 4차원을 2차원에서 돌아갈 수 있게 해결해주는 역할
dense_01 = Dense(32, activation='relu')(flatten_01)  
dropout_01 = Dropout(0.2)(dense_01)
dense_02 = Dense(32, activation='relu')(dropout_01)
output_01 = Dense(3, activation='softmax')(dense_02)
model = Model(inputs=input_01, outputs=output_01)  # 해당 모델의 input과 output을 설정한다.
model.summary()



#3. 컴파일. 훈련
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
loss, acc = model.evaluate(x_test, y_test)  # loss acc 각각 다른 리스트로 출력
                                            # loss = loss / acc = metrics에서 나온 accuracy 값
print('loss : ', loss)
print('acc : ', acc)


result = model.evaluate(x_test, y_test)  # loss acc 각각 다른 리스트로 출력

from sklearn.metrics import accuracy_score
y_predict = model.predict(x_test)
y_predict = np.argmax(y_predict, axis=1)
# print(y_predict)

y_test = np.argmax(y_test, axis=1)
# print(y_test)

acc = accuracy_score(y_test, y_predict)
print('accuracy : ', acc)

print("걸린시간 : ", end_time)








