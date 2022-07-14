# 12개 만들고... 최적의 weight 가중치 파일을 저장할 것import numpy as np

from tensorflow.python.keras.layers import Dense, Conv2D, Flatten, MaxPooling2D, Dropout, Conv1D   
from tensorflow.python.keras.models import Sequential
import pandas as pd
import numpy as np
import tensorflow as tf       
import time
from sklearn.datasets import load_boston
from sklearn.model_selection import train_test_split # 함수 가능성이 높음
from sklearn.preprocessing import MinMaxScaler, StandardScaler # 클래스 가능성이 높음
from sklearn.preprocessing import MaxAbsScaler, RobustScaler
 

datasets = load_boston()
x = datasets.data 
y = datasets.target

 
x_train, x_test, y_train, y_test = train_test_split(x, y,
                                                    train_size=0.7,
                                                    random_state=100,
                                                    )
 
# scaler =  MinMaxScaler()
# scaler = StandardScaler()
# scaler = MaxAbsScaler()
scaler = RobustScaler()

# print(x_train.shape)    # (354, 13)
# print(y_train.shape)    # (354,)
# print(x_test.shape)     # (152, 13)
# print(y_test.shape)     # (152,)

scaler.fit(x_train)
x_train = scaler.transform(x_train) # x_train을 수치로 변환해준다.
x_test = scaler.transform(x_test) # 


x_train = x_train.reshape(354, 13, 1)  # <-- "13, 1 ,1"은 input_shape값          
x_test = x_test.reshape(152, 13, 1)

print(x_train.shape)  # (354, 13, 1, 1)   # 다  곱했을 때 (x_train.shape) (354, 13)의 열의 값과 같아야함 
print(x_test.shape)   # (152, 13, 1, 1)   # 다  곱했을 때 (x_test.shape) (152, 13)의 열의 값과 같아야함 


# 이미지를 뽑으려면 2차원이 아닌 3차원으로 변경해야함. 고로 데이터에서 2차원 데이터를 3차원으로 바꿀 것
 
 #2. 모델구성
model = Sequential()
model.add(Conv1D(filters=50, kernel_size=(1),  
                 input_shape=(13, 1)))     #(batsh_size, row, columns, channels)
                                                                        # channels는 장수  / 1장 2장
model.add(Dropout(0.2))
model.add(Conv1D(64, (1), padding='valid', activation='relu'))                         
model.add(Dropout(0.2))
model.add(Conv1D(32, (1), padding='same', activation='relu'))
model.add(Dropout(0.2))
model.add(Conv1D(128, (1), padding='valid', activation='relu'))                         
model.add(Dropout(0.2))
model.add(Conv1D(128, (1), padding='same', activation='relu'))
model.add(Dropout(0.2))

model.add(Flatten())    # (N, 63)  (N, 175)  <- 다른데이터 수치이므로 무시 그냥 이형태로 바뀜
model.add(Dense(128, activation='relu'))
model.add(Dropout(0.2))
model.add(Dense(128, activation='relu'))
model.add(Dense(1, activation='linear'))
# model.summary()



#3. 컴파일. 훈련
model.compile(loss='mae', optimizer='adam')

from tensorflow.python.keras.callbacks import EarlyStopping, ModelCheckpoint
import datetime

earlyStopping = EarlyStopping(monitor='val_loss', patience=10, mode='auto', verbose=1, 
                              restore_best_weights=True)        

start_time = time.time()
model.fit(x_train, y_train, epochs=100, batch_size=100,
                 validation_split=0.2,
                 callbacks=[earlyStopping],
                 verbose=1)
end_time = time.time() -start_time



#4. 평가, 예측
loss = model.evaluate(x_test, y_test)

y_predict = model.predict(x_test)

from sklearn.metrics import r2_score
r2 = r2_score(y_test, y_predict)   

print('loss : ', loss)
print('r2스코어 : ', r2)
print('걸린시간 : ', end_time) 
 
 

# loss :  3.752948522567749
# r2스코어 :  0.6160197632379797
# 걸린시간 :  2.541934013366699