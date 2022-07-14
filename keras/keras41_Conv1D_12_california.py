# 12개 만들고... 최적의 weight 가중치 파일을 저장할 것import numpy as np

from tensorflow.python.keras.layers import Dense, Conv2D, Flatten, MaxPooling2D, Dropout, Conv1D   
from tensorflow.python.keras.models import Sequential
import pandas as pd
import numpy as np
import tensorflow as tf       
import time
from sklearn.datasets import fetch_california_housing
from sklearn.model_selection import train_test_split # 함수 가능성이 높음
from sklearn.preprocessing import MinMaxScaler, StandardScaler # 클래스 가능성이 높음
from sklearn.preprocessing import MaxAbsScaler, RobustScaler
 

datasets = fetch_california_housing()

x=datasets.data
y=datasets.target


x_train, x_test, y_train, y_test = train_test_split(x, y,
                                                    train_size=0.7,
                                                    shuffle=True,
                                                    random_state=100
                                                    )

#--[해당 데이터 shape 확인]- - - - - - - - - - - - - - - - - - - - -
# print(x_train.shape)   # (14447, 8)
# print(y_train.shape)   # (14447,)
# print(x_test.shape)    # (6193, 8)
# print(y_test.shape)    # (6193,)
#- - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
#--[ 스케일러 작업]- - - - - - - - - - - - - - - - - - - - - - - - -
# scaler =  MinMaxScaler()
# scaler = StandardScaler()
# scaler = MaxAbsScaler()
scaler = RobustScaler()


scaler.fit(x_train)
x_train = scaler.transform(x_train) # x_train을 수치로 변환해준다.
x_test = scaler.transform(x_test) # 
#- - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
#--[차원 변형 작업]- - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - 
x_train = x_train.reshape(14447, 4, 2)              
x_test = x_test.reshape(6193, 4, 2)

# print(x_train.shape)  # (14447, 2, 2, 2)  <-- "2, 2 ,2"는 input_shape값
# print(x_test.shape)   # (6193, 2, 2, 2)
#- - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - 



 #2. 모델구성
model = Sequential()
model.add(Conv1D(filters=50, kernel_size=(1),  
                 input_shape=(4, 2)))     #(batsh_size, row, columns, channels)
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

model.add(Flatten())    
model.add(Dense(128, activation='relu'))
model.add(Dropout(0.2))
model.add(Dense(128, activation='relu'))
model.add(Dense(1, activation='linear'))
# model.summary()


#3. 컴파일. 훈련
model.compile(loss='mse', optimizer='adam')

from tensorflow.python.keras.callbacks import EarlyStopping
earlyStopping = EarlyStopping(monitor='val_loss', patience=100, mode='min', verbose=1,
                              restore_best_weights=True) 
  
start_time = time.time()
model.fit(x_train, y_train, epochs=10, batch_size=50, verbose=1, validation_split=0.2,
                 callbacks=[earlyStopping])  
end_time = time.time() -start_time

#4. 평가, 예측
loss = model.evaluate(x_test, y_test)
print('loss : ', loss)

y_predict = model.predict(x_test)

from sklearn.metrics import r2_score
r2 = r2_score(y_test, y_predict)   # 0 ~ 1까지 지표를 나타낸다 (1에 가까우면 가장 좋다 (1에 마출 것.))
print('r2스코어 : ', r2)

print("걸린시간 : ", end_time)

# loss :  0.3459088206291199
# r2스코어 :  0.7389508134480736
# 걸린시간 :  8.405164957046509