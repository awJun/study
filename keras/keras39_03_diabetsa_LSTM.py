from tensorflow.python.keras.layers import Dense, Conv2D, Flatten, MaxPooling2D, Dropout, LSTM 
from tensorflow.python.keras.models import Sequential
import pandas as pd
import numpy as np
import tensorflow as tf       
import time
from sklearn.datasets import load_diabetes
from sklearn.model_selection import train_test_split # 함수 가능성이 높음
from sklearn.preprocessing import MinMaxScaler, StandardScaler # 클래스 가능성이 높음
from sklearn.preprocessing import MaxAbsScaler, RobustScaler
 

datasets = load_diabetes()
x = datasets.data 
y = datasets.target

 
x_train, x_test, y_train, y_test = train_test_split(x, y,
                                                    train_size=0.8,
                                                    shuffle=True,
                                                    random_state=100
                                                    )

#--[해당 데이터의 unique형태 확인]- - - - - - - - - - - - - - - - - 
# print(np.unique(x_train, return_counts=True))
# print(np.unique(y_train, return_counts=True))    # <--- 이 항목은 확인 필수   회귀 모델이다.
# print(np.unique(x_test, return_counts=True))   
# print(np.unique(y_test, return_counts=True))
#- - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - 
#--[해당 데이터 shape 확인]- - - - - - - - - - - - - - - - - - - - -
# print(x_train.shape)   # (353, 10)
# print(y_train.shape)   # (353,)
# print(x_test.shape)    # (89, 10)
# print(y_test.shape)    # (89,)
# - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -   
#--[스케일러를 사용하기 위해 차원 변형 작업]- - ( 데이터의 형태가 2x2가 아닐때만 사용할 것 )- - - -
#  ( 아래 스케일러들는 2x2형태에서만 돌아가기 때문에  (60000, 28, 28)형태를 2x2로 변환하는 작업임 )

#[사용 안함]
# x_train = x_train.reshape(16512, 4, 2)              
# x_test = x_test.reshape(4128, 4, 2)

# # print(x_train.shape)  # (16512, 4, 2)
# # print(x_test.shape)   # (4128, 4, 2)
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
x_train = x_train.reshape(353, 5, 2)              
x_test = x_test.reshape(89, 5, 2)

print(x_train.shape)  # (16512, 4, 2)
print(x_test.shape)   # (4128, 4, 2)
#- - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
#- -[y데이터를 x와 동일한 차원 형태로 변환] - - - - - - - - - - - - ( [중요]!! 회귀형에서는 할 필요없음  )
# from tensorflow.python.keras.utils.np_utils import to_categorical
# y_train = to_categorical(y_train)
# y_test = to_categorical(y_test)
# print(y_train.shape, y_test.shape) #(120, 3) (30, 3)   

                                          
                                                
#2. 모델구성 
model = Sequential()                            # input_shape=(3, 1) == input_length=3, input_dim=1)
# model.add(SimpleRNN(units=100,activation='relu' ,input_shape=(3, 1)))   # [batch, timesteps, feature]

model.add(LSTM(units=100 ,input_length=5, input_dim=2))   #SimpleRNN  또는 LSTM를 거치면 3차원이 2차원으로 변형되어서 다음 레어어에 간다.
# model.add(SimpleRNN(10))       # <-- 3차원이 아니라 2차원을 넣어서 에러가 발생함.[참고]
model.add(Dense(100, activation='relu'))
model.add(Dense(300, activation='relu'))
model.add(Dense(300, activation='relu'))
model.add(Dense(300, activation='relu'))
model.add(Dense(300, activation='relu'))
model.add(Dense(300, activation='relu'))
model.add(Dense(100, activation='relu'))
model.add(Dense(1))
model.summary()  # https://velog.io/@yelim421/%EC%88%9C%ED%99%98-%EC%8B%A0%EA%B2%BD%EB%A7%9D-Recurrent-Neural-NetworkRNN / Param가 120인 이유 연산과정



#3. 컴파일, 훈련
model.compile(loss='mse', optimizer='adam')

from tensorflow.python.keras.callbacks import EarlyStopping
earlyStopping = EarlyStopping(monitor='val_loss', patience=10, mode='auto', verbose=1,
                              restore_best_weights=True) 

model.fit(x_train, y_train, epochs=100, batch_size=32, verbose=1,   # batch_size의 디폴트값 = 32 즉 32이는 넣으나 마나임 ~
                 callbacks=[earlyStopping])  

#4. 평가, 예측
loss = model.evaluate(x_test, y_test)

y_predict = model.predict(x_test)

from sklearn.metrics import r2_score
r2 = r2_score(y_test, y_predict)   

print('loss : ', loss)
print('r2스코어 : ', r2)

# loss :  3366.887939453125
# r2스코어 :  0.3664150483864922
