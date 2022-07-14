from tensorflow.python.keras.layers import Dense, Conv2D, Flatten, MaxPooling2D, Dropout, Conv1D   
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
                                                    train_size=0.7,
                                                    shuffle=True,
                                                    random_state=72
                                                    )

#--[해당 데이터의 unique형태 확인]- - - - - - - - - - - - - - - - - 
# print(np.unique(x_train, return_counts=True))
# print(np.unique(y_train, return_counts=True))    # <--- 이 항목은 확인 필수 (유니크 값이 많으므로 분류형은 아님) 즉! 수치형이다.
# print(np.unique(x_test, return_counts=True))   
# print(np.unique(y_test, return_counts=True))
#- - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - 
#--[해당 데이터 shape 확인]- - - - - - - - - - - - - - - - - - - - -
# print(x_train.shape)   # (309, 10)
# print(y_train.shape)   # (309,)
# print(x_test.shape)    # (133, 10)
# print(y_test.shape)    # (133,)
#- - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
#--[ 스케일러 작업]- - - - - - - - - - - - - - - - - - - - - - - - -
# scaler =  MinMaxScaler()
scaler = StandardScaler()
# scaler = MaxAbsScaler()
# scaler = RobustScaler()

scaler.fit(x_train)
x_train = scaler.transform(x_train) # x_train을 수치로 변환해준다.
x_test = scaler.transform(x_test) # 
#- - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
#--[차원 변형 작업]- - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - 
x_train = x_train.reshape(309, 5, 2)              
x_test = x_test.reshape(133, 5, 2)

# print(x_train.shape)  # (309, 5, 2, 1)  <-- "5, 2 ,1"는 input_shape값
# print(x_test.shape)   # (133, 5, 2, 1)
#- - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - 



 #2. 모델구성
model = Sequential()
model.add(Conv1D(filters=50, kernel_size=(1),  
                 input_shape=(5, 2)))     #(batsh_size, row, columns, channels)
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


#3. 컴파일, 훈련
model.compile(loss='mae', optimizer='adam')
    
from tensorflow.python.keras.callbacks import EarlyStopping
earlyStopping = EarlyStopping(monitor='val_loss', patience=10, mode='auto', verbose=1, 
                              restore_best_weights=True) 
                              
start_time = time.time()
hist = model.fit(x_train, y_train, epochs=100, batch_size=50,
                 validation_split=0.2,
                 callbacks=[earlyStopping],
                 verbose=1)
end_time = time.time() -start_time


#4. 평가, 예측
y_predict = model.predict(x_test)

loss = model.evaluate(x_test, y_test)

from sklearn.metrics import r2_score
r2 = r2_score(y_test, y_predict)

print('loss : ', loss)
print('r2스코어 : ', r2)
print("걸린시간 : ", end_time)


# loss :  41.12067794799805
# r2스코어 :  0.573054398810408
# 걸린시간 :  3.574906587600708
