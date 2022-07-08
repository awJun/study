import numpy as np
import pandas as pd
import time

from tensorflow.python.keras.models import Sequential, load_model  
# load_model : 세이브 모델을 댕겨주는 역할
from tensorflow.python.keras.layers import Dense
from sklearn.model_selection import train_test_split
from sklearn.datasets import load_boston   # sklearn은 학습 예제가 많음

from sklearn.preprocessing import MinMaxScaler, StandardScaler
from sklearn.preprocessing import MaxAbsScaler, RobustScaler


#1. 데이터
datasets = load_boston()
x = datasets.data
y = datasets.target

x_train, x_test, y_train, y_test = train_test_split(x,y,
                                                    train_size=0.8,
                                                    random_state=66
                                                    )

scaler = MinMaxScaler()
scaler.fit(x_train)
x_train = scaler.transform(x_train)
x_test = scaler.transform(x_test)


#2. 모델구성

# model = Sequential()
# model.add(Dense(256, input_dim=13))
# model.add(Dense(128, activation='relu'))
# model.add(Dense(64, activation='relu'))
# model.add(Dense(32, activation='relu'))
# model.add(Dense(16, activation='relu'))
# model.add(Dense(8, activation='relu'))
# model.add(Dense(1))
# model.summary()


# model.save("./_save/keras23_1_save_model.h5")
model = load_model("./_save/keras23_1_save_model.h5")
# 해당 경로에 있는 저장된 모델과 가중치 값을 가져온다
# 해당 모델을 만들 때 나왔던 가중치 값이 그대로 저장되고 그 값을 가져옴
# 해당 경로에 있는 모델을 불러옴

model.summary()



#3. 컴파일, 훈련
model.compile(loss='mse', optimizer='adam')

start_time = time.time()

from tensorflow.python.keras.callbacks import EarlyStopping
earlyStopping = EarlyStopping(monitor='val_loss', patience=300, mode='auto', verbose=1, 
                              restore_best_weights=True)        

model.fit(x_train, y_train, epochs=3000, batch_size=100,
                 validation_split=0.2,
                 callbacks=[earlyStopping],
                 verbose=1)

end_time = time.time() - start_time

#4. 평가, 예측
loss = model.evaluate(x_test, y_test)
y_predict = model.predict(x_test)
from sklearn.metrics import r2_score
r2 = r2_score(y_test, y_predict)
print('loss : ' , loss)
print('r2스코어 : ', r2)
print("걸린시간 : ", end_time)


# loss :  9.543179512023926
# r2스코어 :  0.8858238373150263
# 걸린시간 :  11.340367555618286

















