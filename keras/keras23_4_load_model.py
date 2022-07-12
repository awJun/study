import numpy as np
import pandas as pd
import time

from tensorflow.python.keras.models import Sequential, load_model
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

start_time = time.time()

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

model = load_model("./_save/keras23_3_save_model.h5")
# [중요] 이미 저장된 상태에서 모델 아래쪽에 위치시키고 컴파일, 훈련을 적용시켜주면
# 저장된 파일에서 새로운 가중치(랜덤)로 값 갱신

#3. 컴파일, 훈련
model.compile(loss='mse', optimizer='adam')

from tensorflow.python.keras.callbacks import EarlyStopping
earlyStopping =EarlyStopping(monitor='val_loss', patience=10, mode='auto', verbose=1, 
                             restore_best_weights=True) 

import time
start_time = time.time()
hist = model.fit(x_train, y_train, epochs=100, batch_size=20, 
                validation_split=0.2,
                callbacks=[earlyStopping],
                verbose=1) #verbose=0 일때는 훈련과정을 보여주지 않음

end_time = time.time() - start_time

model.save("./_save/keras23_3_save_model.h5")   # 새로운 가중치 값이 들어감


model = load_model("./_save/keras23_3_save_model.h5") # 위에서 세이브한 모델을 바로 가져옴

#4. 평가, 예측
loss = model.evaluate(x_test, y_test)
print('loss : ', loss)

y_predict = model.predict(x_test)
from sklearn.metrics import r2_score
r2 = r2_score(y_test, y_predict)
print('r2 스코어 : ', r2)


# loss :  11.555453300476074
# r2스코어 :  0.8617486700876874
# 걸린시간 :  0.11219477653503418











