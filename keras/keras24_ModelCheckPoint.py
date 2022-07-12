"""=[  ModelCheckpoint 설명 ]==============================================================================================

import datetime
date = datetime.datetime.now()
print(date) # 2022-07-07 17:24:51.433145  # 수치형 데이터이다.

date = date.strftime('%m%d_%H%M')  # %m : 월 / %d : 일 / %H : 시 / %M : 분
print(date) # 0707_1724            # 해석: 7월 7일 _ 17시 24분

filepath = './_ModelCheckPoint/k24'
filename = '{epoch:04d}-{val_loss:.4f}.hdf5'
#         {에포의 4자리}-{발로스의 소수점 4째자리} 라는 뜻

mcp = ModelCheckpoint(monitor='val_loss', mode='auto', verbose=1, # 가장 좋은 가중치 저장 위해 / mode가 모니터한 가장 최적 값, val 최저값, accuracy 최고값
                      save_best_only=True,
                      filepath= "".join([filepath, 'k24_', date, '_', filename] # .join안에 있는 모든 문자열을 합치겠다.
                      ))


===[ ModelCheckpoint 사용 ]==============================================================================================

mcp = ModelCheckpoint(monitor='val_loss', mode='auto', verbose=1, # 가장 좋은 가중치 저장 위해 / mode가 모니터한 가장 최적 값, val 최저값, accuracy 최고값
                      save_best_only=True,      # 모니터 후 가장 좋은 값을 저장
                      filepath='./_ModelCheckPoint/keras24_ModelCheckPoint.hdf5' # 가장 낮은 지점이 이 경로에 저장, 낮은 값이 나올 때마다 계속적으로 갱신하여 저장
                      )

# start_time = time.time()
hist = model.fit(x_train, y_train, epochs=1000, batch_size=20, 
                validation_split=0.2,
                callbacks=[earlyStopping, mcp], # 최저값을 체크해 반환해줌
                verbose=1)
# end_time = time.time()
'''

model = load_model('./_ModelCheckPoint/keras24_ModelCheckPoint.hdf5')  # mcp에서 val_loss가 가장 최저값 상태일때의
                                                                         가중치 값을 불러옴

========================================================================================================================
"""   

from sklearn.model_selection import train_test_split
from tensorflow.python.keras.models import Sequential, load_model
from tensorflow.python.keras.layers import Dense
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from sklearn.datasets import load_boston
import numpy as np
import time

#1. 데이터
datasets = load_boston()
x, y = datasets.data, datasets.target

x_train, x_test, y_train, y_test = train_test_split(x,y,
                                                    train_size=0.8,
                                                    random_state=66
                                                    )

scaler = MinMaxScaler()
x_train = scaler.fit_transform(x_train)
x_test = scaler.transform(x_test)

start_time = time.time()

#2. 모델구성
model = Sequential()
model.add(Dense(256, input_dim=13))
model.add(Dense(128, activation='relu'))
model.add(Dense(64, activation='relu'))
model.add(Dense(32, activation='relu'))
model.add(Dense(16, activation='relu'))
model.add(Dense(8, activation='relu'))
model.add(Dense(1))
model.summary()


#3. 컴파일, 훈련
model.compile(loss='mse', optimizer='adam')

from tensorflow.python.keras.callbacks import EarlyStopping, ModelCheckpoint
earlyStopping = EarlyStopping(monitor='val_loss', patience=300, mode='auto', verbose=1, 
                              restore_best_weights=True)        

mcp = ModelCheckpoint(monitor='val_loss', mode='auto', verbose=1, save_best_only=True, 
                      filepath='./_ModelCheckPoint/keras24_ModelCheckPoint.hdf5')

hist = model.fit(x_train, y_train, epochs=800, batch_size=100,
                 validation_split=0.2,
                 callbacks=[earlyStopping, mcp],
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

# loss :  9.568774223327637
# r2스코어 :  0.8855176121925235
# 걸린시간 :  20.815696001052856