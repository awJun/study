"""=[ Dropout 사용 ]==============================================================================================

from tensorflow.python.keras.layers import Dense, Dropout #(데이터의 노드를 중간중간 날려줌)
                                                 # 데이터가 많을수록 성능 좋음.
                    
model = Sequential()
model.add(Dense(100, input_dim=13))
model.add(Dropout(0.3))      # 30%만큼 노드를 지워버리겠다 (이빨을 빼버리겠다.)
model.add(Dense(50, activation='relu'))
model.add(Dropout(0.2))       # 20%만큼 노드를 지워버리겠다 (이빨을 빼버리겠다.)
model.add(Dense(16, activation='relu'))
model.add(Dense(1))

# 노드를 중간중간 날려도 평가, 예측에서는 전체 노드가 다 적용된다.                    
                                                  
========================================================================================================================
"""   
# 12개 만들고... 최적의 weight 가중치 파일을 저장할 것import numpy as np

import time
 
from sklearn.datasets import load_boston
from sklearn.model_selection import train_test_split # 함수 가능성이 높음
from sklearn.preprocessing import MinMaxScaler, StandardScaler # 클래스 가능성이 높음
from sklearn.preprocessing import MaxAbsScaler, RobustScaler
 
from tensorflow.python.keras.models import Sequential
from tensorflow.python.keras.layers import Dense, Dropout #(데이터의 노드를 중간중간 날려줌)
                                                 # 데이터가 많을수록 성능 좋음.
# 아래 두 가지를 알아보고 12개에 적용

datasets = load_boston()
x = datasets.data 
y = datasets.target

# print(np.min(x))  # x의 최소값이 출력된다.
# #   x의 최소값 = 0.0 
# print(np.max(x))  # x의 최소값이 출력된다.
# #   x의 최대값 = 711.0

# x = (x - np.min(x)) / (np.max(x) - np.min(x))
# # MinMaxScaler 표준편차 공식: [(x - 최소값) /(나누기) 최대값 - 최소값]
# # 위에 공식대로 해야 1이 나온다.
# print(x[:10])
 
x_train, x_test, y_train, y_test = train_test_split(x, y,
                                                    train_size=0.7,
                                                    random_state=100,
                                                    )
 
scaler =  MinMaxScaler()
# scaler = StandardScaler()
# scaler = MaxAbsScaler()
# scaler = RobustScaler()

scaler.fit(x_train)
x_train = scaler.transform(x_train) # x_train을 수치로 변환해준다.
x_test = scaler.transform(x_test) # 


 #2. 모델구성
model = Sequential()
model.add(Dense(100, input_dim=13))
model.add(Dropout(0.3))      # 30%만큼 노드를 지워버리겠다 (이빨을 빼버리겠다.)
model.add(Dense(50, activation='relu'))
model.add(Dropout(0.2))       # 20%만큼 노드를 지워버리겠다 (이빨을 빼버리겠다.)
model.add(Dense(16, activation='relu'))
model.add(Dense(1))

# 평가, 예측에서는 전체 노드가 다 적용된다.




#3. 컴파일. 훈련
model.compile(loss='mae', optimizer='adam')

from tensorflow.python.keras.callbacks import EarlyStopping, ModelCheckpoint
import datetime
date = datetime.datetime.now()
date = date.strftime("%m%d_%H%M") # 0707_1723
print(date)

filepath = './_ModelCheckPoint/'
filename = '{epoch:04d}-{val_loss:.4f}.hdf5'

earlyStopping = EarlyStopping(monitor='val_loss', patience=10, mode='auto', verbose=1, 
                              restore_best_weights=True)        

# mcp = ModelCheckpoint(monitor='val_loss', mode='auto', verbose=1, save_best_only=True, 
#                       filepath= "".join([filepath, 'k24_', date, '_', filename])
#                       )

hist = model.fit(x_train, y_train, epochs=100, batch_size=100,
                 validation_split=0.2,
                 callbacks=[earlyStopping, mcp],
                 verbose=1)




#4. 평가, 예측
loss = model.evaluate(x_test, y_test)

y_predict = model.predict(x_test)

from sklearn.metrics import r2_score
r2 = r2_score(y_test, y_predict)   

print('loss : ', loss)
print('r2스코어 : ', r2)

 
#########################################################
"""
Dropout 사용안함

# loss :  3.664133071899414
# r2스코어 :  0.6174005547393366 
"""
#########################################################
"""
Dropout 사용

# loss :  4.155057907104492
# r2스코어 :  0.5597031355739148
"""  
#########################################################
 

 