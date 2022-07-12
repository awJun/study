"""===[ scaler 종류 ]==================================================================================================================

from sklearn.preprocessing import MinMaxScaler, StandardScaler # 클래스 가능성이 높음
                                ,MaxAbsScaler, RobustScaler

scaler =  MinMaxScaler()
scaler = StandardScaler()
scaler = MaxAbsScaler()
scaler = RobustScaler()

-[ scaler 사용 ]- - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - 

scaler =  MinMaxScaler()

scaler.fit(x_train)                 # scaler에 x_train을 사용해서 훈련 받은 값이 저장됨
x_train = scaler.transform(x_train) # x_train을 기준으로 transform값을 틀어서
x_test = scaler.transform(x_test)   # x_

 fit 작업: 특성 열의 최소값과 최대값을 찾습니다(이 스케일링은 데이터 프레임 속성/열 각각에 대해
                                      별도로 적용됨을 염두에 두십시오)
                                      
-[ scaler 사용 ]- - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - 

scaler =  MinMaxScaler()

# x_train = scaler.fit_transform(x_train)
# x_test = scaler.transform(x_test)

# print(np.min(x_train))   # 0.0
# print(np.max(x_train))   # 1.0000000000000002
# print(np.min(x_test))   # -0.06141956477526944
# print(np.max(x_test))   # 1.1478180091225068

-[ scaler 사용 ]- - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - 

# MaxAbsScaler : 절대값이 0~1사이에 매핑되도록 한다. 즉 -1~1 사이로 재조정한다. 
# 양수 데이터로만 구성된 특징 데이터셋에서는 MinMaxScaler와 유사하게 동작하며, 큰 이상치에 민감할 수 있다.

# RobustScaler : 아웃라이어의 영향을 최소화한 기법이다. 중앙값(median)과 IQR(interquartile range)을 사용하기 때문에 
# StandardScaler와 비교해보면 표준화 후 동일한 값을 더 넓게 분포 시키고 있음을 확인 할 수 있다.

========================================================================================================================
"""

import numpy as np
import time
 
from sklearn.datasets import load_boston
from sklearn.model_selection import train_test_split # 함수 가능성이 높음
from sklearn.preprocessing import MinMaxScaler, StandardScaler # 클래스 가능성이 높음
from sklearn.preprocessing import MaxAbsScaler, RobustScaler
 
from tensorflow.python.keras.models import Sequential
from tensorflow.python.keras.layers import Dense

# 아래 두 가지를 알아보고 12개에 적용

datasets = load_boston()
x = datasets.data 
y = datasets.target
print(datasets.DESCR)
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
 
# scaler =  MinMaxScaler()
# scaler = StandardScaler()
# scaler = MaxAbsScaler()
scaler = RobustScaler()



scaler.fit(x_train)
x_train = scaler.transform(x_train) # x_train을 수치로 변환해준다.
x_test = scaler.transform(x_test) # 
# print(np.min(x_train))   # 0.0
# print(np.max(x_train))   # 1.0000000000000002
# print(np.min(x_test))   # -0.06141956477526944
# print(np.max(x_test))   # 1.1478180091225068
 
 
 
# 위에 3줄을 2줄로 줄임
# x_train = scaler.fit_transform(x_train)
# x_test = scaler.transform(x_test)



# #### [ 3가지 성능 비교 ] #####
# scaler 사용하기 전
# scaler =  MinMaxScaler()
# scaler = StandardScaler()


#  #2. 모델구성
# model = Sequential()
# model.add(Dense(300, input_dim=13))
# model.add(Dense(240))
# model.add(Dense(100))
# model.add(Dense(100))
# model.add(Dense(220))
# model.add(Dense(220))
# model.add(Dense(1))



# #3. 컴파일. 훈련
# model.compile(loss='mae', optimizer='adam')

# from tensorflow.python.keras.callbacks import EarlyStopping
# earlyStopping = EarlyStopping(monitor='val_loss', patience=100, mode='auto', verbose=1,
#                               restore_best_weights=True) 

# start_time = time.time()
# model.fit(x_train, y_train, epochs=10000, batch_size=100, verbose=1, validation_split=0.2,
#                  callbacks=[earlyStopping])  
# end_time = time.time() -start_time




# #4. 평가, 예측
# loss = model.evaluate(x_test, y_test)

# y_predict = model.predict(x_test)

# from sklearn.metrics import r2_score
# r2 = r2_score(y_test, y_predict)   

# print('loss : ', loss)
# print('r2스코어 : ', r2)
# print("걸린시간 : ", end_time)
 
#########################################################
"""
scaler 사용 안함

loss :  3.8592958450317383
r2스코어 :  0.6372426021180102        
걸린시간 :  37.72577524185181
"""
#########################################################
"""  
scaler = StandardScaler()

loss :  3.538557291030884
r2스코어 :  0.6525740215917608
걸린시간 :  6.980016231536865
""" 
#########################################################
"""
scaler =  MinMaxScaler()

loss :  3.4968221187591553
r2스코어 :  0.629320025789293
걸린시간 :  11.156143188476562
"""
#########################################################
"""
scaler = MaxAbsScaler()

loss :  3.5338006019592285
r2스코어 :  0.6566347177735201
걸린시간 :  6.393912315368652
"""
#########################################################
"""
scaler = RobustScaler()

loss :  3.4373526573181152
r2스코어 :  0.6522324456885593
걸린시간 :  8.193526268005371
"""  
#########################################################
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 