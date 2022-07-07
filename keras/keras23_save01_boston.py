import numpy as np
import time
 
from sklearn.datasets import load_boston
from sklearn.model_selection import train_test_split # 함수 가능성이 높음
from sklearn.preprocessing import MinMaxScaler, StandardScaler # 클래스 가능성이 높음
from sklearn.preprocessing import MaxAbsScaler, RobustScaler
 


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
 


##### [ 3가지 성능 비교 ] #####
# scaler 사용하기 전
# scaler =  MinMaxScaler()
# scaler = StandardScaler()


 #2. 모델구성
from tensorflow.python.keras.models import Sequential, Model
from tensorflow.python.keras.layers import Dense, Input
"""
### 기존 모델 ###

model = Sequential()
model.add(Dense(300, input_dim=13))
model.add(Dense(240))
model.add(Dense(100))
model.add(Dense(100))
model.add(Dense(220))
model.add(Dense(220))
model.add(Dense(1))
"""

### 함수형 모델 ###
input1 = Input(shape=(13,))   # 처음에 Input 명시하고 Input 대한 shape 명시해준다.
dense1 = Dense(300)(input1)   # Dense 구성을하고  node 값을 넣고 받아오고 싶은 변수 받아온다.
dense2 = Dense(240, activation = 'relu')(dense1)    # 받아온 변수를 통해 훈련의 순서를 사용자가 원하는대로 할 수 있다.
dense3 = Dense(100, activation = 'sigmoid')(dense2)
dense4 = Dense(100, activation = 'sigmoid')(dense3)
dense5 = Dense(220, activation = 'sigmoid')(dense4)
dense6 = Dense(220, activation = 'sigmoid')(dense5)
output1 = Dense(1)(dense6)
model = Model(inputs=input1, outputs=output1) # 해당 모델의 input과 output을 설정한다.


#3. 컴파일. 훈련
model.compile(loss='mae', optimizer='adam')

from tensorflow.python.keras.callbacks import EarlyStopping
earlyStopping = EarlyStopping(monitor='val_loss', patience=100, mode='auto', verbose=1,
                              restore_best_weights=True) 

start_time = time.time()
model.fit(x_train, y_train, epochs=10000, batch_size=100, verbose=1, validation_split=0.2,
                 callbacks=[earlyStopping])  
end_time = time.time() -start_time

model.save("./_save/keras23_1_save_boston.h5")
 # .save를 사용하면 해당 모델의 "#1. 데이터 부분을 제외하고" #2. 모델구성과 #3.컴파일, 훈련
 #  의 정보가 저장된다. 만약 #2. 아래에서 .save를 사용하면 #2. 모델구성만 저장된다. 

#4. 평가, 예측
loss = model.evaluate(x_test, y_test)

y_predict = model.predict(x_test)

from sklearn.metrics import r2_score
r2 = r2_score(y_test, y_predict)   

print('loss : ', loss)
print('r2스코어 : ', r2)
print("걸린시간 : ", end_time)

#########################################################
"""   [best_scaler]

scaler = RobustScaler()

loss :  3.2150421142578125
r2스코어 :  0.6813759649907927
걸린시간 :  9.263357162475586
"""
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
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 