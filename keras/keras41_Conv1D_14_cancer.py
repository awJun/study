from tensorflow.python.keras.layers import Dense, Conv2D, Flatten, MaxPooling2D, Dropout, Conv1D   
from tensorflow.python.keras.models import Sequential
import pandas as pd
import numpy as np
import tensorflow as tf       
import time
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split # 함수 가능성이 높음
from sklearn.preprocessing import MinMaxScaler, StandardScaler # 클래스 가능성이 높음
from sklearn.preprocessing import MaxAbsScaler, RobustScaler
 
#1. 데이터
datasets = load_breast_cancer()
# print(datasets)
# print(datasets.DESCR )
# # (569, 30)
# print(datasets.feature_names)

x = datasets.data  # [data] = /data 안에 키 벨류가 있으므로 똑같은 형태다.
y = datasets.target 
# print(x)
# print(y)

# print(x.shape, y.shape) (569, 30) (569,)  y는 569개

x_train, x_test, y_train, y_test = train_test_split(x, y,
                                                    train_size=0.7,
                                                    shuffle=True,
                                                    random_state=100
                                                    )


#--[해당 데이터의 unique형태 확인]- - - - - - - - - - - - - - - - - 
# print(np.unique(x_train, return_counts=True))
# print(np.unique(y_train, return_counts=True))    # <--- 이 항목은 확인 필수 (array([0, 1]), array([143, 255], dtype=int64)) 이중분류 모델이다.
# print(np.unique(x_test, return_counts=True))   
# print(np.unique(y_test, return_counts=True))
#- - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - 
#--[해당 데이터 shape 확인]- - - - - - - - - - - - - - - - - - - - -
# print(x_train.shape)   # (398, 30)
# print(y_train.shape)   # (398,)
# print(x_test.shape)    # (171, 30)
# print(y_test.shape)    # (171,)
# - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
#--[ 스케일러 작업]- - - - - - - - - - - - - - - - - - - - - - - - -
# scaler =  MinMaxScaler()
scaler = StandardScaler()
# scaler = MaxAbsScaler()
# scaler = RobustScaler()

scaler.fit(x_train)
x_train = scaler.transform(x_train) # x_train을 수치로 변환해준다.
x_test = scaler.transform(x_test) # 
#- - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
#--[차원 변형 작업]- - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - ( [중요]!! 회귀형에서는 할 필요없음  )
x_train = x_train.reshape(398, 6, 5)              
x_test = x_test.reshape(171, 6, 5)

print(x_train.shape)  # (398, 6, 5, 1)  <-- "6, 5 ,1"는 input_shape값
print(x_test.shape)   # (171, 6, 5, 1)
#- - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -

# #2. 모델구성
model = Sequential()
model.add(Conv1D(filters=50, kernel_size=(2),  
                 input_shape=(6, 5)))     #(batsh_size, row, columns, channels)
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
model.add(Dense(1, activation='sigmoid')) # sigmoid에선 output은 1이다. (softmax에서는 유니크 갯수만큼)
# model.summary()


#3. 컴파일. 훈련
model.compile(loss='binary_crossentropy', optimizer='adam',
              metrics=['accuracy']) # loss에 accuracy도 같이 보여달라고 선언한 것이다. 로스외에 다른 지표도 같이 출력해준다.
    # 여기서 mse는 회귀모델에서 사용하는 활성화 함수이므로 분류모델에서는 신용성은 없다. 
from tensorflow.python.keras.callbacks import EarlyStopping
earlyStopping = EarlyStopping(monitor='val_loss', patience=100, mode='auto', verbose=1,
                              restore_best_weights=True) 
  

start_time = time.time()
hist = model.fit(x_train, y_train, epochs=100, batch_size=50,
                 verbose=1,
                 validation_split=0.2,
                 callbacks=[earlyStopping])  
end_time = time.time()



#4. 평가, 예측
loss = model.evaluate(x_test, y_test)
print('loss : ', loss)


# 평가지표는 정수형태로 만들어줘야 작동함
y_predict = model.predict(x_test)
y_predict = y_predict.round(0)   

from sklearn.metrics import r2_score, accuracy_score
acc = accuracy_score(y_test, y_predict)
print('accuracy : ', acc)

print("걸린시간 : ", end_time)

# loss :  0.07069073617458344
# accuracy :  0.9824561403508771
# 걸린시간 :  1657793821.7218528
