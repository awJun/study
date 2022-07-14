from tensorflow.python.keras.layers import Dense, Conv2D, Flatten, MaxPooling2D, Dropout, LSTM 
from tensorflow.python.keras.models import Sequential
import pandas as pd
import numpy as np
import tensorflow as tf       
import time
from sklearn.model_selection import train_test_split # 함수 가능성이 높음
from sklearn.preprocessing import MinMaxScaler, StandardScaler # 클래스 가능성이 높음
from sklearn.preprocessing import MaxAbsScaler, RobustScaler

from sklearn.datasets import load_digits

import tensorflow as tf
tf.random.set_seed(66)  # 텐서플로우의 난수를 66으로 넣어서 사용하겠다. weight의 난수
                        # 텐서플로우의 데이터의 난수 

datasets = load_digits()
x = datasets.data 
y = datasets.target

 
x_train, x_test, y_train, y_test = train_test_split(x, y,
                                                    train_size=0.8,
                                                    shuffle=True,
                                                    random_state=100
                                                    )

#--[해당 데이터의 unique형태 확인]- - - - - - - - - - - - - - - - - 
# print(np.unique(x_train, return_counts=True))
# print(np.unique(y_train, return_counts=True))    # <--- 이 항목은 확인 필수   다중분류 모델이다.
# print(np.unique(x_test, return_counts=True))   
# print(np.unique(y_test, return_counts=True))
#- - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - 
#--[해당 데이터 shape 확인]- - - - - - - - - - - - - - - - - - - - -
# print(x_train.shape)   # (1437, 64)
# print(y_train.shape)   # (1437,)
# print(x_test.shape)    # (360, 64)
# print(y_test.shape)    # (360,)
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
x_train = x_train.reshape(1437, 32, 2)              
x_test = x_test.reshape(360, 32, 2)

# print(x_train.shape)  # (1437, 32, 2, 1)
# print(x_test.shape)   # (360, 32, 2, 1)
#- - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
#- -[y데이터를 x와 동일한 차원 형태로 변환] - - - - - - - - - - - - - - - - - - - - ( [중요]!! 회귀형에서는 할 필요없음  )
from tensorflow.python.keras.utils.np_utils import to_categorical
y_train = to_categorical(y_train)
y_test = to_categorical(y_test)
# print(y_train.shape, y_test.shape) # (1437, 10) (360, 10)
#----------------------------------------------------------------------------------

#2. 모델구성
model = Sequential()
# #2. 모델구성
model = Sequential()
model.add(LSTM(units=100 ,input_length=32, input_dim=2))   #SimpleRNN  또는 LSTM를 거치면 3차원이 2차원으로 변형되어서 다음 레어어에 간다.
# model.add(SimpleRNN(10))       # <-- 3차원이 아니라 2차원을 넣어서 에러가 발생함.[참고]
model.add(Dense(300, activation='relu'))
model.add(Dense(300, activation='relu'))
model.add(Dense(300, activation='relu'))
model.add(Dense(100, activation='relu'))   
model.add(Dense(128, activation='relu'))
model.add(Dropout(0.2))
model.add(Dense(128, activation='relu'))
model.add(Dense(10, activation='softmax')) # sigmoid에선 output은 1이다. (softmax에서는 유니크 갯수만큼)
model.summary()


#3. 컴파일. 훈련
model.compile(loss='categorical_crossentropy', optimizer='adam',
              metrics=['accuracy']) 


from tensorflow.python.keras.callbacks import EarlyStopping
earlyStopping = EarlyStopping(monitor='val_loss', patience=100, mode='min', verbose=1,
                              restore_best_weights=True) 
  
start_time = time.time()  
hist = model.fit(x_train, y_train, epochs=100, batch_size=100,
                 verbose=1,
                 validation_split=0.2,
                 callbacks=[earlyStopping])  
end_time = time.time() -start_time  

#4. 평가, 예측

loss, acc = model.evaluate(x_test, y_test)  # loss acc 각각 다른 리스트로 출력
                                            # loss = loss / acc = metrics에서 나온 accuracy 값
print('loss : ', loss)
print('acc : ', acc)


from sklearn.metrics import accuracy_score
y_predict = model.predict(x_test)

y_predict = np.argmax(y_predict, axis=1)
# print(y_predict)
y_test = np.argmax(y_test, axis=1)
# print(y_test)

acc = accuracy_score(y_test, y_predict)
print('accuracy : ', acc)

print("걸린시간 : ", end_time)


# loss :  0.7688674926757812
# accuracy :  0.8638888888888889
# 걸린시간 :  34.02174234390259
