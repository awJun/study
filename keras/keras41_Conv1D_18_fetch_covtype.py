from tensorflow.python.keras.layers import Dense, Conv2D, Flatten, MaxPooling2D, Dropout, Conv1D   
from tensorflow.python.keras.models import Sequential
import pandas as pd
import numpy as np
import tensorflow as tf       
import time
from sklearn.datasets import fetch_covtype
from sklearn.model_selection import train_test_split # 함수 가능성이 높음
from sklearn.preprocessing import MinMaxScaler, StandardScaler # 클래스 가능성이 높음
from sklearn.preprocessing import MaxAbsScaler, RobustScaler
from sklearn.metrics import r2_score, accuracy_score

#1. 데이터
datasets = fetch_covtype()
x = datasets.data
y = datasets.target

x_train, x_test, y_train, y_test = train_test_split(x,y,
                                                    train_size=0.7,
                                                    random_state=66
                                                    )

#--[해당 데이터의 unique형태 확인]- - - - - - - - - - - - - - - - - 
# print(np.unique(x_train, return_counts=True))
# print(np.unique(y_train, return_counts=True))    # <--- 이 항목은 확인 필수 
  # (array([1, 2, 3, 4, 5, 6, 7]), array([148212, 198518,  24963,   1947,   6665,  
  #                                       12203,  14200], 다중분류 모델이다.
# print(np.unique(x_test, return_counts=True))   
# print(np.unique(y_test, return_counts=True))
#- - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - 
#--[해당 데이터 shape 확인]- - - - - - - - - - - - - - - - - - - - -
# print(x_train.shape)   # (406708, 54)
# print(y_train.shape)   # (406708,)
# print(x_test.shape)    # (174304, 54)
# print(y_test.shape)    # (174304,)
# - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
#--[ 스케일러 작업]- - - - - - - - - - - - - - - -(중간값을 찾아주는 역할 (값들의 차이 완화)
# scaler =  MinMaxScaler()
scaler = StandardScaler()
# scaler = MaxAbsScaler()
# scaler = RobustScaler()

scaler.fit(x_train)
x_train = scaler.transform(x_train) # x_train을 수치로 변환해준다.
x_test = scaler.transform(x_test) # 
#- - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
#--[차원 변형 작업]- - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - 
x_train = x_train.reshape(406708, 18, 3)               
x_test = x_test.reshape(174304, 18, 3)

# print(x_train.shape)  # (406708, 18, 3, 1)     <-- "32, 2 ,1"는 input_shape값
# print(x_test.shape)   # (174304, 18, 3, 1)
#- - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
#- -[y데이터를 x와 동일한 차원 형태로 변환] - - - - - - - - - - - - - - - - ( [중요]!! 회귀형에서는 할 필요없음  )
from tensorflow.python.keras.utils.np_utils import to_categorical
y_train = to_categorical(y_train)
y_test = to_categorical(y_test)
print(y_train.shape, y_test.shape) # (406708, 8) (174304, 8)


#2. 모델구성
model = Sequential()
model.add(Conv1D(filters=50, kernel_size=(2),  
                 input_shape=(18, 3)))     #(batsh_size, row, columns, channels)
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
model.add(Dense(8, activation='softmax')) # sigmoid에선 output은 1이다. (softmax에서는 유니크 갯수만큼)
# model.summary()

#3. 컴파일 훈련

model.compile(loss='categorical_crossentropy', optimizer='adam', # 다중 분류에서는 로스함수를 'categorical_crossentropy' 로 써준다 (99퍼센트로)
              metrics=['accuracy'])

from tensorflow.python.keras.callbacks import EarlyStopping
es = EarlyStopping(monitor='val_loss', patience=1000, mode='auto', verbose=1, 
                              restore_best_weights=True)   
start_time = time.time()
model.fit(x_train, y_train, epochs=1, batch_size=100,
                 validation_split=0.2,
                 callbacks=[es],
                 verbose=1)
end_time = time.time() -start_time

#4. 평가, 예측
loss, acc= model.evaluate(x_test, y_test)
print('loss : ', loss)
print('accuracy : ', acc)

# results= model.evaluate(x_test, y_test)
# print('loss : ', results[0])
# print('accuracy : ', results[1])


y_predict = model.predict(x_test)
print(y_predict)
print(y_test)
y_predict = np.argmax(y_predict, axis= 1)  # 판다스 겟더미 쓸때는 tf.argmax sklearn 원핫인코딩 쓸때는 np
print(y_predict)
y_test = np.argmax(y_test, axis= 1)
print(y_test)
# y_predict = to_categorical(y_predict)
# y_test = np.argmax(y_test, axis= 1)
print(np.unique(y_predict))
print(np.unique(y_test))



acc= accuracy_score(y_test, y_predict)
print('acc스코어 : ', acc) 

print(" end_time : ", end_time)

# loss :  0.5591983199119568
# acc스코어 :  0.7578827795116578
#  end_time :  36.481510400772095