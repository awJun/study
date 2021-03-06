from sklearn.metrics import r2_score, accuracy_score
from tensorflow.python.keras.layers import Dense, Conv2D, Flatten, MaxPooling2D ,Dropout   
from tensorflow.python.keras.models import Sequential
from keras.datasets import mnist,cifar10,cifar100,fashion_mnist
import pandas as pd
import numpy as np
import tensorflow as tf       
import time
from sklearn.model_selection import train_test_split # 함수 가능성이 높음
from sklearn.preprocessing import MinMaxScaler, StandardScaler # 클래스 가능성이 높음
from sklearn.preprocessing import MaxAbsScaler, RobustScaler
from sklearn.datasets import load_digits

#.1 데이터
datasets = load_digits()
x = datasets.data     # (1797, 64)
y = datasets.target   # (1797,)

x_train, x_test, y_train, y_test = train_test_split(x, y,
                                                    train_size=0.8,
                                                    shuffle=True,
                                                    random_state=100
                                                    )

#--[해당 데이터의 unique형태 확인]- - - - - - - - - - - - - - - - - 
# print(np.unique(x_train, return_counts=True))
# print(np.unique(y_train, return_counts=True))    # <--- 이 항목은 확인 필수 
  # (array([0, 1, 2, 3, 4, 5, 6, 7, 8, 9]), array([136, 150, 136, 151, 145, 147, 141, 140, 147, 144], dtype=int64)) 다중분류 모델이다.

# print(np.unique(x_test, return_counts=True))   
# print(np.unique(y_test, return_counts=True))
#- - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - 
#--[해당 데이터 shape 확인]- - - - - - - - - - - - - - - - - - - - -
# print(x_train.shape)   # (1437, 64)
# print(y_train.shape)   # (1437,)
# print(x_test.shape)    # (360, 64)
# print(y_test.shape)    # (360,)
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
x_train = x_train.reshape(1437, 32, 2, 1)               
x_test = x_test.reshape(360, 32, 2, 1)

print(x_train.shape)  # (1437, 32, 2, 1)     <-- "32, 2 ,1"는 input_shape값
print(x_test.shape)   # (360, 32, 2, 1)
#- - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
#- -[y데이터를 x와 동일한 차원 형태로 변환] - - - - - - - - - - - - - - - - - - ( [중요]!! 회귀형에서는 할 필요없음  )
from tensorflow.python.keras.utils.np_utils import to_categorical
y_train = to_categorical(y_train)
y_test = to_categorical(y_test)
print(y_train.shape, y_test.shape) # (1437, 10) (360, 10)


#2. 모델구성
model = Sequential()
model.add(Conv2D(filters=50, kernel_size=(1,1),  
                 input_shape=(32, 2, 1)))     #(batsh_size, row, columns, channels)
                                                                        # channels는 장수  / 1장 2장
model.add(Dropout(0.2))
model.add(Conv2D(64, (1, 1), padding='valid', activation='relu'))                         
model.add(Dropout(0.2))
model.add(Conv2D(32, (1, 1), padding='same', activation='relu'))
model.add(Dropout(0.2))
model.add(Conv2D(128, (1, 1), padding='valid', activation='relu'))                         
model.add(Dropout(0.2))
model.add(Conv2D(128, (1, 1), padding='same', activation='relu'))
model.add(Dropout(0.2))

model.add(Flatten())    
model.add(Dense(128, activation='relu'))
model.add(Dropout(0.2))
model.add(Dense(128, activation='relu'))
model.add(Dense(10, activation='softmax')) # sigmoid에선 output은 1이다. (softmax에서는 유니크 갯수만큼)
# model.summary()

#3. 컴파일. 훈련
model.compile(loss='categorical_crossentropy', optimizer='adam',
              metrics=['accuracy']) 

from tensorflow.python.keras.callbacks import EarlyStopping
earlyStopping = EarlyStopping(monitor='val_loss', patience=100, mode='auto', verbose=1,
                              restore_best_weights=True)

start_time = time.time()
hist = model.fit(x_train, y_train, epochs=10000, batch_size=100,
                 verbose=1,
                 validation_split=0.2,
                 callbacks=[earlyStopping])  
end_time = time.time() -start_time



#4. 평가, 예측

################################################################################
loss, acc = model.evaluate(x_test, y_test)  # loss acc 각각 다른 리스트로 출력
                                            # loss = loss / acc = metrics에서 나온 accuracy 값
print('loss : ', loss)
print('acc : ', acc)



result = model.evaluate(x_test, y_test)  # loss acc 각각 다른 리스트로 출력

from sklearn.metrics import accuracy_score
y_predict = model.predict(x_test)

y_predict = np.argmax(y_predict, axis=1)
# print(y_predict)

y_test = np.argmax(y_test, axis=1)   # 해당 리스트에서 값이 큰 스칼라의 인덱스 번호를 출력 
# print(y_test)

acc = accuracy_score(y_test, y_predict)
print('accuracy : ', acc)

print("걸린시간 : ", end_time)




