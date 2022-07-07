import numpy as np
import time


from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder

from sklearn.datasets import load_digits

from sklearn.preprocessing import MinMaxScaler, StandardScaler
from sklearn.preprocessing import MaxAbsScaler, RobustScaler


#.1 데이터
datasets = load_digits()
x = datasets.data     # (1797, 64)
y = datasets.target   # (1797,)


from tensorflow.keras.utils import to_categorical 
y = to_categorical(y)
# print(y)
# print("+++++++++++++++++++++++++")
# print(x)

# print(x.shape) # (1797, 64)
# print(y.shape)   # (1797, 10)



# x: 17  /  y: 2
# print("y의 라벨값 : ", np.unique(x))


x_train, x_test, y_train, y_test = train_test_split(x, y,
                                                    train_size=0.8,
                                                    shuffle=True,
                                                    random_state=100
                                                    )

scaler =  MinMaxScaler()
# scaler = StandardScaler()
# scaler = MaxAbsScaler()
# scaler = RobustScaler()

scaler.fit(x_train)
x_train = scaler.transform(x_train) # x_train을 수치로 변환해준다.
x_test = scaler.transform(x_test) # 
# print(np.min(x_train))   # 0.0
# print(np.max(x_train))   # 1.0000000000000002
# print(np.min(x_test))   # -0.06141956477526944
# print(np.max(x_test))   # 1.1478180091225068

#2. 모델구성
from tensorflow.python.keras.models import Sequential, Model
from tensorflow.python.keras.layers import Dense, Input
"""
### 기존 모델 ###

model = Sequential()
model.add(Dense(100, activation='relu', input_dim=64))
model.add(Dense(100, activation='relu'))
model.add(Dense(100, activation='relu'))
model.add(Dense(100, activation='relu'))
model.add(Dense(10, activation='softmax')) 
"""
### 새로운 모델 ###
input1 = Input(shape=(64,))   # 처음에 Input 명시하고 Input 대한 shape 명시해준다.
dense1 = Dense(100)(input1)   # Dense 구성을하고  node 값을 넣고 받아오고 싶은 변수 받아온다.
dense2 = Dense(100, activation = 'relu')(dense1)    # 받아온 변수를 통해 훈련의 순서를 사용자가 원하는대로 할 수 있다.
dense3 = Dense(100, activation = 'sigmoid')(dense2)
output1 = Dense(10, activation='softmax')(dense3)
model = Model(inputs=input1, outputs=output1) # 해당 모델의 input과 output을 설정한다.


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

mcp = ModelCheckpoint(monitor='val_loss', mode='auto', verbose=1, save_best_only=True, 
                      filepath= "".join([filepath, 'k24_', date, '_', filename])
                      )

hist = model.fit(x_train, y_train, epochs=100, batch_size=100,
                 validation_split=0.2,
                 callbacks=[earlyStopping, mcp],
                 verbose=1)


# #4. 평가, 예측

# ################################################################################
# loss, acc = model.evaluate(x_test, y_test)  # loss acc 각각 다른 리스트로 출력
#                                             # loss = loss / acc = metrics에서 나온 accuracy 값
# print('loss : ', loss)
# print('acc : ', acc)



# result = model.evaluate(x_test, y_test)  # loss acc 각각 다른 리스트로 출력

# from sklearn.metrics import accuracy_score
# y_predict = model.predict(x_test)

# y_predict = np.argmax(y_predict, axis=1)
# # print(y_predict)

# y_test = np.argmax(y_test, axis=1)   # 해당 리스트에서 값이 큰 스칼라의 인덱스 번호를 출력 
# # print(y_test)

# acc = accuracy_score(y_test, y_predict)
# print('accuracy : ', acc)

# print("걸린시간 : ", end_time)








