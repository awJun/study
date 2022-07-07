from msilib.schema import MsiPatchHeaders
import numpy as np
import time

from tensorflow.python.keras.callbacks import EarlyStopping, ModelCheckpoint
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder
from sklearn.datasets import load_wine

from sklearn.preprocessing import MinMaxScaler, StandardScaler
from sklearn.preprocessing import MaxAbsScaler, RobustScaler


#.1 데이터
datasets = load_wine()
x, y = datasets.data, datasets.target

# x = datasets.data
# y = datasets.target


# print(x.shape)         # (178, 13)
# print(y.shape)         # (178,)
# print(np.unique(y, return_counts=True))    
# (array([0, 1, 2]), array([59, 71, 48], dtype=int64

from tensorflow.keras.utils import to_categorical 
y = to_categorical(y)
print(y)

x_train, x_test, y_train, y_test = train_test_split(x, y,
                                                    train_size=0.8,
                                                    shuffle=True,
                                                    random_state=100
                                                    )

scaler =  MinMaxScaler()
x_train = scaler.fit_transform(x_train) # x_train을 수치로 변환해준다.
x_test = scaler.transform(x_test) # 


# #2. 모델구성
# from tensorflow.python.keras.models import Sequential, Model
# from tensorflow.python.keras.layers import Dense, Input

# model = Sequential()
# model.add(Dense(100, activation='relu', input_dim=13))
# model.add(Dense(100, activation='relu'))
# model.add(Dense(100, activation='relu'))
# model.add(Dense(100, activation='relu'))
# model.add(Dense(3, activation='softmax')) 

# #3. 컴파일, 훈련
# model.compile(loss='categorical_crossentropy', optimizer='adam',
#               metrics=['accuracy']) 

# ##  ModelCheckpoint ##  모니터를한 후 값을 해당 경로에 저장해주는 것을 설정
# from tensorflow.python.keras.callbacks import EarlyStopping, ModelCheckpoint
# earlyStopping = EarlyStopping(monitor='val_loss', patience=100, mode='auto', verbose=1,
#                               restore_best_weights=True)

# mcp = ModelCheckpoint(monitor='val_loss', mode='auto', verbose=1,
#                       save_best_only=True,  # 가장 좋은 값을 세이브 할 것인가
#                       filepath='./_ModelCheckPoint/keras24_ModelCheckPoint.hdf5'   # 경로를 잡아줌
#                       )   # hdf5   /  h5  이름만 다를뿐 별 차이 없음 신경x


# start_time = time.time()
# hist = model.fit(x_train, y_train, epochs=1000, batch_size=1,
#                  verbose=1,
#                  validation_split=0.2,
#                  callbacks=[earlyStopping, mcp])  # mcp 추가함  
# end_time = time.time() -start_time


model = model_model('./_ModelCheckPoint/keras24_ModelCheckPoint.hdf5')

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

y_test = np.argmax(y_test, axis=1)
# print(y_test)

acc = accuracy_score(y_test, y_predict)
print('accuracy : ', acc)

print("걸린시간 : ", end_time)























































































