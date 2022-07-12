"""=[ sigmoid 사용방법 ]===================================================================================================

model.add(Dense(1, activation='sigmoid'))
model.compile(loss='binary_crossentropy', optimizer='adam',

# sigmoid를 사용하고 난 후 losss는 무조건 binary_crossentropy를 사용해야한다.

========================================================================================================================
"""   
 

import numpy as np
from sklearn.datasets import load_breast_cancer
from tensorflow.python.keras.models import Sequential
from tensorflow.python.keras.layers import Dense
import time
from sklearn.model_selection import train_test_split

#1. 데이터
datasets = load_breast_cancer()
# print(datasets)
# print(datasets.DESCR )
# # (569, 30)
print(datasets.feature_names)

x = datasets.data  # [data] = /data 안에 키 벨류가 있으므로 똑같은 형태다.
y = datasets.target 
print(x)
print(y)

# print(x.shape, y.shape) (569, 30) (569,)  y는 569개

x_train, x_test, y_train, y_test = train_test_split(x, y,
                                                    train_size=0.7,
                                                    shuffle=True,
                                                    random_state=100
                                                    )




#2. 모델구성
model = Sequential()
model.add(Dense(100, activation='linear', input_dim=30))
model.add(Dense(100, activation='sigmoid'))
model.add(Dense(100, activation='relu'))
model.add(Dense(100, activation='linear'))
model.add(Dense(1, activation='sigmoid')) # sigmoid의 output = 1개다 다중에선 유니크값 만큼 아웃풋

#3. 컴파일. 훈련
model.compile(loss='binary_crossentropy', optimizer='adam',
              metrics=['accuracy', 'mse']) # loss에 accuracy도 같이 보여달라고 선언한 것이다. 로스외에 다른 지표도 같이 출력해준다.
    # 여기서 mse는 회귀모델에서 사용하는 활성화 함수이므로 분류모델에서는 신용성은 없다. 
from tensorflow.python.keras.callbacks import EarlyStopping
earlyStopping = EarlyStopping(monitor='val_loss', patience=100, mode='min', verbose=1,
                              restore_best_weights=True) 
  
start_time = time.time()


hist = model.fit(x_train, y_train, epochs=10000, batch_size=50,
                 verbose=1,
                 validation_split=0.2,
                 callbacks=[earlyStopping])  


end_time = time.time()

#4. 평가, 예측
loss = model.evaluate(x_test, y_test)
print('loss : ', loss)

print("걸린시간 : ", end_time)

y_predict = model.predict(x_test)
y_predict = y_predict.round(0)

## 과제: 아래 accuracy 스코어 완성
from sklearn.metrics import r2_score, accuracy_score
acc = accuracy_score(y_test, y_predict)
print('accuracy : ', acc)

# # r2 = r2_score(y_test, y_predict)   # 0 ~ 1까지 지표를 나타낸다 (1에 가까우면 가장 좋다 (1에 마출 것.))

import matplotlib.pyplot as plt
plt.figure(figsize = (9,6))     # figsize(가로길이,세로길이)
plt.plot(hist.history['loss'], marker='.', color='red', label='loss')
plt.plot(hist.history['val_loss'], marker='.', color='blue', label='val_loss')
plt.grid(True, axis=('x'))                      # plt.grid(True)
plt.title('asaql')
plt.ylabel('loss')
plt.xlabel('epochs')
plt.legend(loc = 'upper right')   # 그래프가 없는쪽에 알아서 해준다 굳이 명시할 필요 없음
# plt.legend()
plt.show()


# loss :  0.09797556698322296
# 걸린시간 :  1656658585.353482
# r2스코어 :  0.592937553253402






