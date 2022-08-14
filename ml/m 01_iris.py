"""
[핵심]
# 통상적으로 컴퓨터가 좋으면 딥러닝 / 안좋으면 머신러닝을 활용한다.

머신러닝만 사용 할 것이므로 sklearn을 사용한다. 그러므로 tensorflow는 사용안할예정
머신러닝은 sklearn에 다 들어있다.

딥러닝은 레이어를 길게 뺀거
머신러닝은 간결해서 속도가 빠르다.

러닝머신은 원핫 할 필요없음 모델구성에서 알아서 받아짐
훈련에서 튜닝하고 평가할때 이벨류에이트없고 스코어를 사용한다.

LinearSVC
 - 분류모델에서 사용한다.
 - 하나의 레이어로 구성되어 있다. 

LinearSCR
 - 회기모델에서 사용한다.
 - 하나의 레이어로 구성되어 있다.

model.fit
 - model.fit(x_train, y_train)
 - 을 사용하면 fit 부분에서 컴파일까지 같이 자동으로 진행해줘서 여기서 fit과 compile이 같이된다.
 - 해당 방식은 러닝머신 모델에서만 사용이 가능하다.
 
model.score
 - results = model.score(x_test, y_test)  #분류 모델과 회귀 모델에서 score를 쓰면 알아서 자동으로 맞춰서 사용해준다. 
 - print("결과 acc : ", results)          # 회기는 r2 / 분류는 acc로 결과가 나온다.

[TMI]
러닝머신이 나온 이후 딥러닝이 나왔으므로 레이어에 대한 중요성을 몰랐을 때였다. 그때 만든 러닝머신 전용
모델인 LinearSVC, LinearSCR 는 레이어가 한 개인 모델로 만들어져있다. 이로 인해서 m03에서 배울 예정인
SVC, SCR이 만들어졌다. 이 모델은 레이어가 여러개이므로 m02의 Perceptron에서 해결못한 문제점을 해결했다.
"""


import numpy as np
import time
from sklearn.model_selection import train_test_split
from sklearn.datasets import load_iris
from sklearn.preprocessing import OneHotEncoder
# from tensorflow.python.keras.models import Sequential
# from tensorflow.python.keras.layers import Dense
# import tensorflow as tf
# tf.random.set_seed(66)  # 텐서플로우의 난수를 66으로 넣어서 사용하겠다. weight의 난수
#                         # 텐서플로우의 데이터의 난수
                        
##[ 머닝머신 ]##########################################################################################                    
from sklearn.datasets import load_iris
# from tensorflow.python.keras.models import Sequential
# from tensorflow.python.keras.layers import Dense
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import numpy as np
import pandas as pd
from tensorflow.python.keras.callbacks import EarlyStopping
import tensorflow as tf

from sklearn.svm import LinearSVC # 모델
# 서포트 벡터 머신 / 리니어 서포트 벡터 클레시파이어
# 원핫 x, 컴파일 x, argmax x

tf.random.set_seed(99)
# y = wx + b의 w값을 처음 랜덤으로 긋는 것을 어떻게 그을지 고정하고 시작

# 1. 데이터
datasets = load_iris()
print(datasets.DESCR)
'''
- class:
                - Iris-Setosa
                - Iris-Versicolour
                - Iris-Virginica
y값이 3개
이 3개 꽃 중 하나가 나와야 함
3중 분류
'''
print(datasets.feature_names)
x = datasets['data']
y = datasets['target']
print(x, '\n', y)
print(x.shape) # (150, 4)
print(y.shape) # (150,)
print('y의 라벨값: ', np.unique(y))

# sklearn에서는 인코딩 필요 없음
# y = pd.get_dummies(y)
# print(y.shape)
# print(y)

x_train, x_test, y_train, y_test = train_test_split(x, y, train_size=0.8,shuffle=True, random_state=9)
                      
#2. 모델구성
# model = Sequential()
# model.add(Dense(80, input_dim=4, activation='relu'))
# model.add(Dense(100))
# model.add(Dense(90))
# model.add(Dense(70, activation='relu'))
# model.add(Dense(50, activation='relu'))
# model.add(Dense(3, activation='softmax'))

model = LinearSVC()


#3. 컴파일, 훈련
# model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
# sklearn에서는 컴파일 없음
# Es = EarlyStopping(monitor='val_loss', mode='min', verbose=1, patience=50, restore_best_weights=True)
# log = model.fit(x_train, y_train, epochs=100, batch_size=100, callbacks=[Es], validation_split=0.2)

model.fit(x_train, y_train) # 컴파일도 포함되어 있음

#4. 평가, 예측
# loss, acc = model.evaluate(x_test, y_test)
# print('loss : ', loss)
# print('accuracy : ', acc)

# print(y_test)
# print(y_predict)

result = model.score(x_test, y_test) # evaluate 대신 score 사용
print('acc 결과: ', result) # 분류모델에서는 acc score // 회귀모델에서는 R2 score 자동으로 나옴

y_predict = model.predict(x_test)
# y_predict = tf.argmax(y_predict, axis=1)
# y_test = tf.argmax(y_test, axis=1)

acc_sc = accuracy_score(y_test, y_predict)
print('acc스코어 : ', acc_sc)

# loss :  0.005208590067923069
# accuracy :  1.0
# tf.Tensor([2 1 2 2 1 0 0 0 1 0 0 1 1 1 0 1 0 1 2 0 0 0 2 0 2 1 0 2 0 2], shape=(30,), dtype=int64)
# tf.Tensor([2 1 2 2 1 0 0 0 1 0 0 1 1 1 0 1 0 1 2 0 0 0 2 0 2 1 0 2 0 2], shape=(30,), dtype=int64)
# acc스코어 :  1.0

# 머신러닝 LinearSVC
# 결과:  1.0
# acc스코어 :  1.0
# 아주 빠름, 단층레이어임

