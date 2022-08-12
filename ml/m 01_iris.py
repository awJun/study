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
from sklearn.model_selection import train_test_split
from sklearn.svm import LinearSVC
                     
#1. 데이터
datasets = load_iris()
print(datasets.DESCR)  #행(Instances): 150   /   열(Attributes): 4
print(datasets.feature_names)

x = datasets['data']  # .data와 동일 
y = datasets['target']  
print(x.shape)   # (150, 4)
print(y.shape)   # (150,)
print("y의 라벨값 : ", np.unique(y))  # 해당 데이터의 고유값을 출력해준다.

# from tensorflow.keras.utils import to_categorical   # python까지 넣으면 오류남

# # print(y.shape)  (150, 3)
# y = to_categorical(y)
# to_categorical를 사용하면 y의 라벨값의 갯수에 맞춰서 알아서 백터의 양을
# 만들어준다.

# print(y)


x_train, x_test, y_train, y_test = train_test_split(x, y,
                                                    train_size=0.8,
                                                    shuffle=True,
                                                    random_state=100
                                                    )
print(y_train)
print(y_test)



#2. 모델구성
# model = Sequential()
# model.add(Dense(100, activation='relu', input_dim=4))
# model.add(Dense(100, activation='relu'))
# model.add(Dense(100, activation='relu'))
# model.add(Dense(100, activation='relu'))
# model.add(Dense(3, activation='softmax'))  # 다중분류에선 마지막에 softmax를 사용
# # softmax를 사용하면 3개가 출력된다. 3개중 큰쪽으로 찾는다.
# # 다중분류 일 때는 최종 노드의 갯수는 y의 라벨의 갯 수 

#2. 모델구성
model = LinearSVC()  # DL과 다르게 단층 레이어  구성으로 연산에 걸리는 시간을 비교할 수 없다.


# #3. 컴파일. 훈련
# model.compile(loss='categorical_crossentropy', optimizer='adam',
#               metrics=['accuracy']) 
# categorical_crossentropy를 사용하면 훈련을 할 때 값을 0 과 1로 구분해서 fit을 진행한다.
# categorical_crossentropy는 다중분류에서 사용된다.
# 에러는 멈춘다.   /  버그는 잘돌아가나 값이 이상히게 나온다.
# https://cheris8.github.io/artificial%20intelligence/DL-Keras-Loss-Function/  [loss 관련 링크]

# from tensorflow.python.keras.callbacks import EarlyStopping
# earlyStopping = EarlyStopping(monitor='val_loss', patience=100, mode='min', verbose=1,
#                               restore_best_weights=True) 
  
  
# start_time = time.time()
# hist = model.fit(x_train, y_train, epochs=10, batch_size=100,
#                  verbose=1,
#                  validation_split=0.2,
#                  callbacks=[earlyStopping])  
# end_time = time.time()

model.fit(x_train, y_train)
# DL 과 ML의 흐름은 똑같다 데이터 전처리->모델 구성 ->훈련(fit에 컴파일이 포함되어있다.) ->평가,예측  

# #4. 평가, 예측

# ################################################################################
# loss, acc = model.evaluate(x_test, y_test)  # loss acc 각각 다른 리스트로 출력
#                                             # loss = loss / acc = metrics에서 나온 accuracy 값
# print('loss : ', loss)
# print('acc : ', acc)


# result = model.evaluate(x_test, y_test)  # loss acc 각각 다른 리스트로 출력

from sklearn.metrics import accuracy_score
y_predict = model.predict(x_test)

# y_predict = np.argmax(y_predict, axis=1)
# print(y_predict)

# y_test = np.argmax(y_test, axis=1)
# print(y_test)

acc = accuracy_score(y_test, y_predict)
print('accuracy : ', acc)

results = model.score(x_test, y_test)  #분류 모델과 회귀 모델에서 score를 쓰면 알아서 값이 나온다 
print("결과 acc : ", results)          # 회기는 r2 / 분류는 acc로 결과가 나온다.

# 딥러닝과 머신러닝 차이
# 딥러닝은 레이어를 길게 뺀거
# 머신러닝은 간결해서 속도가 빠르다.


# 원핫 할 필요없음 모델구성에서 알아서 받아짐
# 컴파일 없음 훈련도 x y만 하면 된다. fit에 컴파일이 아랑서 포함되어 있다 그러므로 컴파일이 없음
# 훈련에서 튜닝하고 평가랗때 이벨류에이트없고 스코어를 사용한다.

