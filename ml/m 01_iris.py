import numpy as np

from tensorflow.python.keras.models import Sequential
from tensorflow.python.keras.layers import Dense
import time
from sklearn.model_selection import train_test_split
from sklearn.datasets import load_iris
from sklearn.preprocessing import OneHotEncoder
import tensorflow as tf
tf.random.set_seed(66)  # 텐서플로우의 난수를 66으로 넣어서 사용하겠다. weight의 난수
                        # 텐서플로우의 데이터의 난수
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
model = LinearSVC()


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

results = model.score(x_test, y_test)
print("결과 acc : ", results)   # 회기는 r2 / 분류는 acc로 결과가 나온다.

# 딥러닝과 머신러닝 차이
# 딥러닝은 레이어를 길게 뺀거
# 머신러닝은 간결해서 속도가 빠르다.


# 원핫 할 필요없음 모델구성에서 알아서 받아짐
# 컴파일 없음 훈련도 x y만 하면 된다. fit에 컴파일이 아랑서 포함되어 있다 그러므로 컴파일이 없음
# 훈련에서 튜닝하고 평가랗때 이벨류에이트없고 스코어를 사용한다.

