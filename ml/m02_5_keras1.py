"""
[핵심]
model = Sequential()
model.add(Dense(1, activation='sigmoid')) 

이것과 

model = LinearSVC() / model = LinearSVR()

는 레이어가 1개라는 점에서 같은 것이다.

그걸 보여줄려는 취지로 이걸 만듬
"""

import numpy as np
from sklearn.metrics import accuracy_score
from sklearn.svm import LinearSVC
from sklearn.linear_model import Perceptron
from tensorflow.python.keras.models import Sequential
from tensorflow.python.keras.layers import Dense

#1. 데이터
x_data = [[0, 0], [0, 1], [1, 0], [1, 1]]
y_data = [0, 1, 1, 0]

#2. 모델구성
# model = Perceptron()
model = Sequential()
model.add(Dense(1, activation='sigmoid'))

#3. 훈련
# model.fit(x_data, y_data)
model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['acc'])
model.fit(x_data, y_data, batch_size=1, epochs=100)

#4. 평가
y_predict = model.predict(x_data)
print(x_data, "의 예측결과: ", y_predict)


results = model.evaluate(x_data, y_data)
print("model.score : ", results[1])

###[ 결과값 ]######################################################

# [[0, 0], [0, 1], [1, 0], [1, 1]] 의 예측결과: [[0.5113439 ]
#                                                [0.3017459 ]
#                                                [0.6007118 ]
#                                                [0.38320905]]

# model.score :  0.5

#####################################################################




# acc = accuracy_score(y_data, y_predict)
# print("accuracy_score : ",acc)

