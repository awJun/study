"""
XOR
 - 리스트 안의 값들이 모두 동일하면 0 / 서로 다르면 1을 사용한다.
 
"""

import numpy as np
from sklearn.metrics import accuracy_score
from sklearn.svm import LinearSVC
from sklearn.linear_model import Perceptron

#1. 데이터
x_data = [[0, 0], [0, 1], [1, 0], [1, 1]]
y_data = [0, 1, 1, 0]

#2. 모델구성
model = LinearSVC()
model = Perceptron()

#3. 훈련
model.fit(x_data, y_data)

#4. 평가
y_predict = model.predict(x_data)
print(x_data, "의 예측결과: ", y_predict)


results = model.score(x_data, y_data)
print("model.score : ", results)

acc = accuracy_score(y_data, y_predict)
print('acc : ', acc)



