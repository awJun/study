"""
XOR
 - 리스트 안의 값들이 모두 동일하면 0 / 서로 다르면 1을 사용한다.
 
Perceptron의 한계점   # http://www.aistudy.com/neural/perceptron.htm    
퍼셉트론의 한계는 선형으로 분류를 할 수 있지만 XOR와 같이 선형 분류만가능하며 비선형 분류는
불가능하다는 점이다  

레이어가 1개여서 겨울이 왔다고함 그걸 해결하려고 다층 레이어를 활용하는 SVC, SVR가 만들어지고 이걸 사용함
 
"""

import numpy as np
from sklearn.metrics import accuracy_score
from sklearn.svm import LinearSVC, SVC
from sklearn.linear_model import Perceptron

#1. 데이터
x_data = [[0, 0], [0, 1], [1, 0], [1, 1]]
y_data = [0, 1, 1, 0]

#2. 모델구성
# model = LinearSVC()
# model = Perceptron()
model = SVC()

#3. 훈련
model.fit(x_data, y_data)

#4. 평가
y_predict = model.predict(x_data)
print(x_data, "의 예측결과: ", y_predict)


results = model.score(x_data, y_data)
print("model.score : ", results)

acc = accuracy_score(y_data, y_predict)
print('acc : ', acc)




