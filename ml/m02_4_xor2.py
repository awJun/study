"""
XOR
 - 리스트 안의 값들이 모두 동일하면 0 / 서로 다르면 1을 사용한다.
 
Perceptron의 한계점   # http://www.aistudy.com/neural/perceptron.htm    
퍼셉트론의 한계는 선형으로 분류를 할 수 있지만 XOR와 같이 선형 분류만가능하며 비선형 분류는
불가능하다는 점이다  

레이어가 1개여서 겨울이 왔다고함 그걸 해결하려고 다층 레이어를 활용하는 SVC, SVR가 만들어지고 이걸 사용함
 
"""

import numpy as np
from sklearn.svm import LinearSVC
from sklearn.svm import SVC
from sklearn.linear_model import Perceptron
from sklearn.metrics import accuracy_score

# 1. 데이터
x_data = [[0,0],[0,1][1,0],[1,1]]
y_data = [0,1,1,0] # xor: 같으면 0 다르면 1

# 2. 모델
# model = LinearSVC() 
# model = Perceptron()
model = SVC() # MLP 멀티 레이어 퍼셉트론 - xor 문제를 차원을 접는 것으로 해결함, 다층 퍼셉트론으로

# 3. 훈련
model.fit(x_data, y_data)

# 4. 평가, 예측
y_pred = model.predict(x_data)
print(x_data, '의 예측결과: ', y_pred)

result = model.score(x_data, y_data)
print('model.score: ', result)

acc = accuracy_score(y_data, y_pred)
print('accuracy_score: ', acc)