"""
output이 3개  <-- 출력 데이터도 3개가 출력된다.
"""

import numpy as np
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense

#1. 데이터
x = np.array([range(10), range(21, 31), range(201, 211)])
# print(range(10)) # range: 범위, 거리 0~10까지의 정수형 숫자 

#---[ 알고만 있기 ]------------------------------
# for i in range(10):          # for :반복하라 
#     print(i)
#---------------------------------------------

# print(x) # (3, 10)
# [[  0   1   2   3   4   5   6   7   8   9]
#  [ 21  22  23  24  25  26  27  28  29  30]
#  [201 202 203 204 205 206 207 208 209 210]]

print(x.shape) # (3, 10)

x = np.transpose(x) # (10, 3)
print(x.shape)

y = np.array([[1, 2, 3, 4, 5, 6, 7, 8, 9, 10],
             [1, 1.1, 1.2, 1.3, 1.4, 1.5, 1.6, 1.7, 1.8, 1.9],
             [9, 8, 7, 6, 5, 4, 3, 2, 1, 0]])
# y의 배열이 3개이므로 3차원 텐서로 분류한다.

y = np.transpose(y)
print(x.shape)

#2. 모델


model = Sequential()
model.add(Dense(5, input_dim=3))   
model.add(Dense(20))
model.add(Dense(20))
model.add(Dense(30))
model.add(Dense(20))
model.add(Dense(20))
model.add(Dense(30))
model.add(Dense(20))
model.add(Dense(20))
model.add(Dense(20))
model.add(Dense(3)) # y의 백터가 세 개이므로 최종 y의 값도 두 개를 출력해야 한다.

#3. 컴파일, 훈련
model.compile(loss='mse', optimizer='adam')  
model.fit(x, y, epochs=100, batch_size=3)  

#4. 평가, 예측
loss = model.evaluate(x, y)
print('lose : ', loss)

result = model.predict([[9, 30, 210]])  
print('[9, 30, 210]의 예측값 : ', result)

# 예측 : [9, 30, 210]  -> 예상 y값 [[10, 1.9, 0]]

# lose :  0.0020840694196522236
# [9, 30, 210]의 예측값 :  [[ 9.936024    1.9785168  -0.03407807]]


