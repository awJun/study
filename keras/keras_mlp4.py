# mlp "m: 멀티 l: 레이어 p: 퍼셉트론" = "다층 퍼셉트론"

import numpy as np
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense

#1. 데이터
x = np.array([range(10)])  
 # range: 해당 범위 안의 정수를 담고 있는 함수 0 ~ 10-1
 # 여기서 range은 열로 사용한다. 그러므로 위에 x는 3차원 열을 가지고 있다.
 
# for i in range(10):
    # print(i)

print(x.shape) # (1, 10)

x = np.transpose(x) # (10, 1)
print(x.shape)

y = np.array([[1, 2, 3, 4, 5, 6, 7, 8, 9, 10],
             [1, 1.1, 1.2, 1.3, 1.4, 1.5, 1.6, 1.7, 1.8, 1.9],
             [9, 8, 7, 6, 5, 4, 3, 2, 1, 0]])
# y의 배열이 3개이므로 3차원 텐서로 분류한다.

y = np.transpose(y)
print(y.shape)

#2. 모델
# 예측 : [[9]]

model = Sequential()
model.add(Dense(5, input_dim=1))   # x가 하나 
model.add(Dense(20))
model.add(Dense(20))
model.add(Dense(180))
model.add(Dense(30))
model.add(Dense(80))
model.add(Dense(20))
model.add(Dense(50))
model.add(Dense(20))
model.add(Dense(30))
model.add(Dense(30))
model.add(Dense(20))
model.add(Dense(3)) # y가 둘


#3. 컴파일, 훈련
model.compile(loss='mse', optimizer='adam')  # optimizer loss에 좀 더 힘을 실어주는 역할
model.fit(x, y, epochs=500, batch_size=5)  

#4. 평가, 예측
loss = model.evaluate(x, y)
print('lose : ', loss)

result = model.predict([[9]])  # x의 마지막 값 9
# predict 앞에 (열, 컬럼, 픽터, 특성)과 형태가 동일해야 한다. 열우선
print('[9]의 예측값 : ', result)

# lose :  3.449319280722507e-13
# [9]의 예측값 :  [[9.9999981e+00 1.9000006e+00 9.2759728e-07]]

