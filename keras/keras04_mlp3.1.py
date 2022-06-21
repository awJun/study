import numpy as np
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense

#1. 데이터
x = np.array([range(10), range(21, 31), range(201, 211)])  
 # range: 해당 범위 안의 정수를 담고 있는 함수 0 ~ 10-1
 # 여기서 range은 열로 사용한다. 그러므로 위에 x는 3차원 열을 가지고 있다.
 
# for i in range(10):
    # print(i)

print(x.shape) # (3, 10)

x = np.transpose(x) # (10, 3)
print(x.shape)

y = np.array([[1, 2, 3, 4, 5, 6, 7, 8, 9, 10],
             [1, 1.1, 1.2, 1.3, 1.4, 1.5, 1.6, 1.7, 1.8, 1.9]])


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
model.add(Dense(2)) # y의 백터가 두 개이므로 최종 y의 값도 두 개를 출력해야 한다.

#3. 컴파일, 훈련
model.compile(loss='mse', optimizer='adam')  
model.fit(x, y, epochs=100, batch_size=3)  

#4. 평가, 예측
loss = model.evaluate(x, y)
print('lose : ', loss)

result = model.predict([[9, 30, 210]])  
print('[9, 30, 210]의 예측값 : ', result)

# 예측 : [9, 30, 210]  -> 예상 y값 [[10, 1.9]]

# lose :  8.025224582097756e-12
# [9, 30, 210]의 예측값 :  [[9.999998  1.9000027]]



















