import numpy as np
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense

#1. 데이터
x = np.array([[1, 2, 3, 4, 5, 6, 7, 8, 9, 10],
              [1, 1, 1, 1, 2, 1.3, 1.4, 1.5, 1.6, 1.4],
              [9, 8, 7, 6, 5, 4, 3, 2, 1, 0]]
             )    # (2,10)  원래는 10행 2열의 데이터를 사용해야하지만 2행 10열인 데이터이므로
                  # 트랜스포스 하여서 사용

# 대활호 하나 = 리스트  

y = np.array([11, 12, 13, 14, 15, 16, 17, 18, 19, 20])  # (10, )

# 계산을 할 때 행렬 연산으로하므로  x, y의 리스트 모양을 같게 해야한다. 

# y=w(1)x + b(10) x 1번째 데이터와 y비교

print(x.shape)  # (2, 10)
print(y.shape)  # (10, )

# x = x.T             # 행과 열을 바꾼다
x = x.transpose()   # 행과 열을 바꾼다.
# x = x.reshape(10,2)   # 순서 유지
print(x)     
print(x.shape)  #(10, 2)


"""
 [숙제] 모델을 완성하시오
 예측 : [[10, 1.4, 0]] 
"""

model = Sequential()
model.add(Dense(5, input_dim=3))   
model.add(Dense(4))
model.add(Dense(3))
model.add(Dense(4))
model.add(Dense(1))

#3. 컴파일, 훈련
model.compile(loss='mse', optimizer='adam')  
model.fit(x, y, epochs=100, batch_size=3)  

#4. 평가, 예측
loss = model.evaluate(x, y)
print('lose : ', loss)

result = model.predict([[[10, 1.4, 0]]])  # (1, 3)
print('[10, 1.4, 0]의 예측값 : ', result)

# lose :  0.000930643524043262
# [10, 1.4, 0]의 예측값 :  [[20.007872]]

















