import numpy as np
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense

#1. 데이터
x = np.array([[1, 2, 3, 4, 5, 6, 7, 8, 9, 10],
              [1, 1, 1, 1, 2, 1.3, 1.4, 1.5, 1.6, 1.4]]
             )    # (2,10)

# 대활호 하나 = 리스트  

y = np.array([11, 12, 13, 14, 15, 16, 17, 18, 19, 20])  # (10, )

# 계산을 할 때 행렬 연산으로하므로  x, y의 리스트 모양을 같게 해야한다. 

# y=w(1)x + b(10) x 1번째 데이터와 y비교

print(x.shape)  # (2, 10)
print(y.shape)  # (10, )

x = x.T             # 행과 열을 바꾼다
# x = x.transpose()   # 행과 열을 바꾼다.
# x = x.reshape(10,2)   # 순서 유지
print(x)     
print(x.shape)  #(10, 2)



# https://rfriend.tistory.com/289 전차행렬 처리방법 링크 "아래는 3가지 방법 주석"
#  np.dot(a.T, a)
#  np.dot(np.transpose(a), a
#  np.dot(np.swapaxes(a, 0, 1), a)



#2. 모델구성
model = Sequential()
model.add(Dense(5, input_dim=2))   # dim = 열의 갯수 = 차원의 갯수    (?,?) 컴마가 있으면 
model.add(Dense(4))
model.add(Dense(3))
model.add(Dense(4))
model.add(Dense(1))

#3. 컴파일, 훈련
model.compile(loss='mse', optimizer='adam')  # optimizer loss에 좀 더 힘을 실어주는 역할
model.fit(x, y, epochs=100, batch_size=3)  # batch_size가 3이므로 데이터 10개를 3, 3, 3, 1로 나눠서 처리

#4. 평가, 예측
loss = model.evaluate(x, y)
print('lose : ', loss)

result = model.predict([[10, 1.4]])  #값 20
# predict 앞에 (열, 컬럼, 픽터, 특성)과 형태가 동일해야 한다. 열우선
print('[10, 1.4]의 예측값 : ', result)







