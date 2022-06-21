import numpy as np
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense

#1. 데이터
x = np.array([1, 2, 3, 4, 5, 6, 7, 8, 9, 10])
y = np.array([1, 2, 3, 4, 5, 6, 7, 8, 9, 10])

# 완벽하게 정제된 데이터이므로 평가와 예측을 제대로 할 수 없다.


# [과제] 넘파이 리스트의 슬라이싱!! 7:3으로 잘라라
# https://wikidocs.net/2849 슬라이싱 관련 링크
x_train = x[0:7]   
x_test = x[7:10]

y_train = y[0:7]
y_test = y[7:10]




# x_train = np.array([1, 2, 3, 4, 5, 6, 7])
# x_test = np.array([8, 9, 10])

# y_train = np.array([1, 2, 3, 4, 5, 6, 7])
# y_test = np.array([8, 9, 10])


#2. 모델구성
model = Sequential()
model.add(Dense(10, input_dim=1))
model.add(Dense(1))

#3. 컴파일, 훈련
model.compile(loss='mse', optimizer='adam')      # 70%를 훈련
model.fit(x_train, y_train, epochs=100, batch_size=1)

#4. 평가, 예측
loss = model.evaluate(x_test, y_test)  # x y 대신  % 30를 평가 예측
print('loss : ', loss)

result = model.predict([11])
print('[11]의 예측값 : ', result)
























































