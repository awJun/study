import numpy as np
import pandas as pd

from tensorflow.python.keras.models import Sequential
from tensorflow.python.keras.layers import Dense

#1. 데이터
x = np.array(range(1, 17))
y = np.array(range(1, 17))

x_train = x[0:10]
y_train = x[0:10]

x_test = x[10:13]
y_test = x[10:13]

x_val = x[13:17]
y_val = x[13:17]

print(x_train)
print(y_train)
print(x_test)
print(y_test)
print(x_val)
print(y_val)



'''


# x_train = np.array(range(1, 11))
# y_train = np.array(range(1, 11))

# x_test = np.array([11, 12, 13])
# y_test = np.array([11, 12, 13])

# x_val = np.array([14, 15, 16]) # 검증
# y_val = np.array([14, 15, 16])

# train:1~10     test:11~13      validation:14~16  총 1~16까지의 데이터

#2. 모델구성
model = Sequential()
model.add(Dense(5, input_dim=1))
model.add(Dense(1))


#3. 컴파일, 훈련
model.compile(loss="mse", optimizer='adam')
model.fit(x_train, y_train, epochs=100, batch_size=1, validation_data=(x_val, y_val))


#4. 평가, 예측
loss = model.evaluate(x_test, y_test)
print("loss : ", loss)

result = model.predict([17])
print("17의 예측값 : ", result)




'''










