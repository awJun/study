"""=[ validation ]==============================================================================================

 #fit 단계에서 훈련을하면서 해당 데이터에게 답안지를 중간 중간에보여줌 이렇게 하면 성능이 좋을수도 안좋을수도 (튜닝 항목임 ㅋ)

"""   

import numpy as np
import pandas as pd

from tensorflow.python.keras.models import Sequential
from tensorflow.python.keras.layers import Dense

#1. 데이터
x_train = np.array(range(1, 11))
y_train = np.array(range(1, 11))

x_test = np.array([11, 12, 13])
y_test = np.array([11, 12, 13])

x_val = np.array([14, 15, 16]) # 검증
y_val = np.array([14, 15, 16])

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












