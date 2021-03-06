"""=[ r2_score 사용하는 이유 ]==============================================================================================

"loss의 수치: 평가수치"하나만 믿기에는 애매하므로 
 "r2_score의 수치: 모델이 예측한 값과 실제 값이 얼마나 정확한지 나타내주는 정확도 수치"
 를 사용해서 모델이 예측한 값이 얼마나 정확한지 수치로 확인한다.

r2_score : 결정계수

===[ r2_score 사용 ]==============================================================================================

from sklearn.metrics import r2_score    # 정확도 개념과 비슷하다.
r2 = r2_score(y, y_predict)   # 0 ~ 1까지 지표를 나타낸다 (1에 가까우면 가장 좋다 (1에 마출 것.))
print('r2스코어 : ', r2)

# loss :  0.4906355142593384
# r2스코어 :  0.948699448897541

========================================================================================================================
"""   

import numpy as np
from tensorflow.python.keras.models import Sequential
from tensorflow.python.keras.layers import Dense
from sklearn.model_selection import train_test_split

#1. 데이터
x = np.array([1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20])
y = np.array([1, 2, 4, 3, 5, 7, 9, 8, 9, 6, 8, 12, 13, 14, 15, 16, 18, 17, 19, 20])

x_train, x_test, y_train, y_test = train_test_split(x, y,
                                                    train_size=0.7,
                                                    shuffle=True,
                                                    random_state=100
                                                    )

#2. 모델구성
model = Sequential()
model.add(Dense(10, input_dim=1))
model.add(Dense(20))
model.add(Dense(20))
model.add(Dense(30))
model.add(Dense(20))
model.add(Dense(20))
model.add(Dense(30))
model.add(Dense(30))
model.add(Dense(1))

#3. 컴파일. 훈련
model.compile(loss='mse', optimizer='adam')
model.fit(x_train, y_train, epochs=30, batch_size=1)

#4. 평가, 예측
loss = model.evaluate(x_test, y_test)
print('loss : ', loss)

y_predict = model.predict(x)

from sklearn.metrics import r2_score
r2 = r2_score(y, y_predict)   # 0 ~ 1까지 지표를 나타낸다 (1에 가까우면 가장 좋다 (1에 마출 것.))
print('r2스코어 : ', r2)

# loss :  0.4906355142593384
# r2스코어 :  0.9486994488975419



