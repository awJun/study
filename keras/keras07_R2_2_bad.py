    # [실습 시작]
    # 1. 조건: R2 음수가 아닌 0.5 이하로 만들것 
    # 2. 조건: 데이터 건들지 말 것
    # 3. 레이어는 인풋 아웃풋 포함 7개 이상
    # 4. batch_size=1
    # 5. 히든레이어의 노드는 10개 이상 100개 이하
    # 6. train 70%
    # 7. epoch 100번 이상
    # 8. loss지표는 mse, mae

import numpy as np
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from sklearn.model_selection import train_test_split

#1. 데이터
x = np.array([1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20])
y = np.array([1, 2, 4, 3, 5, 7, 9, 8, 9, 6, 8, 12, 13, 14, 15, 16, 18, 17, 19, 20])

x_train, x_test, y_train, y_test = train_test_split(x, y,
                                                    train_size=0.7,
                                                    shuffle=True,
                                                    random_state=12345678
                                                    )

#2. 모델구성
model = Sequential()
model.add(Dense(50, input_dim=1))
model.add(Dense(98))
model.add(Dense(20))
model.add(Dense(30))
model.add(Dense(20))
model.add(Dense(20))
model.add(Dense(20))
model.add(Dense(20))
model.add(Dense(24))
model.add(Dense(30))
model.add(Dense(20))
model.add(Dense(20))
model.add(Dense(20))
model.add(Dense(24))
model.add(Dense(30))
model.add(Dense(20))
model.add(Dense(20))
model.add(Dense(20))
model.add(Dense(24))
model.add(Dense(30))
model.add(Dense(20))
model.add(Dense(20))
model.add(Dense(20))
model.add(Dense(24))
model.add(Dense(30))
model.add(Dense(20))
model.add(Dense(20))
model.add(Dense(20))
model.add(Dense(24))
model.add(Dense(30))
model.add(Dense(20))
model.add(Dense(20))
model.add(Dense(20))
model.add(Dense(24))
model.add(Dense(30))
model.add(Dense(20))
model.add(Dense(20))
model.add(Dense(24))
model.add(Dense(30))
model.add(Dense(20))
model.add(Dense(1))

#3. 컴파일. 훈련
model.compile(loss='mse', optimizer='adam')
model.fit(x_train, y_train, epochs=100, batch_size=1)

#4. 평가, 예측
loss = model.evaluate(x_test, y_test)
print('loss : ', loss)

y_predict = model.predict(x)


from sklearn.metrics import r2_score
r2 = r2_score(y, y_predict)   # 0 ~ 1까지 지표를 나타낸다 (1에 가까우면 가장 좋다 (1에 마출 것.))
print('r2스코어 : ', r2)

# loss :  38.0827522277832
# r2스코어 :  0.009018524311961484
# 레이어의 층을 늘리면 성능저하
# 훈련량을 바꾼다 

