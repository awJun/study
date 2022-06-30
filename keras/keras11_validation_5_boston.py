import numpy as np
import pandas as ap
import matplotlib.pyplot as plt

from tensorflow.python.keras.models import Sequential
from tensorflow.python.keras.layers import Dense
from sklearn.model_selection import train_test_split

from sklearn.datasets import load_boston   # sklearn은 학습 예제가 많음

#1. 데이터
datasets = load_boston()   
x = datasets.data    # 데이터가
y = datasets.target  # y에 들어간다.


x_train, x_test, y_train, y_test = train_test_split(x, y,
                                                    train_size=0.7,
                                                    shuffle=True,
                                                    random_state=12345678
                                                    )


#2. 모델구성
model = Sequential()
model.add(Dense(100, input_dim=13))
model.add(Dense(100))
model.add(Dense(100))
model.add(Dense(1))

#3. 컴파일. 훈련
model.compile(loss='mae', optimizer='adam')
model.fit(x_train, y_train, epochs=100, batch_size=1)

#4. 평가, 예측
loss = model.evaluate(x_test, y_test)
print('loss : ', loss)

y_predict = model.predict(x_test)

from sklearn.metrics import r2_score
r2 = r2_score(y_test, y_predict)   
print('r2스코어 : ', r2)


# loss :  4.962865829467773
# r2스코어 :  0.6251081490148807

#  validation_split=0.2 결과
# loss :  4.4386515617370605
# r2스코어 :  0.6140832517852677

plt_scatter = plt.scatter(y_test, y_predict)
print(plt_scatter)

































































