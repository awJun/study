import numpy as np
import pandas as pd

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score, mean_squared_error 

#1. 데이터 
path = './_data/ddarung/'
train_set = pd.read_csv(path + 'train.csv', index_col=0)

tast_set = pd.read_csv(path + 'test.csv', index_col=0)

train_set = train_set.dropna()
      # (1328, 10)

x = train_set.drop(['count'], axis=1)
      # (1328, 9)
y = train_set['count']
      # (1328,)

x_train, x_test, y_train, y_test = train_test_split(x, y,
                                                    shuffle=True,
                                                    random_state=1,
                                                    train_size=0.7
                                                    )

#2. 모델구성
model = Sequential()
model.add(Dense(50, input_dim=9))
model.add(Dense(100))
model.add(Dense(100))
model.add(Dense(100))
model.add(Dense(50))
model.add(Dense(40))
model.add(Dense(60))
model.add(Dense(1))

#3. 컴파일, 훈련
model.compile(loss='mse', optimizer='adam')
model.fit(x_train, y_train, epochs=100, batch_size=1)

#4. 평가, 예측
loss = model.evalueate(x_test, y_test)
print("loss : ", loss)

y_predict = model.predict(x_test)

def RMSE(a, b):
      return np.np.sqrt(mean_squared_error(a, b))

rmse = RMSE(y_test, y_predict)



















