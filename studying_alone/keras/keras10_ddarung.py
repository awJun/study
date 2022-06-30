import numpy as np
import pandas as pd

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score, mean_squared_error 


#1. 데이터
path = './_data/ddarung/'

train_set = pd.read_csv(path + 'train.csv', index_col=0)
      # train_sex.shape : [1459 rows x 10 columns]
test_set = pd.read_csv(path + 'test.csv', index_col=0)
      # test_set.shape : [715 rows x 9 columns]
      
train_set = train_set.dropna()
      # train_set.shape : [1328 rows x 10 columns]


x = train_set.drop(['count'], axis=1)
      # x.shape = [1328 rows x 9 columns]

y = train_set['count']
      # y.shape = [1328 rows x 1 columns]


x_train, x_test, y_train, y_test = train_test_split(x, y,
                                                    train_size=0.989,
                                                    shuffle=True,
                                                    random_state=100  # 100
                                                    )



#2. 모델구성
model = Sequential()
model.add(Dense(1, activation = 'swish', input_dim=9))
# model.add(Dense(1))


#3. 컴파일, 훈련
model.compile(loss='mae', optimizer='adam')
model.fit(x_train, y_train, epochs=11111, batch_size=3)

#4. 결과, 예측
loss = model.evaluate(x_test, y_test)
print("loss : ", loss)

y_predict = model.predict(x_test)


r2 = r2_score(y_test, y_predict)
print("스코어 : ", r2)

def RMSE(a, b):
    return np.sqrt(mean_squared_error(a, b))
rmse = RMSE(y_test, y_predict)
print("RMSE : ", rmse)


# loss :  15.917014122009277
# 스코어 :  0.8710191426334581
# RMSE :  21.636669808639










