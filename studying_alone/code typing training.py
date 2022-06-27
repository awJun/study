# 데이콘 따릉이 문제풀이
import numpy as np
import pandas as pd

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score, mean_squared_error 

#1. 데이터
path = './_data/ddarung/'  
                                                                   
train_set = pd.read_csv(path + 'train.csv', index_col=0)  
                       # [1459 rows x 10 columns]                  
test_set = pd.read_csv(path + 'test.csv', index_col=0) 
                       # [715 rows x 9 columns]                          
test_set = test_set.fillna(method='ffill')                 
train_set = train_set.fillna(method='ffill') 
    # (1328, 10

x = train_set.drop(['count'], axis=1)  #                                                                          
y = train_set['count']  

x_train, x_test, y_train, y_test = train_test_split(x, y,
                                                    train_size=0.95,
                                                    shuffle=True,
                                                    random_state=100
                                                    )

#2. 모델구성
model = Sequential()
model.add(Dense(100, activation = 'swish', input_dim=9))
model.add(Dense(100, activation = 'swish'))
model.add(Dense(100, activation = 'swish'))
model.add(Dense(100, activation = 'swish'))
model.add(Dense(1))

#3. 컴파일, 훈련
model.compile(loss='mae', optimizer='adam')
model.fit(x_train, y_train, epochs=800               
          , batch_size=3)

#4. 결과, 예측
loss = model.evaluate(x_test, y_test)
print('lose : ', loss)

y_predict = model.predict(x_test)
def RMSE(a, b):
    return np.sqrt(mean_squared_error(a, b))

rmse = RMSE(y_test, y_predict)
print("RMSE : ", rmse)


#####################################################


y_summit = model.predict(test_set)
  
      
submissin_set = pd.read_csv(path + 'submission.csv', index_col=0)
   # print(submissin.shape)   # (715, 1)
submissin_set['count'] = y_summit
submissin_set.to_csv('./_data/ddarung/submission.csv', index = True)
      








