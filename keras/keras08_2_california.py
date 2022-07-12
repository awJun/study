import numpy as np
from tensorflow.python.keras.models import Sequential
from tensorflow.python.keras.layers import Dense
from sklearn.model_selection import train_test_split

from sklearn.datasets import fetch_california_housing
datasets = fetch_california_housing()

x=datasets.data
y=datasets.target

# print(x)
# [[   8.3252       41.            6.98412698 ...    2.55555556
#     37.88       -122.23      ]
#  [   8.3014       21.            6.23813708 ...    2.10984183
#     37.86       -122.22      ]
#  [   7.2574       52.            8.28813559 ...    2.80225989
#     37.85       -122.24      ]
#  ...
#  [   1.7          17.            5.20554273 ...    2.3256351
#     39.43       -121.22      ]
#  [   1.8672       18.            5.32951289 ...    2.12320917
#     39.43       -121.32      ]
#  [   2.3886       16.            5.25471698 ...    2.61698113
#     39.37       -121.24      ]]

# print(y)
# [4.526 3.585 3.521 ... 0.923 0.847 0.894]

print(x.shape, y.shape)   # x (20640, 8) 열이 8개   /  y (20640,)  

print(datasets.feature_names)  
 # ['MedInc', 'HouseAge', 'AveRooms', 'AveBedrms', 'Population', 'AveOccup', 'Latitude', 'Longitude']

print(datasets.DESCR)

 # [실습시작]



x_train, x_test, y_train, y_test = train_test_split(x, y,
                                                    train_size=0.7,
                                                    shuffle=True,
                                                    random_state=30
                                                    )



#2. 모델구성
model = Sequential()
model.add(Dense(300, input_dim=8))
model.add(Dense(240))
model.add(Dense(100))
model.add(Dense(100))
model.add(Dense(220))
model.add(Dense(100))
model.add(Dense(100))
model.add(Dense(220))
model.add(Dense(220))
model.add(Dense(100))
model.add(Dense(100))
model.add(Dense(220))
model.add(Dense(220))
model.add(Dense(220))
model.add(Dense(1))

#3. 컴파일. 훈련
model.compile(loss='mae', optimizer='adam')
model.fit(x_train, y_train, epochs=100, batch_size=100)

#4. 평가, 예측
loss = model.evaluate(x_test, y_test)
print('loss : ', loss)

y_predict = model.predict(x_test)

from sklearn.metrics import r2_score
r2 = r2_score(y_test, y_predict)   
print('r2스코어 : ', r2)


# loss :  0.5735353231430054
# r2스코어 :  0.5503746979022464






























