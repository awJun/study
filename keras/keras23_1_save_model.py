import numpy as np
import pandas as pd
import time

from tensorflow.python.keras.models import Sequential
from tensorflow.python.keras.layers import Dense
from sklearn.model_selection import train_test_split
from sklearn.datasets import load_boston   # sklearn은 학습 예제가 많음

from sklearn.preprocessing import MinMaxScaler, StandardScaler
from sklearn.preprocessing import MaxAbsScaler, RobustScaler


#1. 데이터
datasets = load_boston()   
x = datasets.data    # 데이터가
y = datasets.target  # y에 들어간다.



x_train, x_test, y_train, y_test = train_test_split(x, y,
                                                    train_size=0.8,
                                                    shuffle=True,
                                                    random_state=12345678
                                                    )

# scaler =  MinMaxScaler()
scaler = StandardScaler()
# scaler = MaxAbsScaler()
# scaler = RobustScaler()

scaler.fit(x_train)
x_train = scaler.transform(x_train) # x_train을 수치로 변환해준다.
x_test = scaler.transform(x_test) # 

#2. 모델구성
model = Sequential()
model.add(Dense(64, input_dim=13))
model.add(Dense(32, activation ='relu'))
model.add(Dense(16, activation ='relu'))
model.add(Dense(8, activation ='relu'))
model.add(Dense(1))

model.summary()

model.save("./_save/keras23_1_save_model.h5")
# 모델에 관련 정보를 해당 경로에 세이브 해둠


"""
#3. 컴파일. 훈련
model.compile(loss='mse', optimizer='adam')

from tensorflow.python.keras.callbacks import EarlyStopping
earlyStopping = EarlyStopping(monitor='val_loss', patience=100, mode='min', verbose=1,
                              restore_best_weights=True) 

start_time = time.time()
model.fit(x_train, y_train, epochs=10000, batch_size=50, verbose=1, validation_split=0.2,
                 callbacks=[earlyStopping])  
end_time = time.time() -start_time

#4. 평가, 예측
loss = model.evaluate(x_test, y_test)
print('loss : ', loss)

"""


















