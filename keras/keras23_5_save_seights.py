import numpy as np
import pandas as pd
import time

from tensorflow.python.keras.models import Sequential, load_model
from tensorflow.python.keras.layers import Dense
from sklearn.model_selection import train_test_split
from sklearn.datasets import load_boston   # sklearn은 학습 예제가 많음

from sklearn.preprocessing import MinMaxScaler, StandardScaler
from sklearn.preprocessing import MaxAbsScaler, RobustScaler


#1. 데이터
datasets = load_boston()
x = datasets.data
y = datasets.target

x_train, x_test, y_train, y_test = train_test_split(x,y,
                                                    train_size=0.8,
                                                    random_state=66
                                                    )

scaler = MinMaxScaler()
scaler.fit(x_train)
x_train = scaler.transform(x_train)
x_test = scaler.transform(x_test)

start_time = time.time()

#2. 모델구성
model = Sequential()
model.add(Dense(256, input_dim=13))
model.add(Dense(128, activation='relu'))
model.add(Dense(64, activation='relu'))
model.add(Dense(32, activation='relu'))
model.add(Dense(16, activation='relu'))
model.add(Dense(8, activation='relu'))
model.add(Dense(1))
model.summary()


# model.save("./_save/keras23_1_save_model.h5")

model.save_weights("./_save/keras23_5_save_weights1.h5")

# model = load_model("./_save/keras23_3_save_model.h5")



# 3. 컴파일, 훈련
model.compile(loss='mse', optimizer='adam')



from tensorflow.python.keras.callbacks import EarlyStopping
earlyStopping = EarlyStopping(monitor='val_loss', patience=300, mode='auto', verbose=1, 
                              restore_best_weights=True)        

model.fit(x_train, y_train, epochs=3000, batch_size=100,
                 validation_split=0.2,
                 callbacks=[earlyStopping],
                 verbose=1)



# model.save("./_save/keras23_3_save_model.h5")

model.save_weights("./_save/keras23_5_save_weights2.h5")

# model = load_model("./_save/keras23_3_save_model.h5")

end_time = time.time() - start_time

#4. 평가, 예측
loss = model.evaluate(x_test, y_test)
y_predict = model.predict(x_test)
from sklearn.metrics import r2_score
r2 = r2_score(y_test, y_predict)
print('loss : ' , loss)
print('r2스코어 : ', r2)
print("걸린시간 : ", end_time)


# model = load_model("./_save/keras23_3_save_model.h5")
      # 위에서 해당 경로의 모델을 가져와서 훈련을 시킬 경우 해당 모델에서
      # 저장된 가중치 값을 버리고 아래에서 훈련을 거쳐서 나온 가중치 값으로 된다. 
      
      
      
model.save_weights("keras23_5_save_weights1.h5")
      
# #3. 컴파일. 훈련
model.compile(loss='mse', optimizer='adam')

from tensorflow.python.keras.callbacks import EarlyStopping
earlyStopping = EarlyStopping(monitor='val_loss', patience=100, mode='min', verbose=1,
                              restore_best_weights=True) 

start_time = time.time()
model.fit(x_train, y_train, epochs=10000, batch_size=1000, verbose=1, validation_split=0.2,
                 callbacks=[earlyStopping])  
end_time = time.time() -start_time

model.save("./_save/keras23_3_save_model.h5")




# model = load_model("./_save/keras23_3_save_model.h5")
            #   해당 경로에 있는 모델을 가져옴
            #   해당 경로에 있는 저장된 모델과 가중치 값을 가져온다
            #   해당 모델을 만들 때 나왔던 가중치 값이 그대로 저장되고 그 값을 가져옴

model.save("./_save/keras23_5_save_weights2.h5")


#4. 평가, 예측
loss = model.evaluate(x_test, y_test)
print('loss : ', loss)

y_predict = model.predict(x_test)
from sklearn.metrics import r2_score
r2 = r2_score(y_test, y_predict)
print("r2스코어 : ", r2)




# loss :  9.567126274108887
# r2스코어 :  0.8855373311554513
# 걸린시간 :  24.923790454864502













