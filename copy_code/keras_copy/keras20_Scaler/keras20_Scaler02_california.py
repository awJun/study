import numpy as np
import time

from tensorflow.python.keras.models import Sequential
from tensorflow.python.keras.layers import Dense
from sklearn.model_selection import train_test_split

from sklearn.datasets import fetch_california_housing
datasets = fetch_california_housing()

from sklearn.preprocessing import MinMaxScaler, StandardScaler
from sklearn.preprocessing import MaxAbsScaler, RobustScaler

x=datasets.data
y=datasets.target




x_train, x_test, y_train, y_test = train_test_split(x, y,
                                                    train_size=0.7,
                                                    shuffle=True,
                                                    random_state=100
                                                    )


# scaler =  MinMaxScaler()
# scaler = StandardScaler()
scaler = MaxAbsScaler()
# scaler = RobustScaler()


scaler.fit(x_train)
x_train = scaler.transform(x_train) # x_train을 수치로 변환해준다.
x_test = scaler.transform(x_test) # 
# print(np.min(x_train))   # 0.0
# print(np.max(x_train))   # 1.0000000000000002
# print(np.min(x_test))   # -0.06141956477526944
# print(np.max(x_test))   # 1.1478180091225068
 

#2. 모델구성
model = Sequential()
model.add(Dense(100, input_dim=8))
model.add(Dense(100))
model.add(Dense(100))
model.add(Dense(1))

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

y_predict = model.predict(x_test)

from sklearn.metrics import r2_score
r2 = r2_score(y_test, y_predict)   # 0 ~ 1까지 지표를 나타낸다 (1에 가까우면 가장 좋다 (1에 마출 것.))
print('r2스코어 : ', r2)

print("걸린시간 : ", end_time)

#########################################################
"""   
scaler 사용 안함

loss :  0.6195424795150757
r2스코어 :  0.532446163460877
걸린시간 :  119.76127767562866
"""
#########################################################
"""
scaler = StandardScaler()

loss :  0.6226364970207214
r2스코어 :  0.530111076632425
걸린시간 :  92.76453351974487
"""
#########################################################
"""
scaler =  MinMaxScaler()

loss :  0.5068815350532532
r2스코어 :  0.6174687346200469
걸린시간 :  81.56398820877075
"""
#########################################################
"""
scaler = MaxAbsScaler()

loss :  0.5065802931785583
r2스코어 :  0.6176958806668096
걸린시간 :  290.2430865764618
"""
#########################################################
"""
scaler = RobustScaler()

loss :  0.7556058168411255
r2스코어 :  0.42976221016615657
걸린시간 :  154.67119479179382
"""  
#########################################################
 