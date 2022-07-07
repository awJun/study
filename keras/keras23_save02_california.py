import numpy as np
import time

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
from tensorflow.python.keras.models import Sequential, Model
from tensorflow.python.keras.layers import Dense, Input
"""
### 기존 모델 ###

model = Sequential()
model.add(Dense(100, input_dim=8))
model.add(Dense(100))
model.add(Dense(100))
model.add(Dense(1))
"""
### 새로운 모델 ###
input1 = Input(shape=(8,))   # 처음에 Input 명시하고 Input 대한 shape 명시해준다.
dense1 = Dense(100)(input1)   # Dense 구성을하고  node 값을 넣고 받아오고 싶은 변수 받아온다.
dense2 = Dense(100, activation = 'relu')(dense1)    # 받아온 변수를 통해 훈련의 순서를 사용자가 원하는대로 할 수 있다.
dense3 = Dense(100, activation = 'sigmoid')(dense2)
output1 = Dense(1)(dense3)
model = Model(inputs=input1, outputs=output1) # 해당 모델의 input과 output을 설정한다.


#3. 컴파일. 훈련
model.compile(loss='mse', optimizer='adam')

from tensorflow.python.keras.callbacks import EarlyStopping
earlyStopping = EarlyStopping(monitor='val_loss', patience=100, mode='min', verbose=1,
                              restore_best_weights=True) 
  
start_time = time.time()
model.fit(x_train, y_train, epochs=10000, batch_size=50, verbose=1, validation_split=0.2,
                 callbacks=[earlyStopping])  
end_time = time.time() -start_time

model.save("./_save/keras23_2_save_california.h5")


#4. 평가, 예측
loss = model.evaluate(x_test, y_test)
print('loss : ', loss)

y_predict = model.predict(x_test)

from sklearn.metrics import r2_score
r2 = r2_score(y_test, y_predict)   # 0 ~ 1까지 지표를 나타낸다 (1에 가까우면 가장 좋다 (1에 마출 것.))
print('r2스코어 : ', r2)

print("걸린시간 : ", end_time)




#########################################################
"""   [best_scaler]

scaler = MaxAbsScaler()

loss :  0.30951300263404846
r2스코어 :  0.7664178877268616
걸린시간 :  132.0903663635254
"""
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

 