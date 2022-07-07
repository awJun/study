import numpy as np

import time
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from sklearn.preprocessing import MaxAbsScaler, RobustScaler

from sklearn.datasets import load_diabetes

datasets = load_diabetes()

x = datasets.data
y = datasets.target


x_train, x_test, y_train, y_test = train_test_split(x, y,
                                                    train_size=0.7,
                                                    shuffle=True,
                                                    random_state=72
                                                    )


# # scaler =  MinMaxScaler()
# # scaler = StandardScaler()
# scaler = MaxAbsScaler()
# # scaler = RobustScaler()

# scaler.fit(x_train)
# x_train = scaler.transform(x_train) # x_train을 수치로 변환해준다.
# x_test = scaler.transform(x_test) # 
# # print(np.min(x_train))   # 0.0
# # print(np.max(x_train))   # 1.0000000000000002
# # print(np.min(x_test))   # -0.06141956477526944
# # print(np.max(x_test))   # 1.1478180091225068

#2. 모델구성
from tensorflow.python.keras.models import Sequential, Model
from tensorflow.python.keras.layers import Dense, Input
"""
### 기존 모델 ###

model = Sequential()
model.add(Dense(100, activation='relu', input_dim=10))
model.add(Dense(100, activation='relu'))
model.add(Dense(100, activation='relu'))
model.add(Dense(100, activation='relu'))
model.add(Dense(1, activation='relu'))
"""
### 새로운 모델 ###
input1 = Input(shape=(10,))   # 처음에 Input 명시하고 Input 대한 shape 명시해준다.
dense1 = Dense(100)(input1)   # Dense 구성을하고  node 값을 넣고 받아오고 싶은 변수 받아온다.
dense2 = Dense(100, activation = 'relu')(dense1)    # 받아온 변수를 통해 훈련의 순서를 사용자가 원하는대로 할 수 있다.
dense3 = Dense(100, activation = 'sigmoid')(dense2)
output1 = Dense(1)(dense3)
model = Model(inputs=input1, outputs=output1) # 해당 모델의 input과 output을 설정한다.


#3. 컴파일, 훈련
model.compile(loss='mae', optimizer='adam')
    
from tensorflow.python.keras.callbacks import EarlyStopping
earlyStopping = EarlyStopping(monitor='val_loss', patience=10, mode='auto', verbose=1, 
                              restore_best_weights=True) 
                              
start_time = time.time()
hist = model.fit(x_train, y_train, epochs=100, batch_size=50,
                 validation_split=0.2,
                 callbacks=[earlyStopping],
                 verbose=1)
end_time = time.time() -start_time


#4. 평가, 예측
y_predict = model.predict(x_test)

loss = model.evaluate(x_test, y_test)

from sklearn.metrics import r2_score
r2 = r2_score(y_test, y_predict)

print('loss : ', loss)
print('r2스코어 : ', r2)
print("걸린시간 : ", end_time)

#########################################################
"""   [best_scaler]

scaler 사용 안함

loss :  113.3897705078125
r2스코어 :  -2.1461399476448344
걸린시간 :  2.6223225593566895
"""
#########################################################
"""   
scaler 사용 안함

loss :  38.690216064453125
r2스코어 :  0.6215756666644343        
걸린시간 :  3.0963847637176514 
"""
#########################################################
"""
scaler = StandardScaler()

loss :  39.522029876708984
r2스코어 :  0.573839866429229
걸린시간 :  3.2738006114959717 
"""
#########################################################
"""
scaler =  MinMaxScaler()

loss :  38.84760284423828
r2스코어 :  0.6246470299897755        
걸린시간 :  4.107311010360718
"""
#########################################################
"""
scaler = MaxAbsScaler()

loss :  39.88737487792969
r2스코어 :  0.5806081826115181
걸린시간 :  2.2184596061706543
"""
#########################################################
"""
scaler = RobustScaler()

loss :  39.540592193603516
r2스코어 :  0.5666960138761359
걸린시간 :  3.4847681522369385
"""  
#########################################################
 

























