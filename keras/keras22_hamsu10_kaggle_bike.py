# 캐글 바이크 문제풀이
import numpy as np
import pandas as pd

import time
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score, mean_squared_error 

from sklearn.preprocessing import MinMaxScaler, StandardScaler 
from sklearn.preprocessing import MaxAbsScaler, RobustScaler

#1. 데이터
path = './_data/kaggle_bike/'  

     # 결측치는 없는 것으로 판명                                                               
train_set = pd.read_csv(path + 'train.csv', index_col=0)  
  # (10886, 11)                
test_set = pd.read_csv(path + 'test.csv', index_col=0) 
  # (6493, 8)


x = train_set.drop(['casual',  'registered',  'count'], axis=1)  #       axis = 0/1(축값, 0=행, 1=열)                                                                    
   # (10886, 8)
y = train_set['count']
   # (10886,)
   


x_train, x_test, y_train, y_test = train_test_split(x, y,
                                                    train_size=0.7,
                                                    shuffle=True,
                                                    random_state=100
                                                    )




# scaler =  MinMaxScaler()
scaler = StandardScaler()
# scaler = MaxAbsScaler()
# scaler = RobustScaler()

scaler.fit(x_train)
x_train = scaler.transform(x_train) # x_train을 수치로 변환해준다.
x_test = scaler.transform(x_test) # 
test_set = scaler.transform(test_set) # 

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
model.add(Dense(100, activation = 'swish', input_dim=8))
model.add(Dense(100, activation = 'swish'))
model.add(Dense(100, activation = 'swish'))
model.add(Dense(1))
"""
### 새로운 모델 ###
input1 = Input(shape=(8,))   # 처음에 Input 명시하고 Input 대한 shape 명시해준다.
dense1 = Dense(100)(input1)   # Dense 구성을하고  node 값을 넣고 받아오고 싶은 변수 받아온다.
dense2 = Dense(100, activation = 'relu')(dense1)    # 받아온 변수를 통해 훈련의 순서를 사용자가 원하는대로 할 수 있다.
dense3 = Dense(100, activation = 'sigmoid')(dense2)
output1 = Dense(1)(dense3)
model = Model(inputs=input1, outputs=output1) # 해당 모델의 input과 output을 설정한다.


#3. 컴파일, 훈련
model.compile(loss='mse', optimizer='adam')
    
from tensorflow.python.keras.callbacks import EarlyStopping
earlyStopping = EarlyStopping(monitor='val_loss', patience=10, mode='auto', verbose=1, 
                              restore_best_weights=True) 
                              
start_time = time.time()
hist = model.fit(x_train, y_train, epochs=100, batch_size=50,
                 validation_split=0.2,
                 callbacks=[earlyStopping],
                 verbose=1)
end_time = time.time() -start_time

#4. 결과, 예측
y_predict = model.predict(x_test)

loss = model.evaluate(x_test, y_test) 

def RMSE(a, b): 
    return np.sqrt(mean_squared_error(a, b))

rmse = RMSE(y_test, y_predict)

r2 = r2_score(y_test, y_predict)

print('loss : ', loss)
print("RMSE : ", rmse)
print('r2스코어 : ', r2)
print("걸린시간 : ", end_time)




# y_summit = model.predict(test_set)
  
      
# submissin_set = pd.read_csv(path + 'sampleSubmission.csv', index_col=0)
#    # print(submissin_set.shape)   # (6493, 1)

# submissin_set['count'] = y_summit
# submissin_set['count'] = list(map(abs,submissin_set['count']))
# submissin_set.to_csv('./_data/kaggle_bike/sampleSubmission.csv', index = True)
      
      
      
#########################################################
"""   [best_scaler]

scaler = StandardScaler()

loss :  22122.166015625
RMSE :  148.73522881662836
r2스코어 :  0.3154347265087888
걸린시간 :  12.737728595733643
"""
#########################################################
"""   
scaler 사용 안함

loss :  22468.453125
RMSE :  149.8948254647025
r2스코어 :  0.30471885155109524
걸린시간 :  41.20134615898132
"""
#########################################################
"""
scaler = StandardScaler()

loss :  22162.04296875
RMSE :  148.86921356642546
r2스코어 :  0.31420082086120626
걸린시간 :  31.72047257423401
"""
#########################################################
"""
scaler =  MinMaxScaler()

loss :  22476.05859375
RMSE :  149.9201724919025
r2스코어 :  0.3044836893276287
걸린시간 :  41.15575933456421
"""
#########################################################
"""
scaler = MaxAbsScaler()

loss :  22316.7890625
RMSE :  149.3880408579704
r2스코어 :  0.3094123042166941
걸린시간 :  50.684988021850586
"""
#########################################################
"""
scaler = RobustScaler()

loss :  22168.248046875
RMSE :  148.89006350215016
r2스코어 :  0.3140087076662841
걸린시간 :  119.25653839111328
"""  
#########################################################
 
