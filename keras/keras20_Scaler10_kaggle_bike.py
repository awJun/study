# 캐글 바이크 문제풀이
import numpy as np
import pandas as pd
from tensorflow.python.keras.models import Sequential
from tensorflow.python.keras.layers import Dense
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
model.add(Dense(100, activation = 'swish', input_dim=8))
model.add(Dense(100, activation = 'swish'))
model.add(Dense(100, activation = 'swish'))
model.add(Dense(1))

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
"""   
scaler 사용 안함

loss :  22648.078125
RMSE :  150.49279250784505
r2스코어 :  0.29916049440692305
걸린시간 :  24.058242321014404
"""
#########################################################
"""
scaler = StandardScaler()

loss :  22207.150390625
RMSE :  149.02062962703266
r2스코어 :  0.3128050478018569
걸린시간 :  23.781530380249023
"""
#########################################################
"""
scaler =  MinMaxScaler()

loss :  22727.87109375
RMSE :  150.75766685133425
r2스코어 :  0.29669130284698797
걸린시간 :  20.41472029685974
"""
#########################################################
"""
scaler = MaxAbsScaler()

loss :  22359.943359375
RMSE :  149.53242636779407
r2스코어 :  0.3080767348914789
걸린시간 :  37.655221700668335
"""
#########################################################
"""
scaler = RobustScaler()

loss :  22107.0546875
RMSE :  148.68440581233205
r2스코어 :  0.3159024801127748
걸린시간 :  39.04133200645447
"""  
#########################################################
 
