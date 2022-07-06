import numpy as np
from tensorflow.python.keras.models import Sequential
from tensorflow.python.keras.layers import Dense
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
model.add(Dense(100, activation='relu', input_dim=10))
model.add(Dense(100, activation='relu'))
model.add(Dense(100, activation='relu'))
model.add(Dense(100, activation='relu'))
model.add(Dense(1, activation='relu'))


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
 

























