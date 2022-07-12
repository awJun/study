# 캐글 바이크 문제풀이
from sklearn.metrics import r2_score, accuracy_score
from tensorflow.python.keras.layers import Dense, Conv2D, Flatten, MaxPooling2D ,Dropout   
from tensorflow.python.keras.models import Sequential
import pandas as pd
import numpy as np
import tensorflow as tf       
import time
from sklearn.model_selection import train_test_split # 함수 가능성이 높음
from sklearn.preprocessing import MinMaxScaler, StandardScaler # 클래스 가능성이 높음
from sklearn.preprocessing import MaxAbsScaler, RobustScaler
from sklearn.datasets import fetch_covtype
from tensorflow.python.keras.callbacks import EarlyStopping
from sklearn.metrics import r2_score, mean_squared_error

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



#--[해당 데이터의 unique형태 확인]- - - - - - - - - - - - - - - - - 
# print(np.unique(x_train, return_counts=True))
# print(np.unique(y_train, return_counts=True))    # <--- 이 항목은 확인 필수 (회귀형)
# print(np.unique(x_test, return_counts=True))   
# print(np.unique(y_test, return_counts=True))
#- - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - 
#--[해당 데이터 shape 확인]- - - - - - - - - - - - - - - - - - - - -
# print(x_train.shape)   # (7620, 8)
# print(y_train.shape)   # (7620,)
# print(x_test.shape)    # (3266, 8)
# print(y_test.shape)    # (3266,)
# - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
#--[ 스케일러 작업]- - - - - - - - - - - - - - - -(중간값을 찾아주는 역할 (값들의 차이 완화)
# scaler =  MinMaxScaler()
scaler = StandardScaler()
# scaler = MaxAbsScaler()
# scaler = RobustScaler()

scaler.fit(x_train)
x_train = scaler.transform(x_train) # x_train을 수치로 변환해준다.
x_test = scaler.transform(x_test) 
#- - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
#--[차원 변형 작업]- - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - 
x_train = x_train.reshape(7620, 4, 2, 1)               
x_test = x_test.reshape(3266, 4, 2, 1)

# print(x_train.shape)  # (7620, 4, 2, 1)     <-- "4, 2 ,1"는 input_shape값
# print(x_test.shape)   # (3266, 4, 2, 1)
#- - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
#- -[y데이터를 x와 동일한 차원 형태로 변환] - - - - - - - - - - - ( [중요]!! 회귀형에서는 할 필요없음  )
# from tensorflow.python.keras.utils.np_utils import to_categorical
# y_train = to_categorical(y_train)
# y_test = to_categorical(y_test)
# print(y_train.shape, y_test.shape) # 


#2. 모델구성
model = Sequential()
model.add(Conv2D(filters=50, kernel_size=(2,2),  
                 input_shape=(4, 2, 1)))     #(batsh_size, row, columns, channels)
                                                                        # channels는 장수  / 1장 2장
model.add(Dropout(0.2))
model.add(Conv2D(64, (1, 1), padding='valid', activation='relu'))     # valid 디폴트이므로 패딩 안하겠다 라는 뜻.                     
model.add(Dropout(0.2))
model.add(Conv2D(32, (1, 1), padding='same', activation='relu'))
model.add(Dropout(0.2))
model.add(Conv2D(128, (1, 1), padding='valid', activation='relu'))                         
model.add(Dropout(0.2))
model.add(Conv2D(128, (1, 1), padding='same', activation='relu'))
model.add(Dropout(0.2))

model.add(Flatten())    
model.add(Dense(128, activation='relu'))
model.add(Dropout(0.2))
model.add(Dense(128, activation='relu'))
model.add(Dense(1))
# model.summary()

#3. 컴파일 훈련

model.compile(loss='mse', optimizer='adam', # 다중 분류에서는 로스함수를 'categorical_crossentropy' 로 써준다 (99퍼센트로)
              metrics=['accuracy'])

es = EarlyStopping(monitor='val_loss', patience=1000, mode='auto', verbose=1, 
                              restore_best_weights=True)   
start_time = time.time()
model.fit(x_train, y_train, epochs=10, batch_size=100,
                 validation_split=0.2,
                 callbacks=[es],
                 verbose=1)
end_time = time.time() -start_time



#4. 평가, 예측
loss, acc = model.evaluate(x_test, y_test) 

y_predict = model.predict(x_test)
y_predict = np.argmax(y_predict, axis= 1)  # 위에서 갯더미 썻으므로 argmax로 평가지표를 사용할 수 있게 만듬
# y_test = np.argmax(y_test, axis= 1)   # 회귀형이므로 위에서 갯더미를 안했으므로 쓰면안됨

def RMSE(a, b): 
    return np.sqrt(mean_squared_error(a, b))

rmse = RMSE(y_test, y_predict)


from sklearn.metrics import r2_score
r2 = r2_score(y_test, y_predict)

print('loss : ', loss)
print('acc : ', acc)  # <-- 그냥 뽑아본거임 
print("RMSE : ", rmse)
print('r2스코어 : ', r2)
print("걸린시간:", end_time )

# loss :  23135.341796875
# acc :  0.010716472752392292
# RMSE :  260.90979711200254
# r2스코어 :  -1.1065314473268741
# 걸린시간: 7.220516920089722