# 데이콘 따릉이 문제풀이
import numpy as np
import pandas as pd
from sqlalchemy import true #pandas : 엑셀땡겨올때 씀
from tensorflow.python.keras.models import Sequential
from tensorflow.python.keras.layers import Dense
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score, mean_squared_error
from sklearn.preprocessing import MinMaxScaler, StandardScaler
import time
from sklearn.preprocessing import MaxAbsScaler, RobustScaler


#1. 데이터
path = './_data/ddarung/'
train_set = pd.read_csv(path + 'train.csv', # + 명령어는 문자를 앞문자와 더해줌
                        index_col=0) # index_col=n n번째 컬럼을 인덱스로 인식
print(train_set)
print(train_set.shape) # (1459, 10)

test_set = pd.read_csv(path + 'test.csv', # 예측에서 쓸거임                
                       index_col=0)
print(test_set)
print(test_set.shape) # (715, 9)

print(train_set.columns)
print(train_set.info()) # info 정보출력
print(train_set.describe()) # describe 평균치, 중간값, 최소값 등등 출력

#### 결측치 처리 1. 제거 ####
print(train_set.isnull().sum())
train_set = train_set.fillna(train_set.mean()) # dropna() : train_set 에서 na, null 값 들어간 행 삭제
test_set = test_set.fillna(test_set.mean()) # test_set 에서 이빨빠진데 바로  ffill : 위에서 가져오기 test_set.mean : 평균값
print(train_set.isnull().sum()) 
print(train_set.shape) # (1328, 10)

############################


x = train_set.drop(['count'], axis=1)  # drop 데이터에서 ''사이 값 빼기
print(x)
print(x.columns)
print(x.shape) # (1459, 9)

y = train_set['count'] 
print(y)
print(y.shape) # (1459,)

x_train, x_test, y_train, y_test = train_test_split(x,y,
                                                    train_size=0.8,
                                                    random_state=100
                                                    )



scaler =  MinMaxScaler()
# scaler = StandardScaler()
# scaler = MaxAbsScaler()
# scaler = RobustScaler()

scaler.fit(x_train)
x_train = scaler.transform(x_train) # x_train을 수치로 변환해준다.
x_test = scaler.transform(x_test) # 

test_set = scaler.transform(test_set) # 마지막에 사용할 test_set을 전처리 작업

# print(np.min(x_train))   # 0.0
# print(np.max(x_train))   # 1.0000000000000002
# print(np.min(x_test))   # -0.06141956477526944
# print(np.max(x_test))   # 1.1478180091225068
 
 
 
 
##### [ 3가지 성능 비교 ] #####
# scaler 사용하기 전
# scaler =  MinMaxScaler()
# scaler = StandardScaler()




#2. 모델구성
model = Sequential()
model.add(Dense(100, activation='selu', input_dim=9))
model.add(Dense(100, activation='selu'))
model.add(Dense(100, activation='relu'))
model.add(Dense(100, activation='selu'))
model.add(Dense(1))




#3. 컴파일, 훈련
from tensorflow.python.keras.callbacks import EarlyStopping
earlyStopping = EarlyStopping(monitor='val_loss', patience=500, mode='min', verbose=1, 
                              restore_best_weights=True)




start_time = time.time()
model.compile(loss='mse', optimizer='adam', metrics=['mae'])
model.fit(x_train, y_train, epochs=10000, batch_size=100, verbose=1, validation_split=0.2, callbacks=[earlyStopping])
end_time = time.time()  -start_time





#4. 평가, 예측
loss = model.evaluate(x, y) 


y_predict = model.predict(x_test)

def RMSE(a, b): 
    return np.sqrt(mean_squared_error(a, b))

rmse = RMSE(y_test, y_predict)


from sklearn.metrics import r2_score
r2 = r2_score(y_test, y_predict)

print('loss : ', loss)
print("RMSE : ", rmse)
print('r2스코어 : ', r2)
print("걸린시간:", end_time )


#########################################################
"""   
scaler 사용 안함

loss :  1324.395263671875
RMSE :  43.473589635763254
r2스코어 :  0.7500627360082281
걸린시간: 110.4030532836914
"""
#########################################################
"""
scaler = StandardScaler()

loss :  1469099264.0
RMSE :  41.408083530054775
r2스코어 :  0.7732484472613282
걸린시간: 55.89106202125549
"""
#########################################################
"""
scaler =  MinMaxScaler()

loss :  756685952.0
RMSE :  41.08596719650925
r2스코어 :  0.7767625575278743
걸린시간: 99.75401592254639
"""
#########################################################
"""
scaler = MaxAbsScaler()

loss :  64436884.0
RMSE :  40.579690125032144
r2스코어 :  0.782230295948688
걸린시간: 94.45908308029175
"""
#########################################################
"""
scaler = RobustScaler()

loss :  1181159936.0
RMSE :  41.532002097256786
r2스코어 :  0.771889255069705
걸린시간: 52.941198110580444
"""  
#########################################################
 


