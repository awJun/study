# 데이콘 따릉이 문제풀이
from sklearn.metrics import r2_score, accuracy_score
from tensorflow.python.keras.layers import Dense, Conv2D, Flatten, MaxPooling2D, Dropout, LSTM   
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
path = './_data/ddarung/'
train_set = pd.read_csv(path + 'train.csv', # + 명령어는 문자를 앞문자와 더해줌
                        index_col=0) # index_col=n n번째 컬럼을 인덱스로 인식

test_set = pd.read_csv(path + 'test.csv', # 예측에서 쓸거임                
                       index_col=0)

#--[ 데이터 정보 출력 ]- - - - - - - - - - - - - - - - - - - - -  - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - 
# print(train_set.columns)
# print(train_set.info()) # info 정보출력
# print(train_set.describe()) # describe 평균치, 중간값, 최소값 등등 출력


# #--[ 결측치 확인, 처리 ]- - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - 
# print(train_set.isnull().sum())
train_set = train_set.fillna(train_set.mean()) # train_set의 데이터를 평균으로 채우겠다 !
test_set = test_set.fillna(test_set.mean()) # test_set 데이터를 평균으로 채우겠다 !
# print(train_set.isnull().sum())             # 난값 여부 확인
# print(train_set.shape) # (1328, 10) 
# #- - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -


x = train_set.drop(['count'], axis=1)  # drop 데이터에서 ''사이 값 빼기
# print(x)
# print(x.columns)
# print(x.shape) # (1459, 9)

y = train_set['count'] 
# print(y)
# print(y.shape) # (1459,)

x_train, x_test, y_train, y_test = train_test_split(x,y,
                                                    train_size=0.8,
                                                    random_state=100
                                                    )



#--[해당 데이터의 unique형태 확인]- - - - - - - - - - - - - - - - - 
# print(np.unique(x_train, return_counts=True))
# print(np.unique(y_train, return_counts=True))    # <--- 이 항목은 확인 필수 (회귀형)
# print(np.unique(x_test, return_counts=True))   
# print(np.unique(y_test, return_counts=True))
#- - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - 
#--[해당 데이터 shape 확인]- - - - - - - - - - - - - - - - - - - - -
# print(x_train.shape)   # (1167, 9)
# print(y_train.shape)   # (1167,)
# print(x_test.shape)    # (292, 9)
# print(y_test.shape)    # (292,)
# - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
#--[ 스케일러 작업]- - - - - - - - - - - - - - - -(중간값을 찾아주는 역할 (값들의 차이 완화)
# scaler =  MinMaxScaler()
scaler = StandardScaler()
# scaler = MaxAbsScaler()
# scaler = RobustScaler()

scaler.fit(x_train)
x_train = scaler.transform(x_train) # x_train을 수치로 변환해준다.
x_test = scaler.transform(x_test) # 
#- - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
#--[차원 변형 작업]- - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - 
x_train = x_train.reshape(1167, 3, 3)               
x_test = x_test.reshape(292, 3, 3)

# print(x_train.shape)  # (1167, 3, 3, 1)     <-- "32, 2 ,1"는 input_shape값
# print(x_test.shape)   # (292, 3, 3, 1)
#- - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
#- -[y데이터를 x와 동일한 차원 형태로 변환] - - - - -  - - - - - - - - - - - ( [중요]!! 회귀형에서는 할 필요없음  )
# from tensorflow.python.keras.utils.np_utils import to_categorical
# y_train = to_categorical(y_train)
# y_test = to_categorical(y_test)
# print(y_train.shape, y_test.shape) # (1167, 432) (292, 403)



#2. 모델구성
model = Sequential()
model.add(LSTM(units=100 ,input_length=3, input_dim=3))   #SimpleRNN  또는 LSTM를 거치면 3차원이 2차원으로 변형되어서 다음 레어어에 간다.
# model.add(SimpleRNN(10))       # <-- 3차원이 아니라 2차원을 넣어서 에러가 발생함.[참고]
model.add(Dense(300, activation='relu'))
model.add(Dense(300, activation='relu'))
model.add(Dense(300, activation='relu'))
model.add(Dense(100, activation='relu'))   
model.add(Dense(128, activation='relu'))
model.add(Dropout(0.2))
model.add(Dense(128, activation='relu'))
model.add(Dense(8, activation='softmax')) # sigmoid에선 output은 1이다. (softmax에서는 유니크 갯수만큼)
model.summary()



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

# loss :  20433.466796875
# RMSE :  139.8785187224972
# r2스코어 :  -1.5875101222528052
# 걸린시간: 3.324416399002075




