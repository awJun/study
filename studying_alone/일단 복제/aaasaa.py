"""
평가까지
LSTM Conv1D
스플릿트 사용 컬럼이 뭉치로 사용되어야 한다.
"""
# https://deep-deep-deep.tistory.com/60 참고할 것
# 이 데이터 세트의 주요 목적은 "Deep Learning with Python" 책에서 RNN 연습(6.3.1 온도 예측 문제)을 수행하는 것입니다.

from sklearn.metrics import r2_score, accuracy_score
from tensorflow.python.keras.layers import Dense, Conv2D, Flatten, MaxPooling2D, Dropout, Conv1D, LSTM
from tensorflow.python.keras.models import Sequential
import pandas as pd
import numpy as np
import tensorflow as tf       
import time
from sklearn.model_selection import train_test_split # 함수 가능성이 높음
from sklearn.preprocessing import MinMaxScaler, StandardScaler # 클래스 가능성이 높음
from sklearn.preprocessing import MaxAbsScaler, RobustScaler

from tensorflow.python.keras.callbacks import EarlyStopping
from sklearn.metrics import r2_score, mean_squared_error

df = pd.read_csv('./_data/kaggle_jena/jena_climate_2009_2016.csv', index_col=0)
# df.head()
# print(df.shape)   # (420551, 14)




def split_x(a, b):
    임의의_리스트선언 = []
    for i in range(len(a) - b + 1): 
        subset = a[i : (i + b)]    
        임의의_리스트선언.append(subset)   
    
       
    return np.array(임의의_리스트선언)

#--[trainset 만드는 과정]-----------
size = 5   #  x = 4개  y는 1개
train = split_x(df, size)

# print(train)         
# print(train.shape)    # (96, 5)
#-----------------------------------


#---[trainset에서 x와 y데이터를 추출하는 과정]---------------------------------------------------------------------------------------------------------
# 1, 2, 3, 4를 x데이터로 만드는 과정 5번째 열은 뺌 / 왜냐하면 "1, 2, 3, 4에 대한 예측은 5"라는 형태의 데이터로 만들기 위해 y에서 사용할 것이기 때문이다.
x = train[:, :-1]     
# x에서 빼버린 5번째 열을 y데이터로 사용하겟다는 뜻 ~ 
y = train[:, -1]       
# -----------------------------------------------------------------------------------------------------------------------------------------------------


#-[예측 단계중 predict에서 사용 할 데이터 만드는 과정]- - - - - - - - - - -
size = 4
test = split_x(df, size)
# print(x.shape)      # (420547, 4, 14)
# print(y.shape)      # (420547, 14)
# print(test.shape)   # (420548, 4, 14)

# test = test.reshape(420548, 56)
# print(test.shape)
# # print(x)
# # print(y)


#--[ 데이터 정보 출력 ]- - - - - - - - - - - - - - - - - - - - -  - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - 
# print(df.columns)
# print(df.info()) # info 정보출력
# print(df.describe()) # describe 평균치, 중간값, 최소값 등등 출력


# #--[ 결측치 확인, 처리 ]- - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - 
# print(df.isnull().sum())
# df = df.fillna(df.mean()) # train_set의 데이터를 평균으로 채우겠다 !
 
# test_set 데이터를 평균으로 채우겠다 !
# print(train_set.isnull().sum())             # 난값 여부 확인
# print(train_set.shape) # (1328, 10) 


x_train, x_test, y_train, y_test = train_test_split(x,y,
                                                    train_size=0.8,
                                                    random_state=100
                                                    )



#--[해당 데이터의 unique형태 확인]- - - - - - - - - - - - - - - - - 
# print(np.unique(x_train, return_counts=True))
print(np.unique(y_train, return_counts=True))    # <--- 이 항목은 확인 필수 (회귀형)
# print(np.unique(x_test, return_counts=True))   
# print(np.unique(y_test, return_counts=True))
#- - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - 
#--[해당 데이터 shape 확인]- - - - - - - - - - - - - - - - - - - - -
# print(x_train.shape)   # (336437, 4, 14)
# print(y_train.shape)   # (336437, 14)  
# print(x_test.shape)    # (84110, 4, 14)
# print(y_test.shape)    # (84110, 14)   
# - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
#--[ 스케일러 작업]- - - - - - - - - - - - - - - -(중간값을 찾아주는 역할 (값들의 차이 완화)
# scaler =  MinMaxScaler()
scaler = StandardScaler()
# # scaler = MaxAbsScaler()
# # scaler = RobustScaler()

# scaler.fit(x_train)
# x_train = scaler.transform(x_train) # x_train을 수치로 변환해준다.
# x_test = scaler.transform(x_test) # 
# #- - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
# --[차원 변형 작업]- - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - 
# x_train = x_train.reshape(336437, 3, 3)               
# x_test = x_test.reshape(84110, 3, 3)

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
model.add(LSTM(units=100 ,input_length=4, input_dim=14))   #SimpleRNN  또는 LSTM를 거치면 3차원이 2차원으로 변형되어서 다음 레어어에 간다.
# model.add(SimpleRNN(10))       # <-- 3차원이 아니라 2차원을 넣어서 에러가 발생함.[참고]
model.add(Dense(300, activation='relu'))
model.add(Dense(300, activation='relu'))
model.add(Dense(300, activation='relu'))
model.add(Dense(100, activation='relu'))   
model.add(Dense(128, activation='relu'))
model.add(Dropout(0.2))
model.add(Dense(128, activation='relu'))
model.add(Dense(1)) # sigmoid에선 output은 1이다. (softmax에서는 유니크 갯수만큼)
model.summary()



#3. 컴파일 훈련

model.compile(loss='mse', optimizer='adam', # 다중 분류에서는 로스함수를 'categorical_crossentropy' 로 써준다 (99퍼센트로)
              metrics=['accuracy'])

es = EarlyStopping(monitor='val_loss', patience=1000, mode='auto', verbose=1, 
                              restore_best_weights=True)   
start_time = time.time()
model.fit(x_train, y_train, epochs=1, batch_size=100,
                 validation_split=0.2,
                 callbacks=[es],
                 verbose=1)
end_time = time.time() -start_time





#4. 평가, 예측
loss, acc = model.evaluate(x_test, y_test) 

y_predict = model.predict(y_test)
# y_test = np.argmax(y_test, axis= 1)   # 회귀형이므로 위에서 갯더미를 안했으므로 쓰면안됨

# def RMSE(a, b): 
#     return np.sqrt(mean_squared_error(a, b))

# rmse = RMSE(y_test, y_predict)


from sklearn.metrics import r2_score
r2 = r2_score(y_test, y_predict)

print('loss : ', loss)
print('acc : ', acc)  # <-- 그냥 뽑아본거임 
# print("RMSE : ", rmse)
print('r2스코어 : ', r2)
print("걸린시간:", end_time )






































































































