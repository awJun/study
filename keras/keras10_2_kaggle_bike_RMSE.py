"""=[ RMSE 사용 ]==============================================================================================

from sklearn.metrics import mean_squared_error 

def RMSE(a, b):
    return np.sqrt(mean_squared_error(a, b))

rmse = RMSE(y_test, y_predict)
print("RMSE : ", rmse)

===[ mean_squared_error 선언 이유 ]==============================================================================================

loss에서는 RMSE를 사용할 수 없으므로 mean_squared_error에서 MSE를 불러온 후 sqrt(루트)를 해줘서
RMSE를 직접 만들어서 사용한다.

===[ to_datetime 사용이유 ]==============================================================================================

[to_datetime 설명링크] https://jimmy-ai.tistory.com/156

시간 형식의 object 자료형의 열을 datetime 형식으로 변환

train_set의 데이터중 [datetime]의 열의 형태가 2011-01-01 00:00:00 형태로 되어 있어서 이것을 
년도['year'], 월['month'], 일['day'], 시간['hour'],  분과 초는 모든값이 0이므로 추가 안했음.

===[ objecta란 ]=======================================================================================================

판다스에서는 문자열을 object라는 자료형으로 나타냅니다. 파이썬에서는 문자열을 string이라고 하지만, 판다스는
object라고 합니다. pd.DataFrame을 사용하여 데이터프레임을 만들때 dtype(형식)을 지정해주는게 아니라면
일반적으로 데이터를 받아들일 때 숫자형을 제외한 나머지는 object로 받아준다.

===[ objecta와 category의 차이 ]=======================================================================================================

objecta는 수치형 -->  1, 2, 3, 4, 5, 6, 7 or 일반적인 문자열을 갖는 칼럼은 object로 사용
category는 분리형 0, 1에서


===[ category 설명 ]=========================================================================================================

  [category 설명링크]https://www.tutorialspoint.com/python_pandas/python_pandas_categorical_data.htm

[일반적인 문자열을 갖는 열은 object로 사용하고, "값의 종류가 제한적"일 때 category를 사용]
 
 ex)
 아침식사 여부에 대한 칼럼이라면 값을 0=먹지않음, 1=먹음 으로 두 종류만 갖게 되겠죠. 반대로 아침식사 종류에 
 대한 칼럼이라면 샐러드, 소고기, 바나나 하나, 샌드위치, 어제 먹다 남은 마라탕(?) 등 다양한 값을 가질 수 
 있게 됩니다. 이 경우 아침식사 여부는 category, 아침식사 종류는 object로 지정해주면 효율적으로 
 데이터 프레임을 관리할 수 있겠죠.



===[ get_dummies 설명 ]=========================================================================================================

[get_dummies 설명링크]  https://devuna.tistory.com/67

 ===[ .astype 설명 ]=========================================================================================================
 
[.astype 설명링크]      https://www.askpython.com/python/built-in-methods/python-astype
 
- - - - - - - - - - - - - - - - - - -- - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -- 
 
submissin_set.to_csv('./_data/kaggle_bike/sampleSubmission.csv', index = True)
 # index : 행 인덱스 번호를 쓸지 여부. 기본값(default)은 True입니다.
 
========================================================================================================================
"""   

# 캐글 바이크 문제풀이
import numpy as np
import pandas as pd

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score, mean_squared_error 


#1. 데이터
path = './_data/kaggle_bike/'  

     # 결측치는 없는 것으로 판명                                                               
path = './_data/kaggle_bike/'
train_set = pd.read_csv(path + 'train.csv')
test_set = pd.read_csv(path + 'test.csv')
# print(train_set.shape)  # (10886, 12)
# print(test_set.shape)   # (6493, 9)
# train_set.info() # 데이터 온전한지 확인.

""" [x_train 열 이름]
datetime, season, holiday, workingday, weather, temp, atemp,
humidity, windspeed, casual, registered, count
"""

train_set['datetime'] = pd.to_datetime(train_set['datetime']) 
# [시간 형식의 object 자료형 column을 datetime 형식으로 변경]

# datetime은 날짜와 시간을 나타내는 정보이므로 DTYPE을 datetime으로 변경.
# 세부 날짜별 정보를 보기 위해 날짜 데이터를 년도,월,일, 시간으로 나눈다.

train_set['year'] = train_set['datetime'].dt.year  # 분과 초는 모든값이 0이므로 추가x
train_set['month'] = train_set['datetime'].dt.month
train_set['day'] = train_set['datetime'].dt.day
train_set['hour'] = train_set['datetime'].dt.hour
train_set.drop(['datetime', 'day', 'year'], inplace=True, axis=1)
train_set['month'] = train_set['month'].astype('category')
train_set['hour'] = train_set['hour'].astype('category')
train_set = pd.get_dummies(train_set, columns=['season','weather'])
train_set.drop(['casual', 'registered'], inplace=True, axis=1)
train_set.drop('atemp', inplace=True, axis=1)

print(train_set['month'])


test_set['datetime'] = pd.to_datetime(test_set['datetime'])
test_set['month'] = test_set['datetime'].dt.month
test_set['hour'] = test_set['datetime'].dt.hour
test_set['month'] = test_set['month'].astype('category')
test_set['hour'] = test_set['hour'].astype('category')
test_set = pd.get_dummies(test_set, columns=['season','weather'])
drop_feature = ['datetime', 'atemp']
test_set.drop(drop_feature, inplace=True, axis=1)


print(x.shape) # (10886, 15)
print(y.shape) # (10886, )

# axis 관련 링크 : https://m.blog.naver.com/PostView.naver?isHttpsRedirect=true&blogId=nilsine11202&logNo=221420394534


x = train_set.drop(['casual',  'registered',  'count'], axis=1)  #       axis = 0/1(축값, 0=행, 1=열)                                                                    
#    # (10886, 8)
y = train_set['count']
#    # (10886,)
   



x_train, x_test, y_train, y_test = train_test_split(x, y,
                                                    train_size=0.95,
                                                    shuffle=True,
                                                    random_state=100
                                                    )

#2. 모델구성
model = Sequential()
model.add(Dense(100, activation = 'swish', input_dim=15))
model.add(Dense(100, activation = 'swish'))
model.add(Dense(100, activation = 'swish'))
model.add(Dense(1))

#3. 컴파일, 훈련
model.compile(loss='mae', optimizer='adam')
model.fit(x_train, y_train, epochs=100, batch_size=30)

#4. 결과, 예측
loss = model.evaluate(x_test, y_test)
print('lose : ', loss)

y_predict = model.predict(x_test)

def RMSE(a, b):
    return np.sqrt(mean_squared_error(a, b))

rmse = RMSE(y_test, y_predict)
print("RMSE : ", rmse)


#####################################################


y_summit = model.predict(test_set)
  
      
submissin_set = pd.read_csv(path + 'sampleSubmission.csv', index_col=0)
   # print(submissin_set.shape)   # (6493, 1)

submissin_set['count'] = y_summit
submissin_set['count'] = list(map(abs,submissin_set['count']))
submissin_set.to_csv('./_data/kaggle_bike/sampleSubmission.csv', index = True)
 # index : 행 인덱스 번호를 쓸지 여부. 기본값(default)은 True입니다.      

