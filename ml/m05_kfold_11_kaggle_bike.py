"""
[핵심]
print("y의 라벨값 : ", np.unique(y))  # 해당 데이터의 고유값을 출력해준다.

KFold는 회귀와 분류 둘 다 사용할 수 있다.

cross_val_score()을 사용하기 위에서 괄호 안의 옵션인 KFold를 

n_splits = 5  
kfold = KFold(n_splits=n_splits, shuffle=True, random_state=66)

이런 형태로 작성을 한 후에

scores = cross_val_score(model, x, y, cv=kfold)
print("ACC : ", scores, '\n cross_val_score : ', round(np.mean(scores), 4)) 
                                                # round를 사용해서 소수 4번째까지 출력해라 라고함

이러한 형태로 사용을 한다 
이때 round(np.mean(scores), 4))의 의미는 cross_val_score의 값인  score의 값을 평균값으로 변경한 후
np를 이용해서 반올림 과정을 거친 후 소수 4번째 자리까지 출력하라는 의미이다.

[ cross_val_score 사용하는 이유 ]
기존의 validation은 train 데이터에서 손해를 보면서 분할을하여 모델 검증 데이터로 사용을 하였는데
cross_val_score를 사용하면 전체 데이터에서 n_splits 안에 들어있는 변수의 숫자만큼 나누고 fit을 할 때
마다 나눈 부분을 순차적으로 validation 데이터로 사용한다. 이로인해 train 데이터의 손실을 하지않고
validation의 역할을 수행할 수 있다. 만약 n_splits = 5 라면 전체 데이터를 5개로 분할하여 순차적으로
validation 데이터로 사용하게된다.

cross_val_score를 사용하면 train_test_split을 사용할 필요가 없다.
cross_val_score(model, x, y, cv=kfold) 형태로 사용되기 때문이다.
cross_val_score(model, x, y, cv=kfold) 과정에서 fit과 컴파일도 같이 진행된다.


"""

# Kaggle Bike_sharing
import numpy as np
import pandas as pd 
from pandas import DataFrame 
from tensorflow.python.keras.models import Sequential, Model
from tensorflow.python.keras.layers import Dense, Input
from sklearn.model_selection import train_test_split, KFold, cross_val_score, cross_val_predict
from sklearn.metrics import r2_score
from tensorflow.python.keras.callbacks import EarlyStopping
import time
from sklearn.preprocessing import MaxAbsScaler, RobustScaler
from sklearn.preprocessing import MinMaxScaler, StandardScaler
import warnings
warnings.filterwarnings('ignore') # warnig 출력 안함

# 1. 데이터
path = 'D:\study_data\_data\kaggle_bike/'
train_set = pd.read_csv(path+'train.csv')
# print(train_set)
# print(train_set.shape) # (10886, 11)

test_set = pd.read_csv(path+'test.csv')
# print(test_set)
# print(test_set.shape) # (6493, 8)

# datetime 열 내용을 각각 년월일시간날짜로 분리시켜 새 열들로 생성 후 원래 있던 datetime 열을 통째로 drop
train_set["hour"] = [t.hour for t in pd.DatetimeIndex(train_set.datetime)]
train_set["day"] = [t.dayofweek for t in pd.DatetimeIndex(train_set.datetime)]
train_set["month"] = [t.month for t in pd.DatetimeIndex(train_set.datetime)]
train_set['year'] = [t.year for t in pd.DatetimeIndex(train_set.datetime)]
train_set['year'] = train_set['year'].map({2011:0, 2012:1})

test_set["hour"] = [t.hour for t in pd.DatetimeIndex(test_set.datetime)]
test_set["day"] = [t.dayofweek for t in pd.DatetimeIndex(test_set.datetime)]
test_set["month"] = [t.month for t in pd.DatetimeIndex(test_set.datetime)]
test_set['year'] = [t.year for t in pd.DatetimeIndex(test_set.datetime)]
test_set['year'] = test_set['year'].map({2011:0, 2012:1})

train_set.drop('datetime',axis=1,inplace=True) # train_set에서 데이트타임 드랍
test_set.drop('datetime',axis=1,inplace=True) # test_set에서 데이트타임 드랍
train_set.drop('casual',axis=1,inplace=True) # casul 드랍 이유 모르겠음
train_set.drop('registered',axis=1,inplace=True) # registered 드랍 이유 모르겠음

#print(train_set.info())
# null값이 없으므로 결측치 삭제과정 생략

x = train_set.drop(['count'], axis=1)
y = train_set['count']

print(x.shape, y.shape) # (10886, 14) (10886,)

x_train, x_test, y_train, y_test = train_test_split(x, y, train_size=0.8)
scaler = MinMaxScaler()
# scaler = StandardScaler()
# scaler = MaxAbsScaler() 
# scaler = RobustScaler()
scaler.fit(x_train)
scaler.fit(test_set)
test_set = scaler.transform(test_set)
x_train = scaler.transform(x_train)
x_test = scaler.transform(x_test)

n_splits = 5
kfold = KFold(n_splits=n_splits, shuffle=True, random_state=99)

#2. 모델구성
from sklearn.svm import SVR
model = SVR()
    
# 3. 4. 컴파일, 훈련, 평가, 예측
# model.fit(x_train, y_train)
score = cross_val_score(model, x_train, y_train, cv=kfold)
y_predict = cross_val_predict(model, x_test, y_test, cv=kfold)
r2 = r2_score(y_test, y_predict)

print('acc: ', score, '\n cross_val_score: ', round(np.mean(score),4))
print('cross_val_predict r2: ', r2)

# # 5. 제출 준비
# submission = pd.read_csv(path + 'submission.csv', index_col=0)
# y_submit = model.predict(test_set)
# submission['count'] = np.abs(y_submit) # 마이너스 나오는거 절대값 처리

# submission.to_csv(path + 'submission.csv', index=True)

# cross_val_score:  0.3225
# cross_val_predict r2:  0.15622721833692665