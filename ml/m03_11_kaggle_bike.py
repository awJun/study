"""
[핵심]
각 모델 성능 테스트  (해당 모델들은 회기 모델들이다.)
model_1 = LinearSVR()
model_2 = Perceptron()
model_3 = LinearRegression()
model_4 = KNeighborsRegressor()
model_5 = DecisionTreeRegressor()
model_6 = RandomForestRegressor()

neighbors, tree, ensemble 모델 정리
from sklearn.neighbors   # 이웃하는 ..? 검색하자
from sklearn.tree        # 더 좋은 것을 아래로 뿌리를 내린다(가지치기) 결정나무
from sklearn.ensemble    # 같이 넣었을 때 더 좋은 것을 캐치

[중요]
LogisticRegression
- 이것은 Regression들어가지만 분류 모델이다.
- LinearRegression 이친구가 회기 모델이다.
이 친구 빼고는 나머지는 다 Regression이 들어가면 회기 모델로 생각하면 된다.

Classifier가 들어가면 분류 모델로 생각하면 된다.

"""

# Kaggle Bike_sharing
import numpy as np
import pandas as pd 
from pandas import DataFrame 
from tensorflow.python.keras.models import Sequential, Model
from tensorflow.python.keras.layers import Dense, Input
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score
from tensorflow.python.keras.callbacks import EarlyStopping
import time
from sklearn.preprocessing import MaxAbsScaler, RobustScaler
from sklearn.preprocessing import MinMaxScaler, StandardScaler

# 1. 데이터
path = './_data/kaggle_bike/' 
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

#2. 모델구성
from sklearn.svm import LinearSVR, SVR
from sklearn.linear_model import Perceptron
from sklearn.linear_model import LogisticRegression, LinearRegression
from sklearn.neighbors import KNeighborsClassifier, KNeighborsRegressor
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor

models = [LinearSVR, SVR, Perceptron, LinearRegression, KNeighborsRegressor, DecisionTreeRegressor, RandomForestRegressor]

for model in models:
    model = model()
    model_name = str(model).strip('()')
    model.fit(x_train, y_train)
    result = model.score(x_test, y_test)
    print(model_name, '결과: ', result)

# # 5. 제출 준비
# submission = pd.read_csv(path + 'submission.csv', index_col=0)
# y_submit = model.predict(test_set)
# submission['count'] = np.abs(y_submit) # 마이너스 나오는거 절대값 처리

# submission.to_csv(path + 'submission.csv', index=True)

# LinearSVR 결과:  0.3430750218254607
# SVR 결과:  0.36738403385941565
# Perceptron 결과:  0.004132231404958678
# LinearRegression 결과:  0.4040664143817021
# KNeighborsRegressor 결과:  0.6829281156262785
# DecisionTreeRegressor 결과:  0.9111338017114592
# RandomForestRegressor 결과:  0.9525089430797721








