"""
[핵심]
scaler = [MinMaxScaler, MaxAbsScaler, StandardScaler, RobustScaler, QuantileTransformer, PowerTransformer] 
위에  스케일러들 테스트임

PowerTransformer(method='yeo_johnson')   QuantileTransformer(method='BOX_COX')   메소드를 넣으면 에러 발생해서 일단 빼고 돌렸음 ..ㅠ

"""

import pandas as pd 
import numpy as np
from sklearn.experimental import enable_halving_search_cv
from sklearn.model_selection import train_test_split, KFold,\
    HalvingRandomSearchCV, RandomizedSearchCV
from sklearn.metrics import r2_score
import time
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from sklearn.feature_selection import SelectFromModel
from xgboost import XGBRegressor
import warnings
warnings.filterwarnings('ignore') # warnig 출력 안함

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
x = np.array(x)

print(x.shape, y.shape) # (10886, 14) (10886,)

x_train, x_test, y_train, y_test = train_test_split(x, y, shuffle=True, train_size=0.8, random_state=1234)


###[ 핵심 ]################################################################################################################

from sklearn.model_selection import train_test_split
import numpy as np
import pandas as pd
import warnings
warnings.filterwarnings("ignore")
from sklearn.linear_model import LinearRegression
from sklearn.pipeline import make_pipeline

from sklearn.preprocessing import MinMaxScaler, MaxAbsScaler, StandardScaler, RobustScaler, QuantileTransformer, PowerTransformer
from sklearn.metrics import r2_score, accuracy_score
from sklearn.ensemble import RandomForestRegressor

scaler = [MinMaxScaler, MaxAbsScaler, StandardScaler, RobustScaler, QuantileTransformer, PowerTransformer]


for scalers in scaler:
    scalers = scalers()
    scaler_name = str(scalers).strip('()')  # .strip('()')참고 https://ai-youngjun.tistory.com/68
    x_train = scalers.fit_transform(x_train)
    x_test = scalers.transform(x_test)
    
    model = RandomForestRegressor()
    model.fit(x_train, y_train)
    
    y_predict = model.predict(x_test)
    results = r2_score(y_test, y_predict)
    print(scaler_name + "의 결과 : ", round(results, 4))
    
    
    
# MinMaxScaler의 결과 :  0.9535
# MaxAbsScaler의 결과 :  0.9544
# StandardScaler의 결과 :  0.9538
# RobustScaler의 결과 :  0.953
# QuantileTransformer의 결과 :  0.9525
# PowerTransformer의 결과 :  0.9549