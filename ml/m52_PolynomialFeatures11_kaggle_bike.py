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

scaler = MinMaxScaler()
x_train = scaler.fit_transform(x_train)
x_test = scaler.transform(x_test)




#2. 모델
import numpy as np
import pandas as pd
import warnings
warnings.filterwarnings("ignore")

from sklearn.pipeline import make_pipeline
from sklearn.linear_model import LinearRegression, LogisticRegression
model = make_pipeline(StandardScaler(),
                      LogisticRegression())

model.fit(x_train, y_train)

print("그냥 스코어 : ", model.score(x_test, y_test))
# 0.7665382927362877

from sklearn.model_selection import cross_val_score
scores = cross_val_score(model, x_train, y_train, cv=5, scoring='r2')
print("CV : ", scores)
print("CV 엔빵 : ", np.mean(scores))

############################### PolynomialFeatures 후 #######################################
from sklearn.preprocessing import StandardScaler, PolynomialFeatures
pf = PolynomialFeatures(degree=2, include_bias=False)   # include_bias=Falsed 음수로 나오는 것을 방지해줫다..? 정확히는 모르겟음
xp =  pf.fit_transform(x)

# print(xp.shape, )

#2. 모델
model = make_pipeline(StandardScaler(),
                      LinearRegression())



x_train, x_test, y_train, y_test = train_test_split(xp, y,
                                                    train_size=0.8,
                                                    shuffle=True,
                                                    random_state=1234,
                                                    )

model.fit(x_train, y_train)

print("PolynomialFeatures 후 스코어 : ", model.score(x_test, y_test))


# 증폭 전
# 0.7665382927362877

# 증폭 후
# 0.8745129304823926         # 데이터가 좋아서 과적합이 된 것이라고 하심

# 증폭한 데이터의 과적합 정도를 확인
from sklearn.model_selection import cross_val_score  # 스코어가 얼마나 정확한지 검증하는 용도
scores = cross_val_score(model, x_train, y_train, cv=5, scoring='r2')
print("폴리 CV : ", scores)
print("폴리 CV 엔빵 : ", np.mean(scores))


# 그냥 스코어 :  0.022497704315886134
# CV :  [0.20396506 0.16600575 0.1337792  0.14743666 0.17778961]
# CV 엔빵 :  0.165795256879616

# PolynomialFeatures 후 스코어 :  0.5662966427660572
# 폴리 CV :  [0.53806247 0.5594904  0.54411907 0.53935397 0.53246949]
# 폴리 CV 엔빵 :  0.5426990802971431