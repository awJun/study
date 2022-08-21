# Dacon 따릉이 문제풀이
import pandas as pd
from sklearn.experimental import enable_halving_search_cv
from sklearn.model_selection import train_test_split, KFold,\
    HalvingRandomSearchCV
from sklearn.metrics import r2_score
from sklearn.preprocessing import MaxAbsScaler, RobustScaler
import numpy as np
from sklearn.preprocessing import MinMaxScaler, StandardScaler
import warnings
warnings.filterwarnings('ignore') # warnig 출력 안함


# 1. 데이터
path = './_data/ddarung/'
train_set = pd.read_csv(path+'train.csv',index_col=0) # index_col = n : n번째 칼럼을 인덱스로 인식
# print(train_set)
# print(train_set.shape) # (1459, 10)

test_set = pd.read_csv(path+'test.csv', index_col=0)
# print(test_set)
# print(test_set.shape) # (715, 9)

### 결측치 처리(일단 제거로 처리) ###
print(train_set.info())
print(train_set.isnull().sum()) # 결측치 전부 더함
# train_set = train_set.dropna() # nan 값(결측치) 열 없앰
train_set = train_set.fillna(0) # 결측치 0으로 채움
print(train_set.isnull().sum()) # 없어졌는지 재확인

x = train_set.drop(['count'], axis=1) # axis = 0은 열방향으로 쭉 한줄(가로로 쭉), 1은 행방향으로 쭉 한줄(세로로 쭉)
y = train_set['count']

print(x.shape, y.shape) # (1328, 9) (1328,)

allfeature = round(x.shape[1]*0.2, 0)
print('자를 갯수: ', int(allfeature))


x_train, x_test, y_train, y_test = train_test_split(x, y, shuffle=True, train_size=0.8, random_state=1234)

#.2 모델
###[ 이번 핵심! ]####################################
from sklearn.ensemble import BaggingClassifier, BaggingRegressor
from sklearn.tree import DecisionTreeClassifier
from sklearn.linear_model import LinearRegression  # 2진 분류에서 사용   /  linear_model은 제일 간단한 모델임
from xgboost import XGBClassifier, XGBRegressor

model = BaggingRegressor(XGBRegressor(n_estimators=100,
                          learning_rate=1,
                          max_depth=2,
                          gamma=0,
                          min_child_weight=1,
                          subsample=1,
                          colsample_bytree=0.5,
                          colsample_bylevel=1,
                          colsample_bynode=1,
                          reg_alpha=0.01,
                          reg_lambd=1,
                          tree_method='gpu_hist', predictor='gpu_predictor', gpu_id=0, random_state=1234,
                          ),
                          verbose=1,
                          n_estimators=100,    # 생성할 의사결정 나무 개수
                          n_jobs= -1,
                          random_state=123,
                          )

######################################################

#.3 훈련
model.fit(x_train, y_train)
# BaggingClassifier은 스케일링을 하지 않으면 에러가 발생한다.

#.4 평가, 예측
print(model.score(x_test, y_test))



# LinearRegression의 결과
# 0.6027687282947763

# XGBRegressor의 결과
# 0.7759891252563081